import {
    omit, pick, throttle, uniqBy,
} from 'lodash';
import { Filter, ObjectId } from 'mongodb';
import {
    ContestNotFoundError, HackRejudgeFailedError,
    NotFoundError, PermissionError, PretestRejudgeFailedError, ProblemConfigError,
    ProblemNotFoundError, RecordNotFoundError, UserNotFoundError,
} from '../error';
import { RecordDoc, Tdoc } from '../interface';
import { PERM, PRIV, STATUS } from '../model/builtin';
import * as contest from '../model/contest';
import problem, { ProblemDoc } from '../model/problem';
import record from '../model/record';
import { langs } from '../model/setting';
import storage from '../model/storage';
import system from '../model/system';
import TaskModel from '../model/task';
import user from '../model/user';
import {
    ConnectionHandler, param, subscribe, Types,
} from '../service/server';
import { buildProjection, streamToBuffer, Time } from '../utils';
import { ContestDetailBaseHandler } from './contest';
import { postJudge } from './judge';

type GPUProfileView = 'overview' | 'section' | 'rules' | 'metrics' | 'metric' | 'source' | 'code';
const GPU_PROFILE_VIEWS: GPUProfileView[] = ['overview', 'section', 'rules', 'metrics', 'metric', 'source', 'code'];

const GPU_PROFILE_PAGE_SIZE = {
    section: 100,
    metrics: 50,
    instances: 100,
    source: 400,
};
const GPU_PROFILE_TEXT_PAGE_CHARS = 128 * 1024;

function profileArray(value: any): any[] {
    return Array.isArray(value) ? value : [];
}

function profilePage<T>(items: T[], requestedPage: number, pageSize: number) {
    const totalItems = items.length;
    const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
    const page = Math.min(Math.max(requestedPage || 1, 1), totalPages);
    const offset = (page - 1) * pageSize;
    return {
        items: items.slice(offset, offset + pageSize),
        page,
        totalItems,
        totalPages,
        from: totalItems ? offset + 1 : 0,
        to: Math.min(offset + pageSize, totalItems),
    };
}

function profileTextPage(text: string, requestedPage: number) {
    const lines = (text || '').split('\n');
    const chunks: Array<{ from: number, to: number, text: string }> = [];
    let current: string[] = [];
    let currentChars = 0;
    let currentFrom = 1;
    const flush = (to: number) => {
        if (!current.length) return;
        chunks.push({ from: currentFrom, to, text: current.join('\n') });
        current = [];
        currentChars = 0;
    };
    for (let index = 0; index < lines.length; index++) {
        const line = lines[index];
        if (line.length > GPU_PROFILE_TEXT_PAGE_CHARS) {
            flush(index);
            for (let offset = 0; offset < line.length; offset += GPU_PROFILE_TEXT_PAGE_CHARS) {
                chunks.push({
                    from: index + 1,
                    to: index + 1,
                    text: line.slice(offset, offset + GPU_PROFILE_TEXT_PAGE_CHARS),
                });
            }
            currentFrom = index + 2;
            continue;
        }
        const nextChars = currentChars + line.length + (current.length ? 1 : 0);
        if (current.length && (current.length >= GPU_PROFILE_PAGE_SIZE.source
            || nextChars > GPU_PROFILE_TEXT_PAGE_CHARS)) {
            flush(index);
            currentFrom = index + 1;
        }
        current.push(line);
        currentChars += line.length + (current.length > 1 ? 1 : 0);
    }
    flush(lines.length);
    const totalPages = Math.max(chunks.length, 1);
    const page = Math.min(Math.max(requestedPage || 1, 1), totalPages);
    const selected = chunks[page - 1] || { from: 0, to: 0, text: '' };
    return {
        items: [],
        page,
        totalItems: lines.length,
        totalPages,
        from: selected.from,
        to: selected.to,
        text: selected.text,
    };
}

function cleanNCUText(value: any) {
    if (typeof value !== 'string') return value || '';
    return value
        .replace(/@section:[^:]+:([^@]+)@/g, '$1')
        .replace(/@url:([^:]+):[^@]+@/g, '$1')
        .replace(/@metric:[^:]+:([^@]+)@/g, '$1');
}

function compactProfileRule(rule: any) {
    const message = rule?.rule_message || {};
    const speedup = rule?.speedup_estimation?.speedup;
    return {
        name: message.title || rule?.name || 'Recommendation',
        category: message.title && rule?.name && message.title !== rule.name ? rule.name : '',
        message: cleanNCUText(message.message),
        estimatedSpeedup: typeof speedup === 'number' ? speedup : null,
        focusMetrics: profileArray(rule?.focus_metrics).map((metric) => ({
            name: metric?.name || '',
            value: metric?.value,
            info: cleanNCUText(metric?.info),
        })),
    };
}

function compactProfileAction(action: any, position: number) {
    return {
        position,
        name: action?.name || `Kernel ${position}`,
        device: action?.device,
        grid: profileArray(action?.grid),
        block: profileArray(action?.block),
        achievedOccupancyPct: action?.achievedOccupancyPct,
        theoreticalOccupancyPct: action?.theoreticalOccupancyPct,
        metricCount: Number(action?.metricCount) || profileArray(action?.metrics).length,
        ruleCount: profileArray(action?.rules).length,
        sourceFileCount: profileArray(action?.sourceFiles).length,
        sourceMarkerCount: profileArray(action?.sourceMarkers).length,
    };
}

function profileSourceLabel(path: string, position: number) {
    if (!path) return `Source ${position}`;
    if (path.startsWith('/work/')) return path.slice('/work/'.length);
    const includeAt = path.lastIndexOf('/include/');
    if (includeAt >= 0) return `CUDA include/${path.slice(includeAt + '/include/'.length)}`;
    return path.split('/').filter(Boolean).slice(-2).join('/');
}

export class RecordListHandler extends ContestDetailBaseHandler {
    @param('page', Types.PositiveInt, true)
    @param('pid', Types.ProblemId, true)
    @param('tid', Types.ObjectId, true)
    @param('uidOrName', Types.UidOrName, true)
    @param('lang', Types.String, true)
    @param('status', Types.Int, true)
    @param('fullStatus', Types.Boolean)
    @param('all', Types.Boolean)
    @param('allDomain', Types.Boolean)
    @param('stat', Types.Boolean)
    async get(
        domainId: string, page = 1, pid?: string | number, tid?: ObjectId,
        uidOrName?: string, lang?: string, status?: number, full = false,
        all = false, allDomain = false, stat = false,
    ) {
        const notification = [];
        let tdoc = null;
        let invalid = false;
        this.response.template = 'record_main.html';
        const q: Filter<RecordDoc> = { contest: tid };
        if (full) uidOrName = this.user._id.toString();
        if (uidOrName) {
            const udoc = await user.getById(domainId, +uidOrName)
                || await user.getByUname(domainId, uidOrName)
                || await user.getByEmail(domainId, uidOrName);
            if (udoc) q.uid = udoc._id;
            else invalid = true;
        }
        if (q.uid !== this.user._id) this.checkPerm(PERM.PERM_VIEW_RECORD);
        if (tid) {
            tdoc = await contest.get(domainId, tid);
            this.tdoc = tdoc;
            if (!tdoc) throw new ContestNotFoundError(domainId, pid);
            if (!contest.canShowScoreboard.call(this, tdoc, true)) throw new PermissionError(PERM.PERM_VIEW_CONTEST_HIDDEN_SCOREBOARD);
            if (!contest[q.uid === this.user._id ? 'canShowSelfRecord' : 'canShowRecord'].call(this, tdoc, true)) {
                throw new PermissionError(PERM.PERM_VIEW_CONTEST_HIDDEN_SCOREBOARD);
            }
            if (!(await contest.getStatus(domainId, tid, this.user._id))?.attend) {
                const name = tdoc.rule === 'homework'
                    ? "You haven't claimed this homework yet."
                    : "You haven't attended this contest yet.";
                notification.push({ name, args: { type: 'note' }, checker: () => true });
            }
        }
        if (pid) {
            if (typeof pid === 'string' && tdoc && /^[A-Z]$/.test(pid)) {
                pid = tdoc.pids[Number.parseInt(pid, 36) - 10];
            }
            const pdoc = await problem.get(domainId, pid);
            if (pdoc) q.pid = pdoc.docId;
            else invalid = true;
        }
        if (lang) q.lang = lang;
        if (typeof status === 'number') q.status = status;
        if (all) {
            this.checkPerm(PERM.PERM_VIEW_CONTEST_HIDDEN_SCOREBOARD);
            this.checkPerm(PERM.PERM_VIEW_HOMEWORK_HIDDEN_SCOREBOARD);
            delete q.contest;
        }
        if (allDomain) {
            this.checkPriv(PRIV.PRIV_MANAGE_ALL_DOMAIN);
            delete q.contest;
            q._id = { $gt: Time.getObjectID(new Date(Date.now() - 10 * Time.week)) };
        }
        let cursor = record.getMulti(allDomain ? '' : domainId, q).sort('_id', -1);
        if (!full) cursor = cursor.project(buildProjection(record.PROJECTION_LIST));
        const limit = full ? 10 : system.get('pagination.record');
        let rdocs = invalid
            ? [] as RecordDoc[]
            : await cursor.skip((page - 1) * limit).limit(limit).toArray();
        const canViewHiddenProblem = this.user.hasPerm(PERM.PERM_VIEW_PROBLEM_HIDDEN) || this.user._id;
        const [udict, pdict] = full ? [{}, {}]
            : await Promise.all([
                user.getList(domainId, rdocs.map((rdoc) => rdoc.uid)),
                tid
                    ? problem.getList(domainId, rdocs.map((rdoc) => rdoc.pid), true, false, problem.PROJECTION_CONTEST_LIST)
                    : this.user.hasPerm(PERM.PERM_VIEW_PROBLEM)
                        ? problem.getList(domainId, rdocs.map((rdoc) => rdoc.pid), canViewHiddenProblem, false, problem.PROJECTION_LIST)
                        : Object.fromEntries(uniqBy(rdocs, 'pid').map((rdoc) => [rdoc.pid, { ...problem.default, pid: rdoc.pid }])),
            ]);
        if (this.tdoc && !this.user.own(this.tdoc) && !this.user.hasPerm(PERM.PERM_EDIT_CONTEST)) {
            rdocs = rdocs.map((i) => contest.applyProjection(tdoc, i, this.user));
        }
        this.response.body = {
            page,
            rdocs,
            tdoc,
            pdict,
            udict,
            all,
            allDomain,
            filterPid: pid,
            filterTid: tid,
            filterUidOrName: uidOrName,
            filterLang: lang,
            filterStatus: status,
            notification,
        };
        if (this.user.hasPriv(PRIV.PRIV_VIEW_JUDGE_STATISTICS) && stat) {
            this.response.body.statistics = await record.stat(allDomain ? undefined : domainId);
        }
    }
}

export class RecordDetailHandler extends ContestDetailBaseHandler {
    rdoc: RecordDoc;

    @param('rid', Types.ObjectId)
    async prepare(domainId: string, rid: ObjectId) {
        this.rdoc = await record.get(domainId, rid);
        if (!this.rdoc) throw new RecordNotFoundError(rid);
        if (this.rdoc.uid !== this.user._id) this.checkPerm(PERM.PERM_VIEW_RECORD);
    }

    async download() {
        for (const file of ['code', 'hack']) {
            if (!this.rdoc.files?.[file]) continue;
            const [id, filename] = this.rdoc.files?.[file]?.split('#') || [];
            // eslint-disable-next-line no-await-in-loop
            this.response.redirect = await storage.signDownloadLink(`submission/${id}`, filename || file, true, 'user');
            return;
        }
        const lang = langs[this.rdoc.lang]?.pretest || this.rdoc.lang;
        this.response.body = this.rdoc.code;
        this.response.type = 'text/plain';
        this.response.disposition = `attachment; filename="${langs[lang]?.code_file || `foo.${this.rdoc.lang}`}"`;
    }

    @param('rid', Types.ObjectId)
    @param('download', Types.Boolean)
    @param('rev', Types.ObjectId, true)
    // eslint-disable-next-line consistent-return
    async get(domainId: string, rid: ObjectId, download = false, rev?: ObjectId) {
        let rdoc = this.rdoc;
        const allRev = await record.collHistory.find({ rid }).project({ _id: 1, judgeAt: 1 }).sort({ _id: -1 }).toArray();
        const allRevs: Record<string, Date> = Object.fromEntries(allRev.map((i) => [i._id.toString(), i.judgeAt]));
        if (rev && allRevs[rev.toString()]) {
            rdoc = { ...rdoc, ...omit(await record.collHistory.findOne({ _id: rev }), ['_id']), progress: null };
        }
        let canViewDetail = true;
        if (rdoc.contest?.toString().startsWith('0'.repeat(23))) {
            if (rdoc.uid !== this.user._id) throw new PermissionError(PERM.PERM_READ_RECORD_CODE);
        } else if (rdoc.contest) {
            this.tdoc = await contest.get(domainId, rdoc.contest);
            let canView = this.user.own(this.tdoc);
            canView ||= contest.canShowRecord.call(this, this.tdoc);
            canView ||= contest.canShowSelfRecord.call(this, this.tdoc, true) && rdoc.uid === this.user._id;
            if (!canView && rdoc.uid !== this.user._id) throw new PermissionError(rid);
            canViewDetail = canView;
            this.args.tid = this.tdoc.docId;
            if (!this.user.own(this.tdoc) && !this.user.hasPerm(PERM.PERM_EDIT_CONTEST)) {
                rdoc = contest.applyProjection(this.tdoc, rdoc, this.user);
                this.rdoc = rdoc;
            }
        }

        // eslint-disable-next-line prefer-const
        let [pdoc, self, udoc] = await Promise.all([
            problem.get(rdoc.domainId, rdoc.pid, problem.PROJECTION_LIST.concat('config')),
            problem.getStatus(domainId, rdoc.pid, this.user._id),
            user.getById(domainId, rdoc.uid),
        ]);

        let canViewCode = rdoc.uid === this.user._id;
        canViewCode ||= this.user.hasPriv(PRIV.PRIV_READ_RECORD_CODE);
        canViewCode ||= this.user.hasPerm(PERM.PERM_READ_RECORD_CODE);
        canViewCode ||= this.user.hasPerm(PERM.PERM_READ_RECORD_CODE_ACCEPT) && self?.status === STATUS.STATUS_ACCEPTED;
        if (this.tdoc) {
            this.tsdoc = await contest.getStatus(domainId, this.tdoc.docId, this.user._id);
            canViewCode ||= this.user.own(this.tdoc);
            if (this.tdoc.allowViewCode && contest.isDone(this.tdoc)) {
                canViewCode ||= !!this.tsdoc?.attend;
            }
            if (!this.tsdoc?.attend && pdoc && !problem.canViewBy(pdoc, this.user)) throw new PermissionError(PERM.PERM_VIEW_PROBLEM_HIDDEN);
        } else if (pdoc && !problem.canViewBy(pdoc, this.user)) throw new PermissionError(PERM.PERM_VIEW_PROBLEM_HIDDEN);
        if (!canViewCode) {
            rdoc.code = '';
            rdoc.files = {};
            rdoc.compilerTexts = [];
        } else if (download) return await this.download();
        this.response.template = 'record_detail.html';
        this.response.body = {
            udoc, rdoc: canViewDetail ? rdoc : pick(rdoc, ['_id', 'lang', 'code']), pdoc, tdoc: this.tdoc, rev, allRevs,
        };
    }

    @param('rid', Types.ObjectId)
    async post() {
        this.checkPerm(PERM.PERM_REJUDGE);
        if (this.rdoc.files?.hack) throw new HackRejudgeFailedError();
        if (this.rdoc.contest?.toString().startsWith('0'.repeat(23))) throw new PretestRejudgeFailedError();
    }

    @param('rid', Types.ObjectId)
    async postRejudge(domainId: string, rid: ObjectId) {
        const pdoc = await problem.get(domainId, this.rdoc.pid);
        if (!pdoc?.config || typeof pdoc.config === 'string') throw new ProblemConfigError();
        const priority = await record.submissionPriority(this.user._id, -20);
        const rdoc = await record.reset(domainId, rid, true);
        this.ctx.broadcast('record/change', rdoc);
        await record.judge(domainId, rid, priority, this.rdoc.contest ? { detail: false } : {});
        this.back();
    }

    @param('rid', Types.ObjectId)
    async postCancel(domainId: string, rid: ObjectId) {
        const $set = {
            status: STATUS.STATUS_CANCELED,
            score: 0,
            time: 0,
            memory: 0,
            testCases: [{
                id: 0, subtaskId: 0, status: 9, score: 0, time: 0, memory: 0, message: 'score canceled',
            }],
            subtasks: {},
        };
        const [latest] = await Promise.all([
            record.update(domainId, rid, $set),
            TaskModel.deleteMany({ rid: this.rdoc._id }),
        ]);
        if (latest) {
            this.ctx.broadcast('record/change', latest);
            await postJudge(latest);
        }
        this.back();
    }
}

export class RecordGPUProfileHandler extends RecordDetailHandler {
    @param('rid', Types.ObjectId)
    @param('caseId', Types.PositiveInt)
    @param('download', Types.Boolean)
    @param('rev', Types.ObjectId, true)
    @param('view', Types.Range(GPU_PROFILE_VIEWS), true)
    @param('action', Types.PositiveInt, true)
    @param('section', Types.PositiveInt, true)
    @param('metric', Types.PositiveInt, true)
    @param('file', Types.PositiveInt, true)
    @param('page', Types.PositiveInt, true)
    async get(domainId: string, rid: ObjectId, ...args: any[]) {
        const [
            caseId, download = false, rev, view = 'overview', actionPosition = 1,
            sectionPosition = 1, metricPosition = 1, filePosition = 1, requestedPage = 1,
        ] = args as [number, boolean?, ObjectId?, GPUProfileView?, number?, number?, number?, number?, number?];
        await super.get(domainId, rid, false, rev);
        const detail = this.response.body as {
            rdoc: RecordDoc;
            [key: string]: any;
        };
        const profile = detail.rdoc.gpuProfiles?.[caseId];
        if (!profile) throw new NotFoundError(`GPU profile for testcase ${caseId} not found`);

        const profileUrl = (query: Record<string, any> = {}) => this.url('record_gpu_profile', {
            rid,
            caseId,
            query: { ...(rev ? { rev } : {}), ...query },
        });

        if (download) {
            if (profile.status !== 'ready' || !profile.reportPath) {
                throw new NotFoundError(`GPU profile report for testcase ${caseId} is not ready`);
            }
            this.response.redirect = await storage.signDownloadLink(
                profile.reportPath,
                `hydro-${rid.toHexString()}-case-${caseId}.ncu-rep`,
                false,
                'user',
            );
            return;
        }

        let summary = null;
        let summaryError = '';
        if (profile.status === 'ready') {
            try {
                if (!profile.summaryPath) throw new Error('Profile summary artifact is missing.');
                summary = JSON.parse((await streamToBuffer(await storage.get(profile.summaryPath))).toString('utf8'));
            } catch (error) {
                summaryError = error.message || 'Unable to load the profile summary.';
            }
        }

        const pagination = (result: ReturnType<typeof profilePage>, urlForPage: (page: number) => string) => {
            const pageNumbers = [1, result.page - 2, result.page - 1, result.page,
                result.page + 1, result.page + 2, result.totalPages]
                .filter((item, index, values) => item >= 1 && item <= result.totalPages && values.indexOf(item) === index)
                .sort((left, right) => left - right);
            return {
                page: result.page,
                totalItems: result.totalItems,
                totalPages: result.totalPages,
                from: result.from,
                to: result.to,
                previousUrl: result.page > 1 ? urlForPage(result.page - 1) : '',
                nextUrl: result.page < result.totalPages ? urlForPage(result.page + 1) : '',
                pages: pageNumbers.map((page) => ({ page, url: urlForPage(page) })),
            };
        };

        let profileSummary = null;
        let viewData = null;
        let actionNavigation = [];
        const summaryWarnings = [];
        if (summary) {
            const rawActions = profileArray(summary.ranges)
                .flatMap((reportRange) => profileArray(reportRange?.actions));
            const rawSections = profileArray(summary.detailSections)
                .filter((section) => profileArray(section?.items).length);
            const compactActions = rawActions.map((action, index) => compactProfileAction(action, index + 1));
            actionNavigation = compactActions.map((action) => ({
                ...action,
                rulesUrl: action.ruleCount ? profileUrl({ view: 'rules', action: action.position }) : '',
                metricsUrl: profileUrl({ view: 'metrics', action: action.position }),
                sourceUrl: action.sourceFileCount || action.sourceMarkerCount
                    ? profileUrl({ view: 'source', action: action.position }) : '',
            }));
            summaryWarnings.push(...profileArray(summary.warnings).filter((warning) => typeof warning === 'string'));

            profileSummary = {
                ncuVersion: summary.ncuVersion || profile.ncuVersion,
                set: summary.set || profile.set,
                actions: actionNavigation,
                sections: rawSections.map((section, index) => ({
                    position: index + 1,
                    name: section?.sectionName || 'Other',
                    kernelName: section?.kernelName || '',
                    itemCount: profileArray(section?.items).length,
                    ruleCount: profileArray(section?.rules).length,
                    url: profileUrl({ view: 'section', section: index + 1 }),
                })),
                codeUrl: summary.sourceOutput ? profileUrl({ view: 'code' }) : '',
            };

            const selectedRawAction = rawActions[actionPosition - 1];
            const selectedAction = compactActions[actionPosition - 1];
            if (view === 'section') {
                const selectedSection = rawSections[sectionPosition - 1];
                if (!selectedSection) throw new NotFoundError(`GPU profile section ${sectionPosition} not found`);
                const sectionPage = profilePage(
                    profileArray(selectedSection.items), requestedPage, GPU_PROFILE_PAGE_SIZE.section,
                );
                viewData = {
                    position: sectionPosition,
                    totalSections: rawSections.length,
                    name: selectedSection.sectionName || 'Other',
                    kernelName: selectedSection.kernelName || '',
                    items: sectionPage.items.map((item) => ({
                        bodyItem: item?.bodyItem || '',
                        label: item?.label || item?.name || '',
                        name: item?.name || '',
                        value: item?.value,
                        unit: item?.unit || '',
                    })),
                    rules: profileArray(selectedSection.rules).map((rule) => ({
                        name: rule?.name || 'Recommendation',
                        description: cleanNCUText(rule?.description),
                        estimatedSpeedup: rule?.estimatedSpeedup || '',
                    })),
                    previousSectionUrl: sectionPosition > 1
                        ? profileUrl({ view: 'section', section: sectionPosition - 1 }) : '',
                    nextSectionUrl: sectionPosition < rawSections.length
                        ? profileUrl({ view: 'section', section: sectionPosition + 1 }) : '',
                    pagination: pagination(sectionPage, (page) => profileUrl({
                        view: 'section', section: sectionPosition, page,
                    })),
                };
            } else if (view === 'rules') {
                if (!selectedRawAction) throw new NotFoundError(`GPU profile kernel ${actionPosition} not found`);
                viewData = {
                    action: selectedAction,
                    rules: profileArray(selectedRawAction.rules).map(compactProfileRule),
                };
            } else if (view === 'metrics') {
                if (!selectedRawAction) throw new NotFoundError(`GPU profile kernel ${actionPosition} not found`);
                const metricsPage = profilePage(
                    profileArray(selectedRawAction.metrics), requestedPage, GPU_PROFILE_PAGE_SIZE.metrics,
                );
                viewData = {
                    action: selectedAction,
                    metrics: metricsPage.items.map((metric, index) => {
                        const position = metricsPage.from + index;
                        const storedInstances = profileArray(metric?.instances).length;
                        return {
                            position,
                            name: metric?.name || '',
                            value: metric?.value,
                            unit: metric?.unit || '',
                            description: metric?.description || '',
                            instanceCount: Number(metric?.instanceCount) || 0,
                            storedInstances,
                            instancesTruncated: !!metric?.instancesTruncated,
                            detailUrl: storedInstances ? profileUrl({
                                view: 'metric', action: actionPosition, metric: position,
                            }) : '',
                        };
                    }),
                    pagination: pagination(metricsPage, (page) => profileUrl({
                        view: 'metrics', action: actionPosition, page,
                    })),
                };
            } else if (view === 'metric') {
                if (!selectedRawAction) throw new NotFoundError(`GPU profile kernel ${actionPosition} not found`);
                const selectedMetric = profileArray(selectedRawAction.metrics)[metricPosition - 1];
                if (!selectedMetric) throw new NotFoundError(`GPU profile metric ${metricPosition} not found`);
                const instancesPage = profilePage(
                    profileArray(selectedMetric.instances), requestedPage, GPU_PROFILE_PAGE_SIZE.instances,
                );
                viewData = {
                    action: selectedAction,
                    metric: {
                        position: metricPosition,
                        name: selectedMetric.name || '',
                        value: selectedMetric.value,
                        unit: selectedMetric.unit || '',
                        description: selectedMetric.description || '',
                        instanceCount: Number(selectedMetric.instanceCount) || 0,
                        storedInstanceCount: profileArray(selectedMetric.instances).length,
                        instancesTruncated: !!selectedMetric.instancesTruncated,
                        instances: instancesPage.items.map((instance, index) => ({
                            index: instance?.index ?? instancesPage.from + index - 1,
                            value: instance?.value,
                        })),
                    },
                    metricsUrl: profileUrl({ view: 'metrics', action: actionPosition }),
                    pagination: pagination(instancesPage, (page) => profileUrl({
                        view: 'metric', action: actionPosition, metric: metricPosition, page,
                    })),
                };
            } else if (view === 'source') {
                if (!selectedRawAction) throw new NotFoundError(`GPU profile kernel ${actionPosition} not found`);
                const sourceFiles = profileArray(selectedRawAction.sourceFiles);
                const selectedFile = sourceFiles[filePosition - 1];
                if (sourceFiles.length && !selectedFile) throw new NotFoundError(`GPU profile source file ${filePosition} not found`);
                const sourcePage = profileTextPage(selectedFile?.content || '', requestedPage);
                viewData = {
                    action: selectedAction,
                    files: sourceFiles.map((file, index) => ({
                        position: index + 1,
                        label: profileSourceLabel(file?.path || '', index + 1),
                        url: profileUrl({ view: 'source', action: actionPosition, file: index + 1 }),
                    })),
                    file: selectedFile ? {
                        position: filePosition,
                        label: profileSourceLabel(selectedFile.path || '', filePosition),
                        text: sourcePage.text,
                    } : null,
                    markers: profileArray(selectedRawAction.sourceMarkers).map((marker) => {
                        const location = marker?.source_location;
                        const fileLabel = location?.file_name ? profileSourceLabel(location.file_name, 1) : '';
                        return {
                            location: fileLabel && location?.line ? `${fileLabel}:${location.line}` : fileLabel,
                            message: cleanNCUText(marker?.message),
                        };
                    }).filter((marker) => marker.message || marker.location),
                    pagination: pagination(sourcePage, (page) => profileUrl({
                        view: 'source', action: actionPosition, file: filePosition, page,
                    })),
                };
            } else if (view === 'code') {
                if (!summary.sourceOutput) throw new NotFoundError('CUDA, PTX and SASS output not found');
                const codePage = profileTextPage(summary.sourceOutput, requestedPage);
                viewData = {
                    text: codePage.text,
                    pagination: pagination(codePage, (page) => profileUrl({ view: 'code', page })),
                };
            }
        }
        this.response.template = 'gpu_profile.html';
        this.response.body = {
            ...detail,
            caseId,
            profile,
            profileSummary,
            view,
            viewData,
            actionNavigation,
            warnings: uniqBy([
                ...profileArray(profile.warnings),
                ...summaryWarnings,
            ].filter((warning) => typeof warning === 'string'), (warning) => warning),
            urls: {
                overview: profileUrl(),
                record: this.url('record_detail', { rid, query: rev ? { rev } : {} }),
                download: profile.status === 'ready' && profile.reportPath ? profileUrl({ download: true }) : '',
            },
            summaryError,
            rev,
        };
    }
}

export class RecordMainConnectionHandler extends ConnectionHandler {
    all = false;
    allDomain = false;
    tid: string;
    uid: number;
    pid: number;
    status: number;
    pretest = false;
    tdoc: Tdoc;
    applyProjection = false;
    noTemplate = false;
    queue: Map<string, () => Promise<any>> = new Map();
    throttleQueueClear: () => void;

    @param('tid', Types.ObjectId, true)
    @param('pid', Types.ProblemId, true)
    @param('uidOrName', Types.UidOrName, true)
    @param('status', Types.Int, true)
    @param('pretest', Types.Boolean)
    @param('all', Types.Boolean)
    @param('allDomain', Types.Boolean)
    @param('noTemplate', Types.Boolean, true)
    async prepare(
        domainId: string, tid?: ObjectId, pid?: string | number, uidOrName?: string,
        status?: number, pretest = false, all = false, allDomain = false, noTemplate = false,
    ) {
        if (tid) {
            this.tdoc = await contest.get(domainId, tid);
            if (!this.tdoc) throw new ContestNotFoundError(domainId, tid);
            if (pretest || contest.canShowScoreboard.call(this, this.tdoc, true)) this.tid = tid.toHexString();
            else throw new PermissionError(PERM.PERM_VIEW_CONTEST_HIDDEN_SCOREBOARD);
            if (!this.user.own(this.tdoc) && !this.user.hasPerm(PERM.PERM_EDIT_CONTEST)) {
                this.applyProjection = true;
            }
        }
        if (pretest) {
            this.pretest = true;
            this.uid = this.user._id;
        } else if (uidOrName) {
            let udoc = await user.getById(domainId, +uidOrName);
            if (udoc) this.uid = udoc._id;
            else {
                udoc = await user.getByUname(domainId, uidOrName);
                if (udoc) this.uid = udoc._id;
                else throw new UserNotFoundError(uidOrName);
            }
        }
        if (this.uid !== this.user._id) this.checkPerm(PERM.PERM_VIEW_RECORD);
        if (pid) {
            const pdoc = await problem.get(domainId, pid);
            if (pdoc) this.pid = pdoc.docId;
            else throw new ProblemNotFoundError(domainId, pid);
        }
        if (status) this.status = status;
        if (all) {
            this.checkPerm(PERM.PERM_VIEW_CONTEST_HIDDEN_SCOREBOARD);
            this.checkPerm(PERM.PERM_VIEW_HOMEWORK_HIDDEN_SCOREBOARD);
            this.all = true;
        }
        if (allDomain) {
            this.checkPriv(PRIV.PRIV_MANAGE_ALL_DOMAIN);
            this.allDomain = true;
        }
        this.noTemplate = noTemplate;
        this.throttleQueueClear = throttle(this.queueClear, 100, { trailing: true });
    }

    async message(msg: { rids: string[] }) {
        if (!(msg.rids instanceof Array)) return;
        const rids = msg.rids.map((id) => new ObjectId(id));
        const rdocs = await record.getMulti(this.args.domainId, { _id: { $in: rids } })
            .project<RecordDoc>(buildProjection(record.PROJECTION_LIST)).toArray();
        for (const rdoc of rdocs) this.onRecordChange(rdoc);
    }

    @subscribe('record/change')
    async onRecordChange(rdoc: RecordDoc) {
        if (!this.allDomain) {
            if (rdoc.domainId !== this.args.domainId) return;
            if (!this.pretest && typeof rdoc.input === 'string') return;
            if (!this.all) {
                if (!rdoc.contest && this.tid) return;
                if (rdoc.contest && ![this.tid, '000000000000000000000000'].includes(rdoc.contest.toString())) return;
                if (this.tid && rdoc.contest?.toString() !== '0'.repeat(24)) {
                    if (rdoc.uid !== this.user._id && !contest.canShowRecord.call(this, this.tdoc, true)) return;
                    if (rdoc.uid === this.user._id && !contest.canShowSelfRecord.call(this, this.tdoc, true)) return;
                }
            }
        }
        if (typeof this.pid === 'number' && rdoc.pid !== this.pid) return;
        if (typeof this.uid === 'number' && rdoc.uid !== this.uid) return;

        let [udoc, pdoc] = await Promise.all([
            user.getById(this.args.domainId, rdoc.uid),
            problem.get(rdoc.domainId, rdoc.pid),
        ]);
        const tdoc = this.tid ? this.tdoc : null;
        if (pdoc && !rdoc.contest) {
            if (!problem.canViewBy(pdoc, this.user)) pdoc = null;
            if (!this.user.hasPerm(PERM.PERM_VIEW_PROBLEM)) pdoc = null;
        }
        if (this.applyProjection && rdoc.contest?.toString() !== '0'.repeat(24)) rdoc = contest.applyProjection(tdoc, rdoc, this.user);
        if (this.pretest) {
            this.queueSend(rdoc._id.toHexString(), async () => ({ rdoc: omit(rdoc, ['code', 'input']) }));
        } else if (this.noTemplate) {
            this.queueSend(rdoc._id.toHexString(), async () => ({ rdoc }));
        } else {
            this.queueSend(rdoc._id.toHexString(), async () => ({
                html: await this.renderHTML('record_main_tr.html', {
                    rdoc, udoc, pdoc, tdoc, allDomain: this.allDomain,
                }),
            }));
        }
    }

    queueSend(rid: string, fn: () => Promise<any>) {
        this.queue.set(rid, fn);
        this.throttleQueueClear();
    }

    async queueClear() {
        await Promise.all([...this.queue.values()].map(async (fn) => this.send(await fn())));
        this.queue.clear();
    }
}

export class RecordDetailConnectionHandler extends ConnectionHandler {
    pdoc: ProblemDoc;
    tdoc?: Tdoc;
    rid: string = '';
    disconnectTimeout: NodeJS.Timeout;
    throttleSend: any;
    applyProjection = false;
    noTemplate = false;
    canViewCode = false;

    @param('rid', Types.ObjectId)
    @param('noTemplate', Types.Boolean, true)
    async prepare(domainId: string, rid: ObjectId, noTemplate = false) {
        const rdoc = await record.get(domainId, rid);
        if (!rdoc) return;
        if (rdoc.contest && ![record.RECORD_GENERATE, record.RECORD_PRETEST].some((i) => i.toHexString() === rdoc.contest.toHexString())) {
            this.tdoc = await contest.get(domainId, rdoc.contest);
            let canView = this.user.own(this.tdoc);
            canView ||= contest.canShowRecord.call(this, this.tdoc);
            canView ||= this.user._id === rdoc.uid && contest.canShowSelfRecord.call(this, this.tdoc);
            if (!canView) throw new PermissionError(PERM.PERM_VIEW_CONTEST_HIDDEN_SCOREBOARD);
            if (!this.user.own(this.tdoc) && !this.user.hasPerm(PERM.PERM_EDIT_CONTEST)) {
                this.applyProjection = true;
            }
        }
        const [pdoc, self] = await Promise.all([
            problem.get(rdoc.domainId, rdoc.pid),
            problem.getStatus(domainId, rdoc.pid, this.user._id),
        ]);

        this.canViewCode = rdoc.uid === this.user._id;
        this.canViewCode ||= this.user.hasPriv(PRIV.PRIV_READ_RECORD_CODE);
        this.canViewCode ||= this.user.hasPerm(PERM.PERM_READ_RECORD_CODE);
        this.canViewCode ||= this.user.hasPerm(PERM.PERM_READ_RECORD_CODE_ACCEPT) && self?.status === STATUS.STATUS_ACCEPTED;

        if (!rdoc.contest || this.user._id !== rdoc.uid) {
            if (!problem.canViewBy(pdoc, this.user)) throw new PermissionError(PERM.PERM_VIEW_PROBLEM_HIDDEN);
        }

        this.pdoc = pdoc;
        this.noTemplate = noTemplate;
        this.throttleSend = throttle(this.sendUpdate, 1000, { trailing: true });
        this.rid = rid.toString();
        this.onRecordChange(rdoc);
    }

    async sendUpdate(rdoc: RecordDoc) {
        if (this.noTemplate) {
            this.send({ rdoc });
        } else {
            this.send({
                status: rdoc.status,
                status_html: await this.renderHTML('record_detail_status.html', { rdoc, pdoc: this.pdoc }),
                summary_html: await this.renderHTML('record_detail_summary.html', { rdoc, pdoc: this.pdoc }),
            });
        }
    }

    @subscribe('record/change')
    // eslint-disable-next-line
    async onRecordChange(rdoc: RecordDoc, $set?: any, $push?: any) {
        if (rdoc._id.toString() !== this.rid) return;
        if (this.disconnectTimeout) {
            clearTimeout(this.disconnectTimeout);
            this.disconnectTimeout = null;
        }
        if (this.applyProjection) rdoc = contest.applyProjection(this.tdoc, rdoc, this.user);
        // TODO: frontend doesn't support incremental update
        // if ($set) this.send({ $set, $push });
        if (!this.canViewCode) {
            rdoc = {
                ...rdoc,
                code: '',
                compilerTexts: [],
            };
        }
        if (![STATUS.STATUS_WAITING, STATUS.STATUS_JUDGING, STATUS.STATUS_COMPILING, STATUS.STATUS_FETCHED].includes(rdoc.status)) {
            this.disconnectTimeout = setTimeout(() => this.close(4001, 'Ended'), 30000);
        }
        this.throttleSend(rdoc);
    }
}

export async function apply(ctx) {
    ctx.Route('record_main', '/record', RecordListHandler);
    ctx.Route('record_detail', '/record/:rid', RecordDetailHandler);
    ctx.Route('record_gpu_profile', '/record/:rid/gpu-profile/:caseId', RecordGPUProfileHandler);
    ctx.Connection('record_conn', '/record-conn', RecordMainConnectionHandler);
    ctx.Connection('record_detail_conn', '/record-detail-conn', RecordDetailConnectionHandler);
}
