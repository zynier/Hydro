import path from 'path';
import {
    convertIniConfig, LangConfig, normalizeSubtasks, ProblemConfigFile, readSubtasksFromFiles,
} from '@hydrooj/common';
import { readYamlCases } from '@hydrooj/common/cases';
import {
    changeErrorType, fs, parseTimeMS, yaml,
} from '@hydrooj/utils';
import { getConfig } from './config';
import { FormatError, SystemError } from './error';
import { NextFunction, ParsedConfig } from './interface';
import { ensureFile, parseMemoryMB } from './utils';

function isValidConfig(config) {
    if (config.type !== 'objective' && config.count > (getConfig('testcases_max') || 100)) {
        throw new FormatError('Too many testcases. Cancelled.');
    }
    if (config.type === 'communication') {
        if (!Number.isInteger(config.num_processes) || config.num_processes <= 0) {
            throw new FormatError('Number of processes must be a positive integer for communication type.');
        }
        if (config.num_processes > getConfig('processLimit')) {
            throw new FormatError('Number of processes larger than processLimit');
        }
    }
    const time = (config.num_processes || 1) * Math.sum(...config.subtasks.flatMap((subtask) => subtask.cases.map((c) => c.time)));
    if (time > (getConfig('total_time_limit') || 60) * 1000) {
        throw new FormatError('Total time limit longer than {0}s. Cancelled.', [+getConfig('total_time_limit') || 60]);
    }
    const memMax = Math.max(...config.subtasks.flatMap((subtask) => subtask.cases.map((c) => c.memory)));
    if (memMax > parseMemoryMB(getConfig('memoryMax'))) throw new FormatError('Memory limit larger than memory_max');
    if (!['default', 'strict'].includes(config.checker_type || 'default') && !config.checker) {
        throw new FormatError('You did not specify a checker.');
    }
    if (config.type === 'interactive' && !config.interactor) {
        throw new FormatError('Interactive problems require an interactor.');
    }
    if (config.multi_pass && (!Number.isInteger(config.multi_pass) || config.multi_pass < 2 || config.multi_pass > 20)) {
        throw new FormatError('Multi Pass must be between 2 and 20.');
    }
    if (config.multi_pass > 1 && !['default', 'interactive'].includes(config.type)) {
        throw new FormatError('Multi Pass only supported on default and interactive problems.');
    }
    if (config.multi_pass > 1 && config.type === 'default' && !['testlib', 'kattis'].includes(config.checker_type)) {
        throw new FormatError('Multi Pass on default problems requires a testlib or kattis checker.');
    }
}

async function collectFiles(folder: string) {
    const files = await fs.readdir(folder);
    await Promise.all(['input', 'output'].map(async (t) => {
        if (await fs.pathExists(path.resolve(folder, t))) {
            const f = await fs.readdir(path.resolve(folder, t));
            files.push(...f.map((i) => `${t}/${i}`));
        }
    }));
    return files;
}

interface Args {
    next: NextFunction;
    key: string;
    isSelfSubmission: boolean;
    trusted: boolean;
    lang: string;
    langConfig?: LangConfig;
}

function normalizeGPUConfig(folder: string, config: Record<string, any>, args: Args): ParsedConfig {
    if (args.lang !== 'cuda') throw new FormatError('GPU operator problems only support CUDA C++.');
    const gpu = config.gpu;
    if (!gpu || typeof gpu !== 'object') throw new FormatError('GPU configuration is required.');
    const entry = gpu.entry || 'run_kernel';
    const testcase = gpu.testcase || 'testcase_config.py';
    if (typeof entry !== 'string') throw new FormatError('GPU entry function must be a string.');
    if (typeof testcase !== 'string') throw new FormatError('GPU testcase must be a string.');
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(entry)) throw new FormatError('Invalid GPU entry function.');
    if (path.basename(testcase) !== testcase || !testcase.endsWith('.py')) {
        throw new FormatError('GPU testcase must be a Python filename in the testdata root.');
    }
    if (!fs.existsSync(path.join(folder, testcase))) throw new FormatError('GPU testcase file not found: {0}', [testcase]);
    if (!Array.isArray(gpu.cases) || !gpu.cases.length) throw new FormatError('At least one GPU case is required.');
    if (gpu.cases.length > (getConfig('testcases_max') || 100)) throw new FormatError('Too many testcases. Cancelled.');
    const ids = new Set<number>();
    const cases = gpu.cases.map((test, index) => {
        if (!test || typeof test !== 'object' || Array.isArray(test)) throw new FormatError('GPU cases must be objects.');
        const id = test.id ?? index + 1;
        const warmup = test.warmup ?? gpu.warmup;
        const repeats = test.repeats ?? gpu.repeats;
        if (!Number.isSafeInteger(id) || id < 1 || ids.has(id)) throw new FormatError('GPU case ids must be unique positive integers.');
        if (warmup !== undefined && (!Number.isSafeInteger(warmup) || warmup < 0 || warmup > 1000)) {
            throw new FormatError('GPU warmup must be between 0 and 1000.');
        }
        if (repeats !== undefined && (!Number.isSafeInteger(repeats) || repeats < 1 || repeats > 10000)) {
            throw new FormatError('GPU repeats must be between 1 and 10000.');
        }
        ids.add(id);
        return {
            id,
            memory: parseMemoryMB(test.memory || config.memory || '256m'),
            ...(warmup === undefined ? {} : { warmup }),
            ...(repeats === undefined ? {} : { repeats }),
        };
    });
    return {
        ...config,
        checker_type: 'default',
        count: cases.length,
        detail: config.detail || 'full',
        time: parseTimeMS(config.time || '10s'),
        memory: parseMemoryMB(config.memory || '256m'),
        gpu: {
            ...gpu,
            entry,
            testcase,
            cases,
        },
        subtasks: [],
    } as ParsedConfig;
}

export default async function readCases(folder: string, cfg: ProblemConfigFile = {}, args: Args): Promise<ParsedConfig> {
    const iniConfig = path.resolve(folder, 'config.ini');
    const yamlConfig = path.resolve(folder, 'config.yaml');
    const ymlConfig = path.resolve(folder, 'config.yml');
    const config: Record<string, any> = {
        checker_type: 'default',
        count: 0,
        subtasks: [],
        judge_extra_files: [],
        user_extra_files: [],
        ...cfg,
    };
    try {
        if (fs.existsSync(yamlConfig)) {
            Object.assign(config, yaml.load(await fs.readFile(yamlConfig, 'utf-8')));
        } else if (fs.existsSync(ymlConfig)) {
            Object.assign(config, yaml.load(await fs.readFile(ymlConfig, 'utf-8')));
        } else if (fs.existsSync(iniConfig)) {
            Object.assign(config, convertIniConfig(await fs.readFile(iniConfig, 'utf-8')));
        }
    } catch (e) {
        throw changeErrorType(e, FormatError);
    }
    if (config.type === 'gpu') return normalizeGPUConfig(folder, config, args);
    const timeRate = +(config.time_limit_rate?.[args.lang] || args.langConfig?.time_limit_rate || 1) || 1;
    const memoryRate = +(config.memory_limit_rate?.[args.lang] || args.langConfig?.memory_limit_rate || 1) || 1;
    const checkFile = ensureFile(folder);
    const result = await readYamlCases(config, checkFile)
        .catch((e) => { throw changeErrorType(e, FormatError); });
    result.count = Object.keys(result.answers || {}).length || Math.sum((result.subtasks || []).map((s) => s.cases.length));
    if (!result.count) {
        try {
            result.subtasks = readSubtasksFromFiles(await collectFiles(folder), cfg);
            result.count = Math.sum(result.subtasks.map((i) => i.cases.length));
            if (args.isSelfSubmission) args.next?.({ message: { message: 'Found {0} testcases.', params: [result.count] } });
        } catch (e) {
            throw new SystemError('Cannot parse testdata.', [e.message, ...(e.params || [])]);
        }
    }
    if (result.detail === true) result.detail = 'full';
    else if (result.detail === false) result.detail = 'case';
    else result.detail ||= 'full';
    result.subtasks = normalizeSubtasks(result.subtasks || [], checkFile, config.time, config.memory, false, timeRate, memoryRate);
    if (result.key && args.key !== result.key) throw new FormatError('Incorrect secret key');
    if (!result.key && !args.trusted) isValidConfig(result);
    return result;
}
