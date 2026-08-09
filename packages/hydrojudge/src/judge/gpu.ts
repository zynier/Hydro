import { randomBytes } from 'crypto';
import os from 'os';
import path from 'path';
import {
    GPULanguage, GPUProfileState, GPUProfileTiming, STATUS,
} from '@hydrooj/common';
import { fs } from '@hydrooj/utils';
import { getConfig } from '../config';
import { CompileError } from '../error';
import { gpuPool, theoreticalLowerBoundMs } from '../gpu';
import {
    NormalizedGPUCase, prepareGPUWorkdir, runGPUContainer, runGPUProfiles,
} from '../gpu/runner';
import logger from '../log';
import { Context } from './interface';

function resultMessage(
    deviceName: string,
    kernelTimeMs: number,
    baselineTimeMs: number,
    lowerBoundMs: number,
    workloadBytes: number,
    performanceScore: number,
    warmup: number,
    repeats: number,
    checkerMessage: string,
) {
    const efficiency = lowerBoundMs > 0 && kernelTimeMs > 0 ? Math.min(999.99, lowerBoundMs / kernelTimeMs * 100) : 0;
    const bandwidth = kernelTimeMs > 0 ? workloadBytes / (kernelTimeMs / 1000) / 1e9 : 0;
    const speedup = kernelTimeMs > 0 ? baselineTimeMs / kernelTimeMs : 0;
    const theoretical = lowerBoundMs > 0 ? `${lowerBoundMs.toFixed(6)} ms` : 'unavailable';
    return [
        deviceName,
        `kernel ${kernelTimeMs.toFixed(6)} ms`,
        `PyTorch baseline ${baselineTimeMs.toFixed(6)} ms`,
        `${speedup.toFixed(3)}x baseline speedup`,
        `theoretical lower bound ${theoretical}`,
        `${efficiency.toFixed(2)}% roofline efficiency`,
        `${bandwidth.toFixed(2)} GB/s`,
        `performance score ${performanceScore.toFixed(6)}`,
        `${warmup} warmup / ${repeats} interleaved measured runs`,
        checkerMessage,
    ].join('; ');
}

export function gpuPerformanceScore(kernelTimeMs: number, baselineTimeMs: number, lowerBoundMs: number) {
    if (!(kernelTimeMs > 0) || !(baselineTimeMs > 0)) return 0;
    const effectiveLowerBound = lowerBoundMs > 0 && lowerBoundMs < baselineTimeMs ? lowerBoundMs : 0;
    const headroom = baselineTimeMs - effectiveLowerBound;
    const denominator = kernelTimeMs - effectiveLowerBound + headroom;
    const rawScore = denominator <= 0 ? Number.POSITIVE_INFINITY : 100 * headroom / denominator;
    if (rawScore <= 100) return Math.max(0, rawScore);
    return Math.min(150, 100 + 10 * Math.log2(rawScore / 100));
}

async function readSubmissionCode(ctx: Context, language: string) {
    if ('content' in ctx.code) return ctx.code.content;
    if ('src' in ctx.code) return await fs.readFile(ctx.code.src);
    throw new CompileError({ stderr: `${language} submission source is unavailable.` });
}

export const judge = async (ctx: Context) => {
    const language = ctx.lang as GPULanguage;
    const cases = ctx.config.gpu.cases as unknown as NormalizedGPUCase[];
    const requestedHardware = ctx.request.hardware || '';
    const minimumMemory = Math.max(...cases.map((test) => test.memory));
    const gpuConfig = getConfig('gpu');
    ctx.next({ status: STATUS.STATUS_COMPILING });
    ctx.next({ message: requestedHardware ? `Waiting for an idle ${requestedHardware} GPU.` : 'Waiting for an idle compatible GPU.' });
    const lease = await gpuPool.acquire(requestedHardware, minimumMemory);
    let leaseTransferred = false;
    try {
        const workdir = await fs.mkdtemp(path.join(os.tmpdir(), `hydro-gpu-${ctx.rid}-`));
        await ctx.pushClean(() => fs.remove(workdir));
        const submissionCode = await readSubmissionCode(ctx, ctx.lang);
        const testcasePath = path.join(ctx.folder, ctx.config.gpu.testcase);
        await prepareGPUWorkdir(
            workdir,
            submissionCode,
            testcasePath,
            ctx.config.gpu.entry,
            cases,
            lease.device,
            language,
        );
        ctx.next({
            status: STATUS.STATUS_JUDGING,
            progress: 0,
            message: `Assigned ${lease.device.name} (${lease.device.uuid}, compute capability ${lease.device.computeCapability}).`,
        });
        const execution = await runGPUContainer(workdir, lease.device, ctx.rid, ctx.config.time, cases, language);
        if (execution.timeout === 'compile') throw new CompileError({ stderr: `${language} compilation timed out.` });
        if (!execution.compiled) throw new CompileError({ stdout: execution.stdout, stderr: execution.stderr });
        if (execution.timeout === 'execute') {
            await ctx.end({ status: STATUS.STATUS_TIME_LIMIT_EXCEEDED, score: 0, time: 0, memory: 0 });
            return;
        }
        const resultMap = new Map(execution.results.map((result) => [result.id, result]));
        const profileTimings: Record<number, GPUProfileTiming> = {};
        let totalScore = 0;
        let totalTime = 0;
        let totalStatus = STATUS.STATUS_ACCEPTED;
        let peakMemory = 0;
        for (const test of cases) {
            const result = resultMap.get(test.id);
            if (!result) {
                totalStatus = Math.max(totalStatus, STATUS.STATUS_RUNTIME_ERROR);
                ctx.next({
                    case: {
                        id: test.id,
                        subtaskId: 1,
                        status: STATUS.STATUS_RUNTIME_ERROR,
                        score: 0,
                        time: 0,
                        memory: 0,
                        message: (execution.stderr || execution.stdout || 'GPU benchmark did not produce a result for this case.').slice(-4096),
                    },
                    addProgress: 100 / cases.length,
                });
                continue;
            }
            const lowerBoundMs = theoreticalLowerBoundMs(
                result.workload.flops,
                result.workload.memoryBytes,
                lease.device,
                result.workload.dtype,
            );
            const validTiming = result.kernelTimeMs > 0 && result.baselineTimeMs > 0;
            const status = result.runtimeError || !validTiming
                ? STATUS.STATUS_RUNTIME_ERROR
                : result.correct ? STATUS.STATUS_ACCEPTED : STATUS.STATUS_WRONG_ANSWER;
            const performanceScore = status === STATUS.STATUS_ACCEPTED
                ? gpuPerformanceScore(result.kernelTimeMs, result.baselineTimeMs, lowerBoundMs)
                : 0;
            const score = performanceScore / cases.length;
            const memory = Math.ceil(result.memoryBytes / 1024);
            if (validTiming) {
                profileTimings[test.id] = {
                    kernelTimeMs: result.kernelTimeMs,
                    baselineTimeMs: result.baselineTimeMs,
                    speedup: result.baselineTimeMs / result.kernelTimeMs,
                    theoreticalLowerBoundMs: lowerBoundMs,
                    rooflineEfficiency: lowerBoundMs > 0 ? lowerBoundMs / result.kernelTimeMs * 100 : 0,
                    bandwidthGBps: result.workload.memoryBytes / (result.kernelTimeMs / 1000) / 1e9,
                    performanceScore,
                    warmup: result.warmup,
                    repeats: result.repeats,
                    memoryBytes: result.memoryBytes,
                };
            }
            totalScore += score;
            totalTime += result.kernelTimeMs;
            totalStatus = Math.max(totalStatus, status);
            peakMemory = Math.max(peakMemory, memory);
            ctx.next({
                case: {
                    id: test.id,
                    subtaskId: 1,
                    status,
                    score,
                    time: result.kernelTimeMs,
                    memory,
                    message: status === STATUS.STATUS_RUNTIME_ERROR
                        ? result.message || 'GPU testcase execution failed.'
                        : resultMessage(
                            lease.device.name,
                            result.kernelTimeMs,
                            result.baselineTimeMs,
                            lowerBoundMs,
                            result.workload.memoryBytes,
                            performanceScore,
                            result.warmup,
                            result.repeats,
                            result.correct ? 'PyTorch baseline check passed' : result.message,
                        ),
                },
                addProgress: 100 / cases.length,
            });
        }

        let profileWorkdir: string;
        let profiles: Record<string, GPUProfileState>;
        const profileEnabled = gpuConfig.profile_enabled && !!ctx.session.postGPUProfile;
        if (profileEnabled) {
            const createdAt = new Date().toISOString();
            profiles = Object.fromEntries(cases.map((test) => {
                const measuredRepeats = profileTimings[test.id]?.repeats || gpuConfig.profile_measure_run;
                return [test.id.toString(), {
                    id: randomBytes(16).toString('hex'),
                    status: 'pending',
                    language,
                    hardware: lease.device.name,
                    computeCapability: lease.device.computeCapability,
                    set: gpuConfig.profile_set,
                    measureRun: Math.min(gpuConfig.profile_measure_run, measuredRepeats),
                    nvtxRange: `hydro-case-${test.id}`,
                    timing: profileTimings[test.id],
                    createdAt,
                } satisfies GPUProfileState];
            }));
            try {
                profileWorkdir = await fs.mkdtemp(path.join(os.tmpdir(), `hydro-gpu-profile-${ctx.rid}-`));
                await prepareGPUWorkdir(
                    profileWorkdir,
                    submissionCode,
                    testcasePath,
                    ctx.config.gpu.entry,
                    cases,
                    lease.device,
                    language,
                    {
                        profile: true,
                        profileMeasureRun: gpuConfig.profile_measure_run,
                        profileNcu: gpuConfig.profile_ncu,
                        profileSet: gpuConfig.profile_set,
                        profileMaxInstances: gpuConfig.profile_max_instances,
                    },
                );
            } catch (error) {
                if (profileWorkdir) await fs.remove(profileWorkdir);
                profileWorkdir = null;
                profiles = Object.fromEntries(Object.entries(profiles).map(([id, profile]) => [id, {
                    ...profile,
                    status: 'error',
                    completedAt: new Date().toISOString(),
                    error: `Unable to prepare the Nsight Compute profile pass: ${error.message}`.slice(-4096),
                }]));
            }
            ctx.next({ gpuProfiles: profiles });
        }

        await ctx.end({
            status: totalStatus,
            score: totalScore,
            time: totalTime,
            memory: peakMemory,
        });

        if (profileWorkdir) {
            leaseTransferred = true;
            const completed = new Set<number>();
            const publish = async (caseId: number, profile: GPUProfileState, reportPath?: string, summaryPath?: string) => {
                await ctx.session.postGPUProfile(ctx.rid, caseId, profile, reportPath, summaryPath);
                completed.add(caseId);
            };
            void runGPUProfiles(
                profileWorkdir, lease.device, ctx.rid, cases, language,
                async (result) => {
                    const pending = profiles[result.id];
                    try {
                        if (!result.reportPath || !result.summaryPath) {
                            const error = result.timedOut
                                ? 'Nsight Compute profiling timed out.'
                                : result.stderr || result.stdout || 'Nsight Compute did not produce a report.';
                            await publish(result.id, {
                                ...pending,
                                status: 'error',
                                completedAt: new Date().toISOString(),
                                error: error.slice(-4096),
                            });
                            return;
                        }
                        const summary = JSON.parse(await fs.readFile(result.summaryPath, 'utf8'));
                        const logWarnings = `${result.stdout}\n${result.stderr}`.split('\n')
                            .filter((line) => /warning/i.test(line)).slice(-20);
                        const warnings = [...new Set([...(summary.warnings || []), ...logWarnings])];
                        await publish(result.id, {
                            ...pending,
                            status: 'ready',
                            ncuVersion: summary.ncuVersion,
                            completedAt: new Date().toISOString(),
                            ...warnings.length ? { warnings } : {},
                        }, result.reportPath, result.summaryPath);
                    } catch (error) {
                        await publish(result.id, {
                            ...pending,
                            status: 'error',
                            completedAt: new Date().toISOString(),
                            error: `Unable to store the Nsight Compute report: ${error.message}`.slice(-4096),
                        });
                    }
                },
            ).catch(async (error) => {
                logger.error('GPU profile pass failed for %s: %s', ctx.rid, error.stack || error.message);
                await Promise.all(cases.filter((test) => !completed.has(test.id)).map(async (test) => {
                    const pending = profiles[test.id];
                    await ctx.session.postGPUProfile(ctx.rid, test.id, {
                        ...pending,
                        status: 'error',
                        completedAt: new Date().toISOString(),
                        error: `Nsight Compute profile worker failed: ${error.message}`.slice(-4096),
                    }).catch((uploadError) => logger.error(uploadError));
                }));
            }).finally(async () => {
                await fs.remove(profileWorkdir).catch(() => null);
                await lease.release();
            }).catch((error) => {
                logger.error('GPU profile cleanup failed for %s: %s', ctx.rid, error.stack || error.message);
            });
        }
    } finally {
        if (!leaseTransferred) await lease.release();
    }
};
