import os from 'os';
import path from 'path';
import { STATUS } from '@hydrooj/common';
import { fs } from '@hydrooj/utils';
import { CompileError } from '../error';
import { gpuPool, theoreticalLowerBoundMs } from '../gpu';
import {
    GPULanguage, NormalizedGPUCase, prepareGPUWorkdir, runGPUContainer,
} from '../gpu/runner';
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
    ctx.next({ status: STATUS.STATUS_COMPILING });
    ctx.next({ message: requestedHardware ? `Waiting for an idle ${requestedHardware} GPU.` : 'Waiting for an idle compatible GPU.' });
    const lease = await gpuPool.acquire(requestedHardware, minimumMemory);
    try {
        const workdir = await fs.mkdtemp(path.join(os.tmpdir(), `hydro-gpu-${ctx.rid}-`));
        await ctx.pushClean(() => fs.remove(workdir));
        await prepareGPUWorkdir(
            workdir,
            await readSubmissionCode(ctx, ctx.lang),
            path.join(ctx.folder, ctx.config.gpu.testcase),
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
            ctx.end({ status: STATUS.STATUS_TIME_LIMIT_EXCEEDED, score: 0, time: 0, memory: 0 });
            return;
        }
        const resultMap = new Map(execution.results.map((result) => [result.id, result]));
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
        ctx.end({
            status: totalStatus,
            score: totalScore,
            time: totalTime,
            memory: peakMemory,
        });
    } finally {
        await lease.release();
    }
};
