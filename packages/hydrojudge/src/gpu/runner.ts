/* eslint-disable no-await-in-loop */
import { spawn } from 'child_process';
import { createHash } from 'crypto';
import path from 'path';
import { findFileSync, fs } from '@hydrooj/utils';
import { getConfig } from '../config';
import { GPUDevice } from './index';

export type GPULanguage = 'cuda' | 'tilelang';

export interface NormalizedGPUCase {
    id: number;
    memory: number;
    warmup?: number;
    repeats?: number;
}

export interface GPUWorkload {
    flops: number;
    memoryBytes: number;
    dtype: string;
}

export interface GPUBenchmarkResult {
    id: number;
    kernelTimeMs: number;
    baselineTimeMs: number;
    correct: boolean;
    runtimeError: boolean;
    message: string;
    memoryBytes: number;
    warmup: number;
    repeats: number;
    workload: GPUWorkload;
}

export interface GPUContainerResult {
    code: number;
    stdout: string;
    stderr: string;
    compiled: boolean;
    timeout: 'compile' | 'execute' | null;
    results: GPUBenchmarkResult[];
}

export interface GPUProfileArtifactResult {
    id: number;
    code: number;
    stdout: string;
    stderr: string;
    timedOut: boolean;
    reportPath?: string;
    summaryPath?: string;
}

export interface GPUWorkdirOptions {
    profile?: boolean;
    profileMeasureRun?: number;
    profileNcu?: string;
    profileSet?: string;
    profileMaxInstances?: number;
}

function runnerSource(
    entry: string,
    cases: NormalizedGPUCase[],
    language: GPULanguage,
    profileMeasureRun = 1,
) {
    return String.raw`#!/usr/bin/env python3
import contextlib
import ctypes
import gc
import hashlib
import importlib.util
import inspect
import io
import json
import math
import resource
import secrets
import sys
import traceback

import torch

ENTRY = ${JSON.stringify(entry)}
LANGUAGE = ${JSON.stringify(language)}
CASE_CONFIGS = {item["id"]: item for item in ${JSON.stringify(cases)}}
DEFAULT_WARMUP = 100
DEFAULT_REPEATS = 2000
PROFILE_MEASURE_RUN = ${profileMeasureRun}


def load_testcase_config():
    spec = importlib.util.spec_from_file_location("hydro_gpu_testcase", "/work/testcase_config.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load testcase_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("getTestCaseSize", "genTestCase", "baseline", "check", "getWorkload"):
        if not callable(getattr(module, name, None)):
            raise TypeError(f"testcase_config.py must define {name}()")
    return module


def get_case_size(module, case_id):
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO(f"{case_id}\n")
        result = module.getTestCaseSize()
    finally:
        sys.stdin = old_stdin
    warmup = DEFAULT_WARMUP
    repeats = DEFAULT_REPEATS
    if (isinstance(result, tuple) and len(result) == 2
            and isinstance(result[1], (tuple, list)) and len(result[1]) == 2):
        sizes, (warmup, repeats) = result
    else:
        sizes = result
    case_config = CASE_CONFIGS[case_id]
    warmup = case_config.get("warmup", warmup)
    repeats = case_config.get("repeats", repeats)
    if not isinstance(warmup, int) or not 0 <= warmup <= 1000:
        raise ValueError("warmup must be an integer between 0 and 1000")
    if not isinstance(repeats, int) or not 1 <= repeats <= 10000:
        raise ValueError("repeats must be an integer between 1 and 10000")
    return sizes, warmup, repeats


def generate_inputs(module, sizes):
    signature = inspect.signature(module.genTestCase)
    positional = [
        parameter for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2 or any(
            parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        inputs = module.genTestCase(sizes, "cuda")
    else:
        inputs = module.genTestCase(sizes)
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("genTestCase() must return a list or tuple")
    return list(inputs)


def clone_all(inputs):
    return [value.clone() if torch.is_tensor(value) else value for value in inputs]


def clone_for_invocation(inputs, input_classes):
    cloned = []
    for index, value in enumerate(inputs):
        role = input_classes[index] if index < len(input_classes) else "INOUT"
        if torch.is_tensor(value) and role != "INPUT":
            cloned.append(value.clone())
        else:
            cloned.append(value)
    return cloned


def prepare_interleaved_invocations(inputs, input_classes, count, start_with_target):
    pairs = []
    for index in range(count):
        target_first = start_with_target if index % 2 == 0 else not start_with_target
        first = clone_for_invocation(inputs, input_classes)
        second = clone_for_invocation(inputs, input_classes)
        if target_first:
            target_inputs, baseline_inputs = first, second
        else:
            baseline_inputs, target_inputs = first, second
        pairs.append((target_inputs, baseline_inputs, target_first))
    return pairs


def as_ctype(value):
    if torch.is_tensor(value):
        if not value.is_cuda or not value.is_contiguous():
            raise ValueError("All tensor arguments must be contiguous CUDA tensors")
        return ctypes.c_void_p(value.data_ptr())
    if isinstance(value, bool):
        return ctypes.c_bool(value)
    if isinstance(value, int):
        return ctypes.c_int64(value)
    if isinstance(value, float):
        return ctypes.c_float(value)
    raise TypeError(f"Unsupported kernel argument type: {type(value).__name__}")


def load_python_kernel():
    try:
        import tilelang
    except ImportError as error:
        raise RuntimeError("TileLang 0.1.13 is not installed in the GPU judge image") from error
    version = str(getattr(tilelang, "__version__", ""))
    if version != "0.1.13":
        raise RuntimeError(f"GPU judge requires TileLang 0.1.13, found {version or 'unknown'}")
    spec = importlib.util.spec_from_file_location("hydro_gpu_submission", "/work/submission.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load TileLang submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, "/work")
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    try:
        function = getattr(module, ENTRY)
    except AttributeError as error:
        raise RuntimeError(f"Missing Python function: {ENTRY}") from error
    if not callable(function):
        raise RuntimeError(f"Python submission entry is not callable: {ENTRY}")
    return module, function


def load_kernel():
    if LANGUAGE == "tilelang":
        return load_python_kernel()
    library = ctypes.CDLL("/work/submission.so")
    try:
        function = getattr(library, ENTRY)
    except AttributeError as error:
        raise RuntimeError(f"Missing exported C symbol: {ENTRY}") from error
    function.restype = None

    def invoke(*args):
        function(*(as_ctype(value) for value in args))

    return library, invoke


def run_interleaved(target_call, baseline_call, invocation_pairs):
    for target_inputs, baseline_inputs, target_first in invocation_pairs:
        if target_first:
            target_call(*target_inputs)
            baseline_call(*baseline_inputs)
        else:
            baseline_call(*baseline_inputs)
            target_call(*target_inputs)


def record_call(call, inputs, stream, event_pairs):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record(stream)
    call(*inputs)
    end.record(stream)
    event_pairs.append((begin, end))


def elapsed_samples(event_pairs):
    samples = [begin.elapsed_time(end) for begin, end in event_pairs]
    if not samples or any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise RuntimeError("GPU timing produced an invalid sample")
    return samples


def benchmark_interleaved(target_call, baseline_call, warmup_pairs, timed_pairs):
    # Input clones are asynchronous CUDA work. Finish them before warmup so neither
    # implementation inherits pending setup work from the allocation phase.
    torch.cuda.synchronize()
    stream = torch.cuda.default_stream()
    torch.cuda.set_stream(stream)
    run_interleaved(target_call, baseline_call, warmup_pairs)
    torch.cuda.synchronize()
    torch.cuda.set_stream(stream)

    target_events = []
    baseline_events = []
    for target_inputs, baseline_inputs, target_first in timed_pairs:
        if target_first:
            record_call(target_call, target_inputs, stream, target_events)
            record_call(baseline_call, baseline_inputs, stream, baseline_events)
        else:
            record_call(baseline_call, baseline_inputs, stream, baseline_events)
            record_call(target_call, target_inputs, stream, target_events)

    # Event elapsed times are only valid after every operation queued on the
    # measurement stream has completed.
    torch.cuda.synchronize()
    target_samples = elapsed_samples(target_events)
    baseline_samples = elapsed_samples(baseline_events)
    return (
        sum(target_samples) / len(target_samples),
        sum(baseline_samples) / len(baseline_samples),
    )


def normalize_workload(raw):
    if not isinstance(raw, dict):
        raise TypeError("getWorkload() must return a dict")
    flops = float(raw.get("flops", 0))
    memory_bytes = float(raw.get("memory_bytes", raw.get("memoryBytes", 0)))
    dtype = str(raw.get("dtype", "fp32")).lower()
    if not math.isfinite(flops) or flops <= 0:
        raise ValueError("getWorkload().flops must be positive")
    if not math.isfinite(memory_bytes) or memory_bytes <= 0:
        raise ValueError("getWorkload().memory_bytes must be positive")
    return {"flops": flops, "memoryBytes": memory_bytes, "dtype": dtype}


def prepare_case(module, case_id):
    torch.manual_seed(case_id)
    torch.cuda.manual_seed_all(case_id)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    sizes, warmup, repeats = get_case_size(module, case_id)
    original = generate_inputs(module, sizes)
    input_classes = list(getattr(module, "INPUT_CLASS", ["INOUT"] * len(original)))
    if len(input_classes) != len(original):
        raise ValueError("INPUT_CLASS must contain one role for each kernel argument")
    if any(role not in ("INPUT", "OUTPUT", "INOUT") for role in input_classes):
        raise ValueError("INPUT_CLASS roles must be INPUT, OUTPUT, or INOUT")
    workload = normalize_workload(module.getWorkload(sizes))
    return sizes, warmup, repeats, original, input_classes, workload


def run_case(module, kernel, case_id):
    sizes, warmup, repeats, original, input_classes, workload = prepare_case(module, case_id)

    invocation_pairs = prepare_interleaved_invocations(
        original,
        input_classes,
        warmup + repeats,
        start_with_target=case_id % 2 == 1,
    )
    kernel_time_ms, baseline_time_ms = benchmark_interleaved(
        kernel,
        module.baseline,
        invocation_pairs[:warmup],
        invocation_pairs[warmup:],
    )
    del invocation_pairs
    gc.collect()
    torch.cuda.empty_cache()

    original_for_check = clone_all(original)
    target_for_check = clone_all(original)
    baseline_for_check = clone_all(original)
    torch.cuda.synchronize()
    kernel(*target_for_check)
    module.baseline(*baseline_for_check)
    torch.cuda.synchronize()
    checker_output = io.StringIO()
    with contextlib.redirect_stdout(checker_output), contextlib.redirect_stderr(checker_output):
        correct = bool(module.check(sizes, original_for_check, target_for_check, baseline_for_check))
    message = checker_output.getvalue().strip()[-4096:]
    if not correct and not message:
        message = "Output differs from the PyTorch baseline."

    return {
        "id": case_id,
        "kernelTimeMs": kernel_time_ms,
        "baselineTimeMs": baseline_time_ms,
        "correct": correct,
        "runtimeError": False,
        "message": message,
        "memoryBytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "warmup": warmup,
        "repeats": repeats,
        "workload": workload,
    }


def run_profile_case(module, kernel, case_id):
    _, warmup, repeats, original, input_classes, _ = prepare_case(module, case_id)
    measure_run = min(PROFILE_MEASURE_RUN, repeats)
    invocation_pairs = prepare_interleaved_invocations(
        original,
        input_classes,
        warmup + measure_run,
        start_with_target=case_id % 2 == 1,
    )

    # Match the clean pass through warmup and the measured invocations before
    # the selected run. Only the selected target call is inside the NVTX range.
    torch.cuda.synchronize()
    stream = torch.cuda.default_stream()
    torch.cuda.set_stream(stream)
    run_interleaved(kernel, module.baseline, invocation_pairs[:warmup + measure_run - 1])
    torch.cuda.synchronize()
    target_inputs, baseline_inputs, target_first = invocation_pairs[-1]
    if not target_first:
        module.baseline(*baseline_inputs)
        torch.cuda.synchronize()

    range_name = f"hydro-case-{case_id}"
    torch.cuda.nvtx.range_push(range_name)
    try:
        kernel(*target_inputs)
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"HYDRO_GPU_PROFILE_OK {case_id} {measure_run} {range_name}", flush=True)


def main():
    profile_mode = len(sys.argv) == 3 and sys.argv[1] == "--profile"
    case_id = int(sys.argv[2] if profile_mode else sys.argv[1])
    if case_id not in CASE_CONFIGS:
        raise ValueError(f"Unknown GPU testcase id: {case_id}")
    nonce = secrets.token_hex(32)
    print(f"HYDRO_GPU_NONCE {case_id} {nonce}", flush=True)
    module = load_testcase_config()
    torch.cuda.set_device(0)
    library, kernel = load_kernel()
    if profile_mode:
        run_profile_case(module, kernel, case_id)
        return
    try:
        result = run_case(module, kernel, case_id)
    except Exception as error:
        result = {
            "id": case_id,
            "kernelTimeMs": 0,
            "baselineTimeMs": 0,
            "correct": False,
            "runtimeError": True,
            "message": "".join(traceback.format_exception_only(type(error), error)).strip()[-4096:],
            "memoryBytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "warmup": 0,
            "repeats": 0,
            "workload": {"flops": 0, "memoryBytes": 0, "dtype": "unknown"},
        }
    del library
    payload = json.dumps(result, separators=(",", ":"), allow_nan=False)
    signature = hashlib.sha256((nonce + payload).encode()).hexdigest()
    print(f"HYDRO_GPU_RESULT {case_id} {signature} {payload}", flush=True)


if __name__ == "__main__":
    main()
`;
}

function compileScript(computeCapability: string, language: GPULanguage, profile = false) {
    if (language === 'tilelang') {
        return `#!/bin/bash
set -eu
python3 -m py_compile /work/submission.py
echo HYDRO_GPU_COMPILE_OK
`;
    }
    const arch = computeCapability.replace('.', '');
    return `#!/bin/bash
set -eu
nvcc -O3 --use_fast_math --extra-device-vectorization ${profile ? '-lineinfo ' : ''}-std=c++17 -DNDEBUG \\
  -Xptxas=-O3 -shared -Xcompiler=-O3 -Xcompiler=-fPIC \\
  -gencode=arch=compute_${arch},code=sm_${arch} \\
  -gencode=arch=compute_${arch},code=compute_${arch} \\
  -o /work/submission.so /work/submission.cu
echo HYDRO_GPU_COMPILE_OK
`;
}

function shellQuote(value: string) {
    return `'${value.replace(/'/g, '\u0027\\\u0027\u0027')}'`;
}

function profileScript(ncu: string, sectionSet: string, measureRun: number, maxInstances: number) {
    return `#!/bin/bash
set -eu
case_id="$1"
ncu_cmd=${shellQuote(ncu)}
section_set=${shellQuote(sectionSet)}
measure_run=${measureRun}
range_name="hydro-case-$case_id"
profile_dir="/profile"
report="$profile_dir/case-$case_id.ncu-rep"
details="$profile_dir/case-$case_id.details.csv"
session="$profile_dir/case-$case_id.session.txt"
source="$profile_dir/case-$case_id.source.txt"
summary="$profile_dir/case-$case_id.summary.json"
mkdir -p "$profile_dir"

"$ncu_cmd" \\
  --set "$section_set" \\
  --nvtx \\
  --nvtx-include "$range_name/" \\
  --replay-mode kernel \\
  --target-processes application-only \\
  --import-sass on \\
  --import-source on \\
  --source-folders /work,/tmp \\
  --export "$report" \\
  --force-overwrite \\
  python3 -u /work/runner.py --profile "$case_id"

"$ncu_cmd" --import "$report" --page details --csv --print-details all \\
  --print-metric-name label-name --print-units base --print-fp > "$details"
"$ncu_cmd" --import "$report" --page session > "$session"
"$ncu_cmd" --import "$report" --page source --print-source cuda,sass > "$source" 2>&1 || true
"$ncu_cmd" --import "$report" --page source --print-source ptx >> "$source" 2>&1 || true

HYDRO_NCU="$ncu_cmd" HYDRO_PROFILE_MAX_INSTANCES=${maxInstances} \\
  python3 /work/profile_summary.py "$report" "$details" "$session" "$source" "$summary" \\
  "$section_set" "$measure_run" "$range_name"
`;
}

export async function prepareGPUWorkdir(
    workdir: string,
    code: string | Buffer,
    testcase: string,
    entry: string,
    cases: NormalizedGPUCase[],
    device: GPUDevice,
    language: GPULanguage = 'cuda',
    options: GPUWorkdirOptions = {},
) {
    const profileMeasureRun = options.profileMeasureRun || 1;
    const writes: Promise<any>[] = [
        fs.writeFile(path.join(workdir, language === 'tilelang' ? 'submission.py' : 'submission.cu'), code),
        fs.copy(testcase, path.join(workdir, 'testcase_config.py')),
        fs.writeFile(path.join(workdir, 'runner.py'), runnerSource(entry, cases, language, profileMeasureRun), { mode: 0o700 }),
        fs.writeFile(path.join(workdir, 'compile.sh'), compileScript(device.computeCapability, language, options.profile), { mode: 0o700 }),
    ];
    if (options.profile) {
        writes.push(
            fs.copy(
                findFileSync('@hydrooj/hydrojudge/gpu/profile_summary.py'),
                path.join(workdir, 'profile_summary.py'),
            ),
            fs.writeFile(
                path.join(workdir, 'profile.sh'),
                profileScript(
                    options.profileNcu || 'ncu',
                    options.profileSet || 'full',
                    profileMeasureRun,
                    options.profileMaxInstances ?? 100000,
                ),
                { mode: 0o700 },
            ),
        );
    }
    await Promise.all(writes);
}

export function parseResults(stdout: string): GPUBenchmarkResult[] {
    const nonces = new Map<number, string>();
    const results = new Map<number, GPUBenchmarkResult>();
    for (const line of stdout.split('\n')) {
        if (line.startsWith('HYDRO_GPU_NONCE ')) {
            const match = /^HYDRO_GPU_NONCE (\d+) ([a-f0-9]{64})$/.exec(line);
            if (match && !nonces.has(+match[1])) nonces.set(+match[1], match[2]);
            continue;
        }
        if (!line.startsWith('HYDRO_GPU_RESULT ')) continue;
        try {
            const match = /^HYDRO_GPU_RESULT (\d+) ([a-f0-9]{64}) (.+)$/.exec(line);
            if (!match) continue;
            const id = +match[1];
            const nonce = nonces.get(id);
            const expected = nonce && createHash('sha256').update(nonce + match[3]).digest('hex');
            if (!expected || expected !== match[2]) continue;
            const result = JSON.parse(match[3]);
            const valid = result.id === id
                && typeof result.correct === 'boolean'
                && typeof result.runtimeError === 'boolean'
                && typeof result.message === 'string'
                && Number.isFinite(result.kernelTimeMs)
                && Number.isFinite(result.baselineTimeMs)
                && Number.isFinite(result.memoryBytes)
                && Number.isSafeInteger(result.warmup)
                && Number.isSafeInteger(result.repeats)
                && Number.isFinite(result.workload?.flops)
                && Number.isFinite(result.workload?.memoryBytes)
                && typeof result.workload?.dtype === 'string';
            if (valid) results.set(id, result);
        } catch { }
    }
    return [...results.values()];
}

function gpuContainerArgs(name: string, language: GPULanguage, tmpfsSize = '1g') {
    const config = getConfig('gpu');
    return [
        'run', '--rm', '--init', '--name', name,
        '--user', `${process.getuid?.() ?? 65534}:${process.getgid?.() ?? 65534}`,
        '--network', 'none',
        '--read-only',
        // TileLang JIT and Nsight Compute use executable temporary files.
        '--tmpfs', `/tmp:rw,nosuid,nodev,exec,size=${tmpfsSize}`,
        '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges',
        '--pids-limit', config.pids_limit.toString(),
        '--memory', config.memory,
        '--cpus', config.cpus.toString(),
        '--ulimit', 'core=0',
        '--env', 'HOME=/tmp',
        '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', `HYDRO_GPU_LANGUAGE=${language}`,
    ];
}

async function runOCIContainer(name: string, args: string[], timeout: number) {
    const runtime = getConfig('gpu').runtime;
    return await new Promise<{
        code: number;
        stdout: string;
        stderr: string;
        timedOut: boolean;
    }>((resolve, reject) => {
        const child = spawn(runtime, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        let stdout = '';
        let stderr = '';
        let timedOut = false;
        const append = (current: string, chunk: Buffer) => (current + chunk.toString('utf8')).slice(-1024 * 1024);
        const stopContainer = () => {
            child.kill('SIGKILL');
            const cleanup = spawn(runtime, ['kill', name], { stdio: 'ignore' });
            cleanup.unref();
        };
        const killTimer = setTimeout(() => {
            timedOut = true;
            stopContainer();
        }, timeout);
        child.stdout.on('data', (chunk: Buffer) => { stdout = append(stdout, chunk); });
        child.stderr.on('data', (chunk: Buffer) => { stderr = append(stderr, chunk); });
        child.on('error', (error) => {
            clearTimeout(killTimer);
            reject(error);
        });
        child.on('close', (code) => {
            clearTimeout(killTimer);
            resolve({ code: code ?? -1, stdout, stderr, timedOut });
        });
    });
}

const randomSuffix = () => Math.random().toString(36).slice(2, 8);

export async function runGPUContainer(
    workdir: string,
    device: GPUDevice,
    rid: string,
    executionTimeout: number,
    cases: NormalizedGPUCase[],
    language: GPULanguage = 'cuda',
): Promise<GPUContainerResult> {
    const config = getConfig('gpu');
    const safeRid = rid.replace(/[^A-Za-z0-9_.-]/g, '').slice(-24);
    const compileName = `hydro-gpu-compile-${safeRid}-${randomSuffix()}`;
    const compile = await runOCIContainer(compileName, [
        ...gpuContainerArgs(compileName, language),
        '--volume', `${workdir}:/work:rw`,
        '--workdir', '/work',
        config.container_image,
        '/bin/bash', '/work/compile.sh',
    ], config.compile_timeout);
    const compiled = compile.code === 0 && compile.stdout.includes('HYDRO_GPU_COMPILE_OK');
    if (!compiled || compile.timedOut) {
        return {
            code: compile.code,
            stdout: compile.stdout,
            stderr: compile.stderr,
            compiled: false,
            timeout: compile.timedOut ? 'compile' : null,
            results: [],
        };
    }

    let stdout = compile.stdout;
    let stderr = compile.stderr;
    let code = 0;
    const results: GPUBenchmarkResult[] = [];
    const executionLimit = Math.min(config.execution_timeout, executionTimeout);
    const deadline = Date.now() + executionLimit;
    for (const test of cases) {
        const remaining = deadline - Date.now();
        if (remaining <= 0) {
            return { code, stdout, stderr, compiled, timeout: 'execute', results };
        }
        const name = `hydro-gpu-case-${test.id}-${safeRid}-${randomSuffix()}`;
        const execution = await runOCIContainer(name, [
            ...gpuContainerArgs(name, language),
            '--gpus', `device=${device.uuid}`,
            '--env', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
            '--volume', `${workdir}:/work:ro`,
            '--workdir', '/work',
            config.container_image,
            'python3', '-u', '/work/runner.py', test.id.toString(),
        ], remaining);
        stdout = (stdout + execution.stdout).slice(-1024 * 1024);
        stderr = (stderr + execution.stderr).slice(-1024 * 1024);
        results.push(...parseResults(execution.stdout));
        if (execution.code !== 0 && code === 0) code = execution.code;
        if (execution.timedOut) return { code, stdout, stderr, compiled, timeout: 'execute', results };
    }
    return { code, stdout, stderr, compiled, timeout: null, results };
}

export async function runGPUProfiles(
    workdir: string,
    device: GPUDevice,
    rid: string,
    cases: NormalizedGPUCase[],
    language: GPULanguage,
    onResult?: (result: GPUProfileArtifactResult) => Promise<void> | void,
) {
    const config = getConfig('gpu');
    const safeRid = rid.replace(/[^A-Za-z0-9_.-]/g, '').slice(-24);
    const profileDir = path.join(workdir, 'profile');
    await fs.ensureDir(profileDir);
    const results: GPUProfileArtifactResult[] = [];
    const publish = async (result: GPUProfileArtifactResult) => {
        results.push(result);
        await onResult?.(result);
    };

    const compileName = `hydro-gpu-profile-compile-${safeRid}-${randomSuffix()}`;
    const compile = await runOCIContainer(compileName, [
        ...gpuContainerArgs(compileName, language, config.profile_tmpfs),
        '--volume', `${workdir}:/work:rw`,
        '--workdir', '/work',
        config.container_image,
        '/bin/bash', '/work/compile.sh',
    ], config.compile_timeout);
    const compiled = compile.code === 0 && compile.stdout.includes('HYDRO_GPU_COMPILE_OK');
    if (!compiled || compile.timedOut) {
        for (const test of cases) {
            await publish({
                id: test.id,
                code: compile.code,
                stdout: compile.stdout,
                stderr: compile.stderr || 'Nsight Compute profile compilation failed.',
                timedOut: compile.timedOut,
            });
        }
        return results;
    }

    for (const test of cases) {
        const name = `hydro-gpu-profile-${test.id}-${safeRid}-${randomSuffix()}`;
        const execution = await runOCIContainer(name, [
            ...gpuContainerArgs(name, language, config.profile_tmpfs),
            '--gpus', `device=${device.uuid}`,
            '--env', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
            '--volume', `${workdir}:/work:ro`,
            '--volume', `${profileDir}:/profile:rw`,
            '--workdir', '/work',
            config.container_image,
            '/bin/bash', '/work/profile.sh', test.id.toString(),
        ], config.profile_timeout);
        const reportPath = path.join(workdir, 'profile', `case-${test.id}.ncu-rep`);
        const summaryPath = path.join(workdir, 'profile', `case-${test.id}.summary.json`);
        let validArtifacts = execution.code === 0 && !execution.timedOut
            && await fs.pathExists(reportPath) && await fs.pathExists(summaryPath);
        if (validArtifacts) {
            try {
                const summary = JSON.parse(await fs.readFile(summaryPath, 'utf8'));
                validArtifacts = Number.isSafeInteger(summary.actionCount) && summary.actionCount > 0;
                if (!validArtifacts) execution.stderr += '\nNsight Compute did not capture a target kernel.';
            } catch (error) {
                validArtifacts = false;
                execution.stderr += `\nUnable to validate the Nsight Compute summary: ${error.message}`;
            }
        }
        await publish({
            id: test.id,
            code: execution.code,
            stdout: execution.stdout,
            stderr: execution.stderr,
            timedOut: execution.timedOut,
            ...validArtifacts ? { reportPath, summaryPath } : {},
        });
    }
    return results;
}
