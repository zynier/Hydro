import { createHash } from 'crypto';
import os from 'os';
import path from 'path';
import { expect } from 'chai';
import fs from 'fs-extra';
import { describe, it } from 'node:test';

global.Hydro = {} as any;

const { GPUResourcePool, theoreticalLowerBoundMs } = require('../src/gpu');
const { gpuPerformanceScore } = require('../src/judge/gpu');
const { parseResults, prepareGPUWorkdir } = require('../src/gpu/runner');
const readCases = require('../src/cases').default;

describe('GPU operator judge', () => {
    it('uses the larger roofline component', () => {
        const hardware = { bandwidthGBps: 1000, fp32TFLOPS: 10 };
        expect(theoreticalLowerBoundMs(1e9, 12e9, hardware)).to.equal(12);
        expect(theoreticalLowerBoundMs(100e9, 1e9, hardware)).to.equal(10);
        expect(theoreticalLowerBoundMs(100e9, 1e9, hardware, 'fp16')).to.equal(5);
    });

    it('groups identical GPU models', () => {
        const pool = new GPUResourcePool();
        pool.devices = [0, 1].map((index) => ({
            index,
            uuid: `GPU-${index}`,
            name: 'Example GPU',
            computeCapability: '9.0',
            memoryMiB: 81920,
            freeMemoryMiB: 80000,
            bandwidthGBps: 2000,
            fp32TFLOPS: 50,
            pciBusId: `0000:0${index}:00.0`,
        }));
        const [hardware] = pool.hardware();
        expect(hardware.count).to.equal(2);
        expect(hardware).to.include({ name: 'Example GPU', fp32TFLOPS: 50, bandwidthGBps: 2000 });
    });

    it('rejects a GPU memory request that no device can satisfy', async () => {
        const pool = new GPUResourcePool();
        pool.devices = [{
            index: 0,
            uuid: 'GPU-0',
            name: 'Example GPU',
            computeCapability: '9.0',
            memoryMiB: 81920,
            freeMemoryMiB: 80000,
            bandwidthGBps: 2000,
            fp32TFLOPS: 50,
            pciBusId: '0000:01:00.0',
        }];
        let message = '';
        try {
            await pool.acquire('', 90000);
        } catch (error) {
            message = error.message;
        }
        expect(message).to.include('enough memory');
    });

    it('anchors performance scoring at the baseline and hardware limit', () => {
        expect(gpuPerformanceScore(10, 10, 2)).to.equal(50);
        expect(gpuPerformanceScore(2, 10, 2)).to.equal(100);
        expect(gpuPerformanceScore(18, 10, 2)).to.be.closeTo(100 / 3, 1e-12);
        expect(gpuPerformanceScore(1, 10, 2)).to.be.greaterThan(100).and.at.most(150);
        expect(gpuPerformanceScore(0, 10, 2)).to.equal(0);
    });

    it('accepts only nonce-signed benchmark results', () => {
        const nonce = 'a'.repeat(64);
        const result = {
            id: 3,
            kernelTimeMs: 1,
            baselineTimeMs: 2,
            correct: true,
            runtimeError: false,
            message: '',
            memoryBytes: 1024,
            warmup: 3,
            repeats: 30,
            workload: { flops: 10, memoryBytes: 20, dtype: 'fp16' },
        };
        const payload = JSON.stringify(result);
        const signature = createHash('sha256').update(nonce + payload).digest('hex');
        const output = [
            `HYDRO_GPU_NONCE 3 ${nonce}`,
            `HYDRO_GPU_RESULT 3 ${'0'.repeat(64)} ${payload}`,
            `HYDRO_GPU_RESULT 3 ${signature} ${payload}`,
        ].join('\n');
        expect(parseResults(output)).to.deep.equal([result]);
    });

    it('generates the optimized interleaved runner and CUDA build flags', async () => {
        const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'hydro-gpu-runner-test-'));
        try {
            const testcase = path.join(folder, 'trusted-testcase.py');
            await fs.writeFile(testcase, '# trusted testcase definition\n');
            await prepareGPUWorkdir(
                folder,
                'extern "C" void run_kernel() {}\n',
                testcase,
                'run_kernel',
                [{ id: 1, memory: 512, warmup: 3, repeats: 30 }],
                {
                    index: 0,
                    uuid: 'GPU-0',
                    name: 'Example GPU',
                    computeCapability: '9.0',
                    memoryMiB: 81920,
                    freeMemoryMiB: 80000,
                    bandwidthGBps: 2000,
                    fp32TFLOPS: 50,
                    pciBusId: '0000:01:00.0',
                },
            );
            const compile = await fs.readFile(path.join(folder, 'compile.sh'), 'utf8');
            const runner = await fs.readFile(path.join(folder, 'runner.py'), 'utf8');
            expect(compile).to.include('--use_fast_math --extra-device-vectorization');
            expect(compile).to.include('-Xptxas=-O3 -shared -Xcompiler=-O3 -Xcompiler=-fPIC');
            expect(runner).to.include('prepare_interleaved_invocations');
            expect(runner).to.include('benchmark_interleaved');
            expect(runner).to.include('torch.cuda.default_stream()');
        } finally {
            await fs.remove(folder);
        }
    });

    it('generates a TileLang Python runner and syntax-only build step', async () => {
        const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'hydro-gpu-tilelang-test-'));
        try {
            const testcase = path.join(folder, 'trusted-testcase.py');
            await fs.writeFile(testcase, '# trusted testcase definition\n');
            await prepareGPUWorkdir(
                folder,
                'def run_kernel(*args):\n    pass\n',
                testcase,
                'run_kernel',
                [{ id: 1, memory: 512, warmup: 3, repeats: 30 }],
                {
                    index: 0,
                    uuid: 'GPU-0',
                    name: 'Example GPU',
                    computeCapability: '9.0',
                    memoryMiB: 81920,
                    freeMemoryMiB: 80000,
                    bandwidthGBps: 2000,
                    fp32TFLOPS: 50,
                    pciBusId: '0000:01:00.0',
                },
                'tilelang',
            );
            const compile = await fs.readFile(path.join(folder, 'compile.sh'), 'utf8');
            const runner = await fs.readFile(path.join(folder, 'runner.py'), 'utf8');
            expect(await fs.pathExists(path.join(folder, 'submission.py'))).to.equal(true);
            expect(compile).to.include('python3 -m py_compile /work/submission.py');
            expect(compile).to.not.include('nvcc');
            expect(runner).to.include('GPU judge requires TileLang 0.1.13');
            expect(runner).to.include('Missing Python function');
            expect(runner).to.include('def run_profile_case(module, kernel, case_id):');
        } finally {
            await fs.remove(folder);
        }
    });

    it('generates an isolated full NCU profile pass', async () => {
        const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'hydro-gpu-profile-test-'));
        try {
            const testcase = path.join(folder, 'trusted-testcase.py');
            await fs.writeFile(testcase, '# trusted testcase definition\n');
            await prepareGPUWorkdir(
                folder,
                'extern "C" void run_kernel() {}\n',
                testcase,
                'run_kernel',
                [{ id: 3, memory: 512, warmup: 3, repeats: 30 }],
                {
                    index: 0,
                    uuid: 'GPU-0',
                    name: 'Example GPU',
                    computeCapability: '9.0',
                    memoryMiB: 81920,
                    freeMemoryMiB: 80000,
                    bandwidthGBps: 2000,
                    fp32TFLOPS: 50,
                    pciBusId: '0000:01:00.0',
                },
                'cuda',
                {
                    profile: true,
                    profileMeasureRun: 2,
                    profileNcu: 'ncu',
                    profileSet: 'full',
                    profileMaxInstances: 5000,
                },
            );
            const compile = await fs.readFile(path.join(folder, 'compile.sh'), 'utf8');
            const runner = await fs.readFile(path.join(folder, 'runner.py'), 'utf8');
            const script = await fs.readFile(path.join(folder, 'profile.sh'), 'utf8');
            const profileFunction = runner.slice(
                runner.indexOf('def run_profile_case'),
                runner.indexOf('\ndef main():'),
            );
            expect(compile).to.include('-lineinfo');
            expect(compile).to.not.include(' -G ');
            expect(runner).to.include('PROFILE_MEASURE_RUN = 2');
            expect(profileFunction).to.include('torch.cuda.nvtx.range_push(range_name)');
            expect(profileFunction).to.include('kernel(*target_inputs)');
            expect(profileFunction).to.not.include('checkCorrectness');
            expect(script).to.include('--set "$section_set"');
            expect(script).to.include('--nvtx-include "$range_name/"');
            expect(script).to.include('--replay-mode kernel');
            expect(script).to.include('--target-processes application-only');
            expect(script).to.include('--import-sass on');
            expect(script).to.include('--import-source on');
            expect(script).to.include('--export "$report"');
            expect(script).to.include('--print-source ptx');
            expect(script).to.include('profile_summary.py');
            expect(await fs.pathExists(path.join(folder, 'profile_summary.py'))).to.equal(true);
        } finally {
            await fs.remove(folder);
        }
    });

    it('normalizes a GPU config without input/output files', async () => {
        const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'hydro-gpu-config-test-'));
        try {
            await fs.writeFile(path.join(folder, 'testcase_config.py'), '# trusted testcase definition\n');
            await fs.writeFile(path.join(folder, 'config.yaml'), `
type: gpu
langs: [cuda]
time: 10s
memory: 512m
gpu:
  warmup: 3
  repeats: 30
  cases:
    - id: 7
`);
            const config = await readCases(folder, { type: 'gpu' }, {
                next: () => null,
                key: '',
                isSelfSubmission: false,
                trusted: false,
                lang: 'cuda',
            });
            expect(config.count).to.equal(1);
            expect(config.gpu).to.include({ entry: 'run_kernel', testcase: 'testcase_config.py' });
            expect(config.gpu.cases[0]).to.include({
                id: 7,
                memory: 512,
                warmup: 3,
                repeats: 30,
            });
        } finally {
            await fs.remove(folder);
        }
    });

    it('accepts TileLang as a GPU operator language', async () => {
        const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'hydro-gpu-tilelang-config-test-'));
        try {
            await fs.writeFile(path.join(folder, 'testcase_config.py'), '# trusted testcase definition\n');
            await fs.writeFile(path.join(folder, 'config.yaml'), `
type: gpu
langs: [tilelang]
time: 10s
memory: 512m
gpu:
  cases:
    - id: 7
`);
            const config = await readCases(folder, { type: 'gpu' }, {
                next: () => null,
                key: '',
                isSelfSubmission: false,
                trusted: false,
                lang: 'tilelang',
            });
            expect(config.count).to.equal(1);
            expect(config.gpu).to.include({ entry: 'run_kernel', testcase: 'testcase_config.py' });
        } finally {
            await fs.remove(folder);
        }
    });
});
