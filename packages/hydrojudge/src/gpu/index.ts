/* eslint-disable no-await-in-loop */
import { execFile } from 'child_process';
import { promises as nativeFs } from 'fs';
import os from 'os';
import path from 'path';
import { promisify } from 'util';
import { fs, sleep } from '@hydrooj/utils';
import { getConfig } from '../config';
import { FormatError, SystemError } from '../error';
import logger from '../log';

const execFileAsync = promisify(execFile);

export interface GPUDevice {
    index: number;
    uuid: string;
    name: string;
    computeCapability: string;
    memoryMiB: number;
    freeMemoryMiB: number;
    bandwidthGBps: number;
    fp32TFLOPS: number;
    pciBusId: string;
}

export interface GPUHardware {
    id: string;
    name: string;
    count: number;
    computeCapability: string;
    memoryMiB: number;
    bandwidthGBps: number;
    fp32TFLOPS: number;
}

export interface GPULease {
    device: GPUDevice;
    release: () => Promise<void>;
}

interface CUDAProbe {
    index: number;
    memoryBusWidth: number;
    memoryClockKHz: number;
    multiProcessorCount: number;
    smClockKHz: number;
}

const probeSource = String.raw`
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 1;
    for (int i = 0; i < count; ++i) {
        int memoryBusWidth = 0;
        int memoryClockKHz = 0;
        int multiProcessorCount = 0;
        int smClockKHz = 0;
        if (cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, i) != cudaSuccess) return 2;
        if (cudaDeviceGetAttribute(&memoryClockKHz, cudaDevAttrMemoryClockRate, i) != cudaSuccess) return 3;
        if (cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, i) != cudaSuccess) return 4;
        if (cudaDeviceGetAttribute(&smClockKHz, cudaDevAttrClockRate, i) != cudaSuccess) return 5;
        std::printf("{\"index\":%d,\"memoryBusWidth\":%d,\"memoryClockKHz\":%d,"
                    "\"multiProcessorCount\":%d,\"smClockKHz\":%d}\n",
                    i, memoryBusWidth, memoryClockKHz, multiProcessorCount, smClockKHz);
    }
    return 0;
}
`;

function fp32UnitsPerSM(computeCapability: string) {
    const [major, minor] = computeCapability.split('.').map(Number);
    if (major === 2) return minor === 0 ? 32 : 48;
    if (major === 3) return 192;
    if (major === 5) return 128;
    if (major === 6) return minor === 0 ? 64 : 128;
    if (major === 7) return 64;
    if (major === 8) return minor === 0 ? 64 : 128;
    if (major >= 9) return 128;
    return 0;
}

function parseCSVLine(line: string) {
    return line.split(',').map((value) => value.trim());
}

async function queryNvidiaSMI(fields: string[]) {
    const config = getConfig('gpu');
    const { stdout } = await execFileAsync(config.nvidia_smi, [
        `--query-gpu=${fields.join(',')}`,
        '--format=csv,noheader,nounits',
    ], { timeout: 10000, maxBuffer: 1024 * 1024, encoding: 'utf8' });
    return stdout.trim().split('\n').map((line) => line.trim()).filter(Boolean);
}

async function probeCUDA(): Promise<Map<number, CUDAProbe>> {
    const config = getConfig('gpu');
    const dir = await nativeFs.mkdtemp(path.join(os.tmpdir(), 'hydro-gpu-probe-'));
    const source = path.join(dir, 'probe.cu');
    const binary = path.join(dir, 'probe');
    try {
        await fs.writeFile(source, probeSource);
        const candidates = [
            config.nvcc,
            process.env.CUDA_HOME && path.join(process.env.CUDA_HOME, 'bin', 'nvcc'),
            process.env.CUDA_PATH && path.join(process.env.CUDA_PATH, 'bin', 'nvcc'),
            process.env.CONDA_PREFIX && path.join(process.env.CONDA_PREFIX, 'bin', 'nvcc'),
            process.env.CONDA_EXE && path.join(path.dirname(process.env.CONDA_EXE), 'nvcc'),
            '/usr/local/cuda/bin/nvcc',
        ].filter(Boolean);
        let nvcc = candidates[0];
        for (const candidate of candidates.filter((item) => item.includes('/'))) {
            try {
                await nativeFs.access(candidate, fs.constants.X_OK);
                nvcc = candidate;
                break;
            } catch { }
        }
        await execFileAsync(nvcc, ['-O2', '-o', binary, source], {
            timeout: config.compile_timeout,
            maxBuffer: 4 * 1024 * 1024,
        });
        const { stdout } = await execFileAsync(binary, [], { timeout: 10000, maxBuffer: 1024 * 1024, encoding: 'utf8' });
        const result = new Map<number, CUDAProbe>();
        for (const line of stdout.trim().split('\n').filter(Boolean)) {
            const item = JSON.parse(line) as CUDAProbe;
            result.set(item.index, item);
        }
        return result;
    } finally {
        await fs.remove(dir);
    }
}

async function detectDevices(): Promise<GPUDevice[]> {
    const rows = await queryNvidiaSMI([
        'index', 'uuid', 'name', 'pci.bus_id', 'compute_cap', 'memory.total', 'memory.free',
    ]);
    let probe = new Map<number, CUDAProbe>();
    try {
        probe = await probeCUDA();
    } catch (e) {
        logger.warn('CUDA device property probe failed; theoretical GPU limits are unavailable: %s', e.message);
    }
    return rows.map((line) => {
        const [index, uuid, name, pciBusId, computeCapability, memoryMiB, freeMemoryMiB] = parseCSVLine(line);
        const properties = probe.get(+index);
        const bandwidthGBps = properties
            ? (2 * properties.memoryClockKHz * 1000 * properties.memoryBusWidth / 8) / 1e9
            : 0;
        const fp32TFLOPS = properties
            ? (fp32UnitsPerSM(computeCapability) * properties.multiProcessorCount * 2 * properties.smClockKHz * 1000) / 1e12
            : 0;
        return {
            index: +index,
            uuid,
            name,
            pciBusId,
            computeCapability,
            memoryMiB: +memoryMiB,
            freeMemoryMiB: +freeMemoryMiB,
            bandwidthGBps,
            fp32TFLOPS,
        };
    });
}

async function busyGPUUUIDs() {
    const config = getConfig('gpu');
    const { stdout } = await execFileAsync(config.nvidia_smi, [
        '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits',
    ], { timeout: 10000, maxBuffer: 1024 * 1024, encoding: 'utf8' });
    return new Set(stdout.trim().split('\n').map((line) => line.trim()).filter((line) => line.startsWith('GPU-')));
}

function lockFilename(uuid: string) {
    return `${uuid.replace(/[^A-Za-z0-9_-]/g, '_')}.lock`;
}

async function tryLock(uuid: string): Promise<(() => Promise<void>) | null> {
    const lockDir = getConfig('gpu').lock_dir;
    await fs.ensureDir(lockDir);
    const file = path.join(lockDir, lockFilename(uuid));
    try {
        const handle = await nativeFs.open(file, 'wx');
        await handle.writeFile(JSON.stringify({ pid: process.pid, createdAt: new Date().toISOString() }));
        await handle.close();
        return async () => {
            try {
                await nativeFs.unlink(file);
            } catch (e) {
                if (e.code !== 'ENOENT') throw e;
            }
        };
    } catch (e) {
        if (e.code !== 'EEXIST') throw e;
        try {
            const payload = JSON.parse(await nativeFs.readFile(file, 'utf8'));
            process.kill(payload.pid, 0);
        } catch (lockError) {
            if (lockError.code === 'EPERM') return null;
            try {
                await nativeFs.unlink(file);
            } catch { }
        }
        return null;
    }
}

function denseTensorCoreRatio(computeCapability?: string) {
    const [major, minor] = (computeCapability || '').split('.').map(Number);
    if (major >= 10) return 30;
    if (major === 9) return 15;
    if (major === 8 && minor === 0) return 16;
    if (major === 8 && minor === 6) return 4;
    if (major === 8 && minor === 9) return 2;
    if (major === 7) return 8;
    return 2;
}

function peakTFLOPSForDtype(
    hardware: Pick<GPUHardware, 'fp32TFLOPS'> & Partial<Pick<GPUHardware, 'computeCapability'>>,
    dtype: string,
) {
    if (['fp16', 'float16', 'half', 'bf16', 'bfloat16'].includes(dtype.toLowerCase())) {
        return hardware.fp32TFLOPS * denseTensorCoreRatio(hardware.computeCapability);
    }
    if (['fp64', 'float64', 'double'].includes(dtype.toLowerCase())) return hardware.fp32TFLOPS / 2;
    return hardware.fp32TFLOPS;
}

export function theoreticalLowerBoundMs(
    flops: number,
    memoryBytes: number,
    hardware: Pick<GPUHardware, 'bandwidthGBps' | 'fp32TFLOPS'>
        & Partial<Pick<GPUHardware, 'computeCapability'>>,
    dtype = 'fp32',
) {
    if (!(hardware.bandwidthGBps > 0) || !(hardware.fp32TFLOPS > 0)) return 0;
    const computeSeconds = flops / (peakTFLOPSForDtype(hardware, dtype) * 1e12);
    const memorySeconds = memoryBytes / (hardware.bandwidthGBps * 1e9);
    return Math.max(computeSeconds, memorySeconds) * 1000;
}

export class GPUResourcePool {
    devices: GPUDevice[] = [];
    private reserved = new Set<string>();

    async start() {
        if (!getConfig('gpu').enabled) return;
        try {
            this.devices = await detectDevices();
            if (!this.devices.length) logger.warn('GPU operator judge enabled, but no NVIDIA GPU was detected.');
            for (const device of this.devices) {
                logger.info(
                    'GPU detected: %s (%s, sm_%s, %s GiB, %s GB/s, %s TFLOPS FP32)',
                    device.name, device.uuid, device.computeCapability.replace('.', ''),
                    (device.memoryMiB / 1024).toFixed(1), device.bandwidthGBps.toFixed(1), device.fp32TFLOPS.toFixed(2),
                );
            }
        } catch (e) {
            this.devices = [];
            logger.warn('GPU detection failed: %s', e.message);
        }
    }

    hardware(): GPUHardware[] {
        const grouped = new Map<string, GPUHardware>();
        for (const device of this.devices) {
            const item = grouped.get(device.name);
            if (item) item.count++;
            else {
                grouped.set(device.name, {
                    id: device.name,
                    name: device.name,
                    count: 1,
                    computeCapability: device.computeCapability,
                    memoryMiB: device.memoryMiB,
                    bandwidthGBps: device.bandwidthGBps,
                    fp32TFLOPS: device.fp32TFLOPS,
                });
            }
        }
        return [...grouped.values()];
    }

    async acquire(hardware = '', minimumFreeMemoryMiB = 0): Promise<GPULease> {
        if (!getConfig('gpu').enabled) throw new SystemError('GPU operator judge is disabled.');
        if (!this.devices.length) throw new SystemError('No NVIDIA GPU is available to judge this submission.');
        const matchingDevices = this.devices.filter((device) => !hardware || device.name === hardware);
        if (hardware && !matchingDevices.length) {
            throw new FormatError('Requested GPU hardware is not available: {0}', [hardware]);
        }
        if (!matchingDevices.some((device) => device.memoryMiB >= minimumFreeMemoryMiB)) {
            throw new SystemError('No compatible GPU has enough memory for this problem.');
        }
        while (true) {
            const [busy, memoryRows] = await Promise.all([
                busyGPUUUIDs(),
                queryNvidiaSMI(['uuid', 'memory.free']),
            ]);
            const freeMemory = new Map(memoryRows.map((line) => {
                const [uuid, value] = parseCSVLine(line);
                return [uuid, +value];
            }));
            const candidates = this.devices
                .filter((device) => !hardware || device.name === hardware)
                .filter((device) => !this.reserved.has(device.uuid) && !busy.has(device.uuid))
                .filter((device) => (freeMemory.get(device.uuid) || 0) >= minimumFreeMemoryMiB)
                .sort((a, b) => (freeMemory.get(b.uuid) || 0) - (freeMemory.get(a.uuid) || 0));
            for (const device of candidates) {
                const unlock = await tryLock(device.uuid);
                if (!unlock) continue;
                let nowBusy: Set<string>;
                try {
                    nowBusy = await busyGPUUUIDs();
                } catch (error) {
                    await unlock();
                    throw error;
                }
                if (nowBusy.has(device.uuid)) {
                    await unlock();
                    continue;
                }
                this.reserved.add(device.uuid);
                return {
                    device: { ...device, freeMemoryMiB: freeMemory.get(device.uuid) || device.freeMemoryMiB },
                    release: async () => {
                        this.reserved.delete(device.uuid);
                        await unlock();
                    },
                };
            }
            await sleep(getConfig('gpu').poll_interval);
        }
    }
}

export const gpuPool = new GPUResourcePool();
