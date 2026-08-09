import os from 'os';
import path from 'path';
import cac from 'cac';
import Schema from 'schemastery';
import { fs, randomstring, yaml } from '@hydrooj/utils';

const argv = cac().parse();

const defaultEnv = `\
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOME=/w
# modify to your python version installed
PYTHONPATH=/lib/python3.13/site-packages
`;

export const JudgeSettings = Schema.object({
    cache_dir: Schema.string().default(path.resolve(os.homedir(), '.cache', 'hydro', 'judge')).description('Testdata cache directory'),
    tmp_dir: Schema.string().default(path.resolve(os.tmpdir(), 'hydro', 'judge')),
    stdio_size: Schema.string().pattern(/^\d+[kmg]b?$/g).default('32m'),
    memoryMax: Schema.string().pattern(/^\d+[kmg]b?$/g).default('512m'),
    strict_memory: Schema.boolean().default(false).description('Use address space memory limit'),
    sandbox_host: Schema.string().role('url').default('http://localhost:5050'),
    testcases_max: Schema.number().default(100).min(1).step(1),
    total_time_limit: Schema.number().default(60).min(1),
    processLimit: Schema.number().default(32).min(1).step(1),
    parallelism: Schema.number().default(2).min(1).step(1),
    concurrency: Schema.number(),
    singleTaskParallelism: Schema.number().default(2).min(1).step(1),
    rerun: Schema.number().description('Re-Run testcase if time-limit-exceeded (max per submission)').default(0).min(0).step(1),
    rate: Schema.number().default(1),
    env: Schema.string().default(defaultEnv).role('textarea'),
    host: Schema.any(),
    secret: Schema.string().description('Judge Token Secret').default(randomstring(32)),
    disable: Schema.boolean().description('Disable builtin judge').default(false),
    tracing: Schema.object({
        endpoint: Schema.string().role('url').description('Tempo endpoint').default('http://localhost:4318'),
        samplePercentage: Schema.number().description('Sample percentage').default(0).min(0).max(1),
    }),
    pipe_proxy: Schema.boolean().description('Enable pipe proxy').default(true),
    detail: Schema.union([
        Schema.const('full'),
        Schema.const('case'),
        Schema.const('none'),
        Schema.transform(Schema.union([
            Schema.boolean().deprecated(),
            Schema.const('full'),
            Schema.const('none'),
            Schema.const('case'),
        ]), (v) => (typeof v === 'boolean' ? (v ? 'full' : 'case') : v)),
    ]).description('Show diff detail').default('full'),
    performance: Schema.boolean().description('Performance mode').default(false),
    gpu: Schema.object({
        enabled: Schema.boolean().default(true).description('Enable native GPU operator judge'),
        runtime: Schema.string().default('docker').description('OCI container runtime executable'),
        container_image: Schema.string().default('pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel')
            .description('PyTorch CUDA image; build packages/hydrojudge/gpu for TileLang 0.1.13 support'),
        nvidia_smi: Schema.string().default('nvidia-smi'),
        nvcc: Schema.string().default('nvcc'),
        lock_dir: Schema.string().default(path.resolve(os.tmpdir(), 'hydro', 'gpu-locks')),
        poll_interval: Schema.number().default(1000).min(100).step(100),
        compile_timeout: Schema.number().default(120000).min(1000).step(1000),
        execution_timeout: Schema.number().default(120000).min(1000).step(1000),
        profile_enabled: Schema.boolean().default(true).description('Generate an asynchronous Nsight Compute report for each GPU testcase'),
        profile_ncu: Schema.string().default('ncu').description('Nsight Compute CLI executable inside the GPU image'),
        profile_set: Schema.string().default('full').description('Nsight Compute section set'),
        profile_timeout: Schema.number().default(900000).min(1000).step(1000),
        profile_measure_run: Schema.number().default(1).min(1).step(1),
        profile_max_instances: Schema.number().default(100000).min(0).step(1000)
            .description('Maximum total raw metric instance values stored in the web summary'),
        profile_tmpfs: Schema.string().pattern(/^\d+[kmg]b?$/g).default('4g'),
        cpus: Schema.number().default(4).min(1),
        memory: Schema.string().pattern(/^\d+[kmg]b?$/g).default('8g'),
        pids_limit: Schema.number().default(256).min(16).step(1),
    }),
});

const oldPath = path.resolve(os.homedir(), '.config', 'hydro', 'judge.yaml');
const newPath = path.resolve(os.homedir(), '.hydro', 'judge.yaml');

let config = global.Hydro
    ? JudgeSettings({})
    : (() => {
        const base: any = {};
        if (process.env.TEMP_DIR || argv.options.tmp) {
            base.tmp_dir = path.resolve(process.env.TEMP_DIR || argv.options.tmp);
        }
        if (process.env.CACHE_DIR || argv.options.cache) {
            base.cache_dir = path.resolve(process.env.CACHE_DIR || argv.options.cache);
        }
        if (process.env.EXECUTION_HOST || argv.options.sandbox) {
            base.sandbox_host = path.resolve(process.env.EXECUTION_HOST || argv.options.sandbox);
        }
        const configFilePath = (process.env.CONFIG_FILE || argv.options.config)
            ? path.resolve(process.env.CONFIG_FILE || argv.options.config)
            : fs.existsSync(newPath) ? newPath : oldPath;
        const configFile = fs.readFileSync(configFilePath, 'utf-8');
        Object.assign(base, yaml.load(configFile) as any);
        if (process.env.OVERRIDE_CONFIG) {
            if (fs.existsSync(process.env.OVERRIDE_CONFIG)) {
                const overrideConfigFile = fs.readFileSync(process.env.OVERRIDE_CONFIG, 'utf-8');
                Object.assign(base, yaml.load(overrideConfigFile) as any);
            } else console.warn('Override config file not found');
        }
        const cfg = JudgeSettings(base);
        return JudgeSettings(cfg);
    })();

export function overrideConfig(update: ReturnType<typeof JudgeSettings>) {
    config = JudgeSettings(update);
}

export const getConfig: <K extends keyof typeof config>(key: K) => typeof config[K] = (key) => config[key];
