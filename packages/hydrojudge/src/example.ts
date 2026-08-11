import path from 'path';
import Schema from 'schemastery';
import { fs } from '@hydrooj/utils';
import {
    Context, ProblemModel,
} from 'hydrooj';

export function registerGPUExample(ctx: Context) {
    return ctx.addScript(
        'gpuExample',
        'Install the bundled CUDA, TileLang, TIRx, and Triton a += b GPU operator problem.',
        Schema.object({
            domainId: Schema.string().default('system'),
            pid: Schema.string().default('GPU1'),
            owner: Schema.number().default(1).min(1).step(1),
            force: Schema.boolean().default(false),
        }),
        async ({ domainId, pid, owner, force }, report) => {
            const root = path.resolve(__dirname, '../../../examples/gpu-a-plus-b');
            const [content, config, testcase] = await Promise.all([
                fs.readFile(path.join(root, 'problem.md'), 'utf8'),
                fs.readFile(path.join(root, 'config.yaml')),
                fs.readFile(path.join(root, 'testcase_config.py')),
            ]);
            let pdoc = await ProblemModel.get(domainId, pid);
            if (pdoc && !force) throw new Error(`Problem ${domainId}/${pid} already exists. Pass force=true to update it.`);
            if (pdoc) {
                pdoc = await ProblemModel.edit(domainId, pdoc.docId, {
                    title: 'a += b (fp16)',
                    content,
                    tag: ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                    html: false,
                });
            } else {
                const docId = await ProblemModel.add(
                    domainId, pid, 'a += b (fp16)', content, owner, ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                );
                pdoc = await ProblemModel.get(domainId, docId);
            }
            await Promise.all([
                ProblemModel.addTestdata(domainId, pdoc.docId, 'config.yaml', config, owner),
                ProblemModel.addTestdata(domainId, pdoc.docId, 'testcase_config.py', testcase, owner),
            ]);
            report({ message: `Installed ${domainId}/${pid} (numeric id ${pdoc.docId}).` });
            return true;
        },
    );
}

export function registerGPUDenseGemmExample(ctx: Context) {
    return ctx.addScript(
        'gpuDenseGemmExample',
        'Install the XPUOJ problem 2 compatible bf16 dense GeMM GPU operator problem.',
        Schema.object({
            domainId: Schema.string().default('system'),
            pid: Schema.string().default('GPU2'),
            owner: Schema.number().default(1).min(1).step(1),
            force: Schema.boolean().default(false),
        }),
        async ({ domainId, pid, owner, force }, report) => {
            const root = path.resolve(__dirname, '../../../examples/gpu-dense-gemm-bf16');
            const [content, config, testcase] = await Promise.all([
                fs.readFile(path.join(root, 'problem.md'), 'utf8'),
                fs.readFile(path.join(root, 'config.yaml')),
                fs.readFile(path.join(root, 'testcase_config.py')),
            ]);
            let pdoc = await ProblemModel.get(domainId, pid);
            if (pdoc && !force) throw new Error(`Problem ${domainId}/${pid} already exists. Pass force=true to update it.`);
            if (pdoc) {
                pdoc = await ProblemModel.edit(domainId, pdoc.docId, {
                    title: 'Dense GeMM (bf16)',
                    content,
                    tag: ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                    html: false,
                });
            } else {
                const docId = await ProblemModel.add(
                    domainId, pid, 'Dense GeMM (bf16)', content, owner,
                    ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                );
                pdoc = await ProblemModel.get(domainId, docId);
            }
            await Promise.all([
                ProblemModel.addTestdata(domainId, pdoc.docId, 'config.yaml', config, owner),
                ProblemModel.addTestdata(domainId, pdoc.docId, 'testcase_config.py', testcase, owner),
            ]);
            report({ message: `Installed ${domainId}/${pid} (numeric id ${pdoc.docId}).` });
            return true;
        },
    );
}

export function registerGPUTopPRenormExample(ctx: Context) {
    return ctx.addScript(
        'gpuTopPRenormExample',
        'Install the XPUOJ problem 3 compatible Top-p renormalization GPU operator problem.',
        Schema.object({
            domainId: Schema.string().default('system'),
            pid: Schema.string().default('GPU3'),
            owner: Schema.number().default(1).min(1).step(1),
            force: Schema.boolean().default(false),
        }),
        async ({ domainId, pid, owner, force }, report) => {
            const root = path.resolve(__dirname, '../../../examples/gpu-top-p-renorm');
            const [content, config, testcase] = await Promise.all([
                fs.readFile(path.join(root, 'problem.md'), 'utf8'),
                fs.readFile(path.join(root, 'config.yaml')),
                fs.readFile(path.join(root, 'testcase_config.py')),
            ]);
            let pdoc = await ProblemModel.get(domainId, pid);
            if (pdoc && !force) throw new Error(`Problem ${domainId}/${pid} already exists. Pass force=true to update it.`);
            if (pdoc) {
                pdoc = await ProblemModel.edit(domainId, pdoc.docId, {
                    title: 'Top-p Renorm Probs',
                    content,
                    tag: ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                    html: false,
                });
            } else {
                const docId = await ProblemModel.add(
                    domainId, pid, 'Top-p Renorm Probs', content, owner,
                    ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                );
                pdoc = await ProblemModel.get(domainId, docId);
            }
            await Promise.all([
                ProblemModel.addTestdata(domainId, pdoc.docId, 'config.yaml', config, owner),
                ProblemModel.addTestdata(domainId, pdoc.docId, 'testcase_config.py', testcase, owner),
            ]);
            report({ message: `Installed ${domainId}/${pid} (numeric id ${pdoc.docId}).` });
            return true;
        },
    );
}

export function registerGPUTopKRenormExample(ctx: Context) {
    return ctx.addScript(
        'gpuTopKRenormExample',
        'Install the XPUOJ problem 4 compatible Top-k renormalization GPU operator problem.',
        Schema.object({
            domainId: Schema.string().default('system'),
            pid: Schema.string().default('GPU4'),
            owner: Schema.number().default(1).min(1).step(1),
            force: Schema.boolean().default(false),
        }),
        async ({ domainId, pid, owner, force }, report) => {
            const root = path.resolve(__dirname, '../../../examples/gpu-top-k-renorm');
            const [content, config, testcase] = await Promise.all([
                fs.readFile(path.join(root, 'problem.md'), 'utf8'),
                fs.readFile(path.join(root, 'config.yaml')),
                fs.readFile(path.join(root, 'testcase_config.py')),
            ]);
            let pdoc = await ProblemModel.get(domainId, pid);
            if (pdoc && !force) throw new Error(`Problem ${domainId}/${pid} already exists. Pass force=true to update it.`);
            if (pdoc) {
                pdoc = await ProblemModel.edit(domainId, pdoc.docId, {
                    title: 'Top k Renorm Probs',
                    content,
                    tag: ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                    html: false,
                });
            } else {
                const docId = await ProblemModel.add(
                    domainId, pid, 'Top k Renorm Probs', content, owner,
                    ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                );
                pdoc = await ProblemModel.get(domainId, docId);
            }
            await Promise.all([
                ProblemModel.addTestdata(domainId, pdoc.docId, 'config.yaml', config, owner),
                ProblemModel.addTestdata(domainId, pdoc.docId, 'testcase_config.py', testcase, owner),
            ]);
            report({ message: `Installed ${domainId}/${pid} (numeric id ${pdoc.docId}).` });
            return true;
        },
    );
}

export function registerGPUTransposeExample(ctx: Context) {
    return ctx.addScript(
        'gpuTransposeExample',
        'Install the XPUOJ problem 100 compatible fp32 transpose GPU operator problem.',
        Schema.object({
            domainId: Schema.string().default('system'),
            pid: Schema.string().default('GPU5'),
            owner: Schema.number().default(1).min(1).step(1),
            force: Schema.boolean().default(false),
        }),
        async ({ domainId, pid, owner, force }, report) => {
            const root = path.resolve(__dirname, '../../../examples/gpu-transpose-fp32');
            const [content, config, testcase] = await Promise.all([
                fs.readFile(path.join(root, 'problem.md'), 'utf8'),
                fs.readFile(path.join(root, 'config.yaml')),
                fs.readFile(path.join(root, 'testcase_config.py')),
            ]);
            let pdoc = await ProblemModel.get(domainId, pid);
            if (pdoc && !force) throw new Error(`Problem ${domainId}/${pid} already exists. Pass force=true to update it.`);
            if (pdoc) {
                pdoc = await ProblemModel.edit(domainId, pdoc.docId, {
                    title: 'Transpose (fp32)',
                    content,
                    tag: ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                    html: false,
                });
            } else {
                const docId = await ProblemModel.add(
                    domainId, pid, 'Transpose (fp32)', content, owner,
                    ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                );
                pdoc = await ProblemModel.get(domainId, docId);
            }
            await Promise.all([
                ProblemModel.addTestdata(domainId, pdoc.docId, 'config.yaml', config, owner),
                ProblemModel.addTestdata(domainId, pdoc.docId, 'testcase_config.py', testcase, owner),
            ]);
            report({ message: `Installed ${domainId}/${pid} (numeric id ${pdoc.docId}).` });
            return true;
        },
    );
}

export function registerGPUFFTExample(ctx: Context) {
    return ctx.addScript(
        'gpuFFTExample',
        'Install the XPUOJ problem 96 compatible fp32 batched FFT GPU operator problem.',
        Schema.object({
            domainId: Schema.string().default('system'),
            pid: Schema.string().default('GPU6'),
            owner: Schema.number().default(1).min(1).step(1),
            force: Schema.boolean().default(false),
        }),
        async ({ domainId, pid, owner, force }, report) => {
            const root = path.resolve(__dirname, '../../../examples/gpu-fft-fp32');
            const [content, config, testcase] = await Promise.all([
                fs.readFile(path.join(root, 'problem.md'), 'utf8'),
                fs.readFile(path.join(root, 'config.yaml')),
                fs.readFile(path.join(root, 'testcase_config.py')),
            ]);
            let pdoc = await ProblemModel.get(domainId, pid);
            if (pdoc && !force) throw new Error(`Problem ${domainId}/${pid} already exists. Pass force=true to update it.`);
            if (pdoc) {
                pdoc = await ProblemModel.edit(domainId, pdoc.docId, {
                    title: 'FFT (fp32)',
                    content,
                    tag: ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                    html: false,
                });
            } else {
                const docId = await ProblemModel.add(
                    domainId, pid, 'FFT (fp32)', content, owner,
                    ['GPU', 'CUDA', 'TileLang', 'TIRx', 'Triton'],
                );
                pdoc = await ProblemModel.get(domainId, docId);
            }
            await Promise.all([
                ProblemModel.addTestdata(domainId, pdoc.docId, 'config.yaml', config, owner),
                ProblemModel.addTestdata(domainId, pdoc.docId, 'testcase_config.py', testcase, owner),
            ]);
            report({ message: `Installed ${domainId}/${pid} (numeric id ${pdoc.docId}).` });
            return true;
        },
    );
}

export function registerGPUEinsumOuterExample(ctx: Context) {
    return ctx.addScript(
        'gpuEinsumOuterExample',
        'Install the XPUOJ problem 139 compatible bfloat16 Einsum Outer GPU operator problem.',
        Schema.object({
            domainId: Schema.string().default('system'),
            pid: Schema.string().default('GPU7'),
            owner: Schema.number().default(1).min(1).step(1),
            force: Schema.boolean().default(false),
        }),
        async ({ domainId, pid, owner, force }, report) => {
            const root = path.resolve(__dirname, '../../../examples/gpu-einsum-outer-bf16');
            const [content, config, testcase] = await Promise.all([
                fs.readFile(path.join(root, 'problem.md'), 'utf8'),
                fs.readFile(path.join(root, 'config.yaml')),
                fs.readFile(path.join(root, 'testcase_config.py')),
            ]);
            let pdoc = await ProblemModel.get(domainId, pid);
            if (pdoc && !force) throw new Error(`Problem ${domainId}/${pid} already exists. Pass force=true to update it.`);
            if (pdoc) {
                pdoc = await ProblemModel.edit(domainId, pdoc.docId, {
                    title: 'Einsum Outer (bf16)',
                    content,
                    tag: [],
                    html: false,
                });
            } else {
                const docId = await ProblemModel.add(
                    domainId, pid, 'Einsum Outer (bf16)', content, owner, [],
                );
                pdoc = await ProblemModel.get(domainId, docId);
            }
            await Promise.all([
                ProblemModel.addTestdata(domainId, pdoc.docId, 'config.yaml', config, owner),
                ProblemModel.addTestdata(domainId, pdoc.docId, 'testcase_config.py', testcase, owner),
            ]);
            report({ message: `Installed ${domainId}/${pid} (numeric id ${pdoc.docId}).` });
            return true;
        },
    );
}
