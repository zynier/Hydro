import { parseLang } from '@hydrooj/common';
import { yaml } from '@hydrooj/utils';
import {
    Context, SettingModel, SystemModel,
} from 'hydrooj';
import { JudgeSettings, overrideConfig } from './config';
import { registerGPUExample } from './example';

export const Config = JudgeSettings;

async function ensureGPULanguages() {
    const raw = SystemModel.get('hydrooj.langs');
    if (typeof raw !== 'string') return;
    const languages = yaml.load(raw) as Record<string, any>;
    if (!languages) return;
    let updated = raw;
    if (!languages.cuda) {
        languages.cuda = {
            compile: 'echo "CUDA C++ is only available for GPU operator problems"',
            code_file: 'submission.cu',
            execute: 'echo "CUDA C++ is only available for GPU operator problems"',
            highlight: 'cpp',
            monaco: 'cpp',
            display: 'CUDA C++',
            hidden: true,
            process_limit: 1,
        };
        updated = yaml.dump(languages);
    }
    if (!languages.tilelang) {
        languages.tilelang = {
            compile: 'echo "TileLang is only available for GPU operator problems"',
            code_file: 'submission.tilelang.py',
            execute: 'echo "TileLang is only available for GPU operator problems"',
            highlight: 'python',
            monaco: 'python',
            display: 'TileLang 0.1.13',
            version: '0.1.13',
            hidden: true,
            process_limit: 1,
        };
        updated = yaml.dump(languages);
    }
    if (updated !== raw) await SystemModel.set('hydrooj.langs', updated);
    Object.assign(SettingModel.langs, parseLang(updated));
}

export async function apply(ctx: Context, config: ReturnType<typeof Config>) {
    overrideConfig(config);
    await ensureGPULanguages();
    registerGPUExample(ctx);
    if (process.env.NODE_APP_INSTANCE !== '0') return;
    // eslint-disable-next-line consistent-return
    if (!config.disable) return require('./hosts/builtin').apply(ctx);
}
