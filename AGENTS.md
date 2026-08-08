# Hydro OJ 当前环境运维

本文记录本机无 `sudo` 用户环境下的 Hydro OJ 部署。路径和服务名以当前账号 `dongwu.chen` 为准；迁移到其他账号时需要替换其中的 home 路径。

## 部署概览

- 仓库：`/data0/shared/dongwu.chen/Hydro`
- Conda 环境：`/home/dongwu.chen/miniconda3/envs/hydro-oj`
- Hydro profile：`conda-local`
- Hydro 配置：`/home/dongwu.chen/.hydro/profiles/conda-local/`
- PM2 状态：`/data0/shared/dongwu.chen/.hydro-conda/pm2`
- Hydro 文件存储：`/data0/shared/dongwu.chen/.hydro-conda/file`
- MongoDB 数据：`/data0/shared/dongwu.chen/.hydro-conda/mongodb`
- 用户级 systemd 单元：`hydro-conda.service`
- 当前版本：HydroOJ `5.0.4`、Node.js `22.21.1`、MongoDB `7.0.34`、Go Judge `v1.12.2`

当前由 PM2 管理三个进程：

| 进程 | 监听地址 | 用途 |
| --- | --- | --- |
| `hydro-conda` | `0.0.0.0:11037` | HydroOJ Web/API 服务 |
| `hydro-sandbox` | `127.0.0.1:5050` | Go Judge 判题沙箱 |
| `mongo-conda` | `127.0.0.1:27017` | MongoDB |

Hydro 的 `sandbox_host` 当前使用默认值 `http://localhost:5050`。profile 中的 MongoDB URI 是 `mongodb://127.0.0.1:27017/hydro`。

`node14` 的实际业务网卡地址是 `10.30.64.14`；`192.168.80.116:12345` 是外层 SSH 映射，不是节点网卡地址。Hydro 已监听全部节点网卡，但要通过 `192.168.80.116:11037` 访问，仍需网络侧增加到 `10.30.64.14:11037` 的端口映射。

## 命令环境

在终端中先准备环境：

```bash
cd /data0/shared/dongwu.chen/Hydro
source /home/dongwu.chen/miniconda3/etc/profile.d/conda.sh
conda activate hydro-oj
export PATH=/home/dongwu.chen/miniconda3/envs/hydro-oj/bin:/home/dongwu.chen/.local/bin:$PATH
export PM2_HOME=/data0/shared/dongwu.chen/.hydro-conda/pm2
export HYDRO_PROFILE=conda-local
```

非交互脚本至少要设置 `PATH`、`PM2_HOME` 和 `HYDRO_PROFILE`。不要使用系统 PM2 或系统 MongoDB 的默认数据目录。

## 日常运维

用户级 systemd 单元负责调用 `pm2 resurrect`，PM2 的进程清单保存在 `dump.pm2`。常用命令：

```bash
# 查看服务和进程
systemctl --user status hydro-conda.service --no-pager
pm2 ls

# 启动、停止、重启整个 Hydro 栈
systemctl --user start hydro-conda.service
systemctl --user stop hydro-conda.service
systemctl --user restart hydro-conda.service

# 仅操作单个进程
pm2 restart hydro-conda
pm2 restart hydro-sandbox
pm2 restart mongo-conda

# 修改 PM2 进程后保存开机恢复清单
pm2 save
```

`hydro-conda.service` 是 `Type=oneshot`、`RemainAfterExit=yes`，所以正常状态显示为 `active (exited)`，实际工作进程状态应以 `pm2 ls` 为准。

当前用户级单元已启用，且 `loginctl show-user dongwu.chen -p Linger` 返回 `Linger=yes`，因此退出登录或机器重启后 systemd 用户实例也会恢复 PM2 清单。

## 健康检查

```bash
pm2 ls
ss -ltnp | grep -E ':(27017|5050|11037)\b'
curl -fsS http://127.0.0.1:5050/version
curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:11037/
```

判题沙箱至少应满足：`hydro-sandbox` 为 `online`、重启次数不持续增长、`5050` 正在监听，并且 `/version` 返回 Go Judge 版本信息。

## 日志

```bash
pm2 logs hydro-conda --lines 200
pm2 logs hydro-sandbox --lines 200
pm2 logs mongo-conda --lines 200
journalctl --user -u hydro-conda.service -n 200 --no-pager
```

应用日志文件位于 `/data0/shared/dongwu.chen/.hydro-conda/logs/`：

- `hydro-out.log`、`hydro-error.log`
- `sandbox-out.log`、`sandbox-error.log`
- `mongo-out.log`、`mongo-error.log`

PM2 daemon 日志位于 `/data0/shared/dongwu.chen/.hydro-conda/pm2/pm2.log`。

`sandbox-error.log` 在沙箱崩溃循环时可能快速增长。清理前先保留需要的诊断信息，例如复制到带日期的备份文件后再截断；不要删除 MongoDB 数据目录。

## 沙箱故障处理

报错 `Failed to connect to sandbox` 通常表示沙箱进程没有监听 `5050`，不一定是 `sandbox_host` 配置错误。当前 Go Judge 使用固定的 systemd transient scope `gojudge.scope`；异常退出后残留该 scope 会导致新进程报：`Unit gojudge.scope already exists`。

启动包装脚本位于 `/home/dongwu.chen/.local/bin/hydro-sandbox-start`，启动前会清理残留 scope；systemd 单元停止时也会清理。

Ubuntu 24 当前启用了 `kernel.apparmor_restrict_unprivileged_userns=1`。包装脚本必须通过 `aa-exec -p rootlesskit --` 启动 Go Judge，否则会报 `fork/exec /proc/self/exe: permission denied`。手工恢复：

```bash
systemctl --user stop gojudge.scope || true
pm2 restart hydro-sandbox
pm2 save
ss -ltnp | grep -E ':5050\b'
curl -fsS http://127.0.0.1:5050/version
```

如果此前有提交因沙箱断连而失败，修复后需要在 Web 界面重新判题或重新提交；旧记录不会自动改写。

## systemd/PM2 恢复

如果 `systemctl --user status hydro-conda.service` 显示 `failed`，或者 `pm2 ls` 为空：

```bash
systemctl --user reset-failed hydro-conda.service
systemctl --user start hydro-conda.service
pm2 ls
```

日志中若出现 `No space left on device`，先检查磁盘空间和 inode：

```bash
df -h /data0/shared/dongwu.chen
df -ih /data0/shared/dongwu.chen
du -sh /data0/shared/dongwu.chen/.hydro-conda/*
```

优先归档并轮换过大的 PM2/沙箱日志；如果是系统 inotify 限制，需要管理员调整内核参数，普通用户无法用 sudo 修复。

## 数据备份

MongoDB 没有使用系统服务，数据在用户目录。备份前停止服务以得到一致快照：

```bash
backup_dir="/data0/shared/dongwu.chen/.hydro-conda/backups/hydro-$(date +%F-%H%M%S)"
mkdir -p "$backup_dir"
systemctl --user stop hydro-conda.service || true
pm2 ls  # 确认 hydro-conda、hydro-sandbox、mongo-conda 均已停止
tar -C /data0/shared/dongwu.chen/.hydro-conda -czf "$backup_dir/data.tar.gz" mongodb file
systemctl --user reset-failed hydro-conda.service || true
systemctl --user start hydro-conda.service
```

备份至少应包含 `mongodb` 和 `file` 两个目录。恢复 MongoDB 数据前必须先停止 `mongo-conda`，并确认目标目录为空或已另行保存。

## Hydro 管理命令

在仓库根目录并使用上述环境变量时，CLI 入口为 `node_modules/.bin/hydrooj`：

```bash
# 查看 CLI 帮助
node_modules/.bin/hydrooj cli --help

# 按 UID 设置超级管理员（需要把 UID 替换为实际注册用户 UID）
node_modules/.bin/hydrooj cli user setSuperAdmin <UID>
```

执行管理命令后重新登录 Web 界面，使权限缓存和会话刷新。

## 配置文件

- `/home/dongwu.chen/.hydro/profiles/conda-local/config.json`：MongoDB URI
- `/home/dongwu.chen/.hydro/profiles/conda-local/addon.json`：`ui-default` 和 `hydrojudge` 插件
- `/home/dongwu.chen/.hydro/profiles/conda-local/env`：`DEFAULT_STORE_PATH`
- `/home/dongwu.chen/.config/systemd/user/hydro-conda.service`：用户级服务定义
- `/home/dongwu.chen/.config/hydro-conda/ecosystem.config.js`：PM2 三进程定义
- `/home/dongwu.chen/.local/bin/hydro-sandbox-start`：沙箱启动包装脚本

修改 profile 或 PM2 进程后，先执行一次健康检查，再运行 `pm2 save`；不要把用户目录中的运行时数据、日志或凭据提交到 Git。

## GPU 算子题测评约定

新增或调整 GPU 算子题时，统一沿用以下计时和配置规则：

- target 与 PyTorch baseline 必须在同一测试点内成对交错测速，轮流采用 `target -> baseline` 和 `baseline -> target` 顺序；禁止固定先测完一方再测另一方。
- target 与 baseline 的 `OUTPUT`、`INOUT` 参数必须使用独立副本，并交错分配可变输入，避免显存地址和分配阶段系统性偏向其中一方。
- 数据生成及输入克隆完成后同步一次，预热完成后同步一次；计时使用同一默认 CUDA stream 上的独立 CUDA Event 对，全部计时操作入队后再统一同步并读取 Event。checker 使用的新副本也要在执行前后明确同步。
- 数据生成、输入克隆、容器启动、编译和 checker 不计入 kernel 时间。提交不得自行调用 `cudaDeviceSynchronize()`，也不得把主要工作放到 runner 无法计时的其他 stream。
- 默认编译使用 `nvcc -O3 --use_fast_math --extra-device-vectorization`、`ptxas -O3` 和 host `-O3`。出题时 checker 的容差必须覆盖 fast math 允许的数值差异，同时仍能拒绝错误算法。
- 每题至少进行多轮预热和多轮测速；根据 `INOUT`/`OUTPUT` 克隆的显存开销设置合理的 warmup、repeats 和 `memory`，并在当前实际 GPU 上验证全部测试点。
- 性能分继续使用 PyTorch baseline、理论硬件下限和 XPUOJ 对齐公式；上线前必须检查交换测速起始顺序后分数没有明显系统性漂移。
