## 题目描述

实现 Top-k 概率重归一化。对 `probs` 的每一行保留数值最大的前 `top_k[row]` 个元素，将其余元素置零，再将保留值除以它们的和。输出保留原始类别顺序。

本题复现 [XPUOJ 第 4 题 Top k Renorm Probs](https://xpuoj.com/p/4) 的算子、数据类型和测试点 shape。

## CUDA 接口约定

提交 CUDA C++ 源码并导出以下 C 符号：

```cpp
#include <stdint.h>

extern "C" void run_kernel(
    const float* probs,
    const int32_t* top_k,
    float* renorm_probs,
    int64_t batch_size,
    int64_t num_classes
);
```

`probs` 和 `top_k` 是只读的连续 CUDA tensor，`renorm_probs` 是输出 tensor。提交中不要定义 `main`。`run_kernel` 必须将工作排入默认 CUDA stream，不得调用 `cudaDeviceSynchronize()`，也不得将主要工作放到评测无法计时的其他 stream。

## TileLang、TIRx 和 Triton

本题也支持现有 GPU 算子题语言 `tilelang`（0.1.13）、`tirx`（Apache TVM 0.25.0.post1）和 `triton`（3.7.1）。这些提交必须提供同名的 `run_kernel(probs, top_k, renorm_probs, batch_size, num_classes)`，接收当前 GPU 上的连续 `torch.float32` / `torch.int32` tensor 和 Python `int` 标量，并把工作排入默认 CUDA stream。请缓存 shape 特化后的 JIT kernel；首次编译由不计时预热吸收。禁止显式同步。

## 输入与输出

评测直接在 GPU 上生成参数，不使用标准输入输出：

- `probs`：`float32`，shape 为 `(batch_size, num_classes)`，每行均为概率分布；
- `top_k`：`int32`，shape 为 `(batch_size,)`，每个值位于 `[1, num_classes)`；
- `renorm_probs`：待写入的 `float32` 输出，shape 与 `probs` 相同；
- `batch_size`、`num_classes`：对应的 `int64` 标量。

`probs` 与 `top_k` 不得修改。输出每行必须恰有 `top_k[row]` 个非零项且总和为 1，并通过与 PyTorch 参考结果的数值比较。

## 测试点

| 测试点 | `probs` / `renorm_probs` | `top_k` | 预热 | 交错测速 |
| :-: | :-: | :-: | :-: | :-: |
| 1 | `(5120, 1024)` | `(5120,)` | 3 | 21 |
| 2 | `(17152, 2048)` | `(17152,)` | 3 | 24 |
| 3 | `(32768, 4096)` | `(32768,)` | 3 | 39 |
| 4 | `(7144, 4906)` | `(7144,)` | 3 | 20 |
| 5 | `(70400, 1024)` | `(70400,)` | 3 | 51 |

每个测试点独立执行。数据生成和输入克隆后会先同步，随后完成多轮预热。目标实现与 PyTorch baseline 在同一默认 CUDA stream 上成对交错测速，每轮轮换 `target -> baseline` 与 `baseline -> target` 的起始顺序；所有 CUDA Event 入队后统一同步。正确性检查使用独立的新副本，不计入 kernel 时间。

## PyTorch 参考实现

```python
k = top_k.reshape(batch_size, 1).to(dtype=torch.long)
max_k = int(k.max().item())
top_values, top_indices = torch.topk(
    probs, k=max_k, dim=-1, largest=True, sorted=True,
)
mask = torch.arange(max_k, device=probs.device).reshape(1, max_k) < k
filtered = top_values * mask.to(top_values.dtype)
denominator = filtered.sum(dim=-1, keepdim=True)
normalized = torch.where(
    denominator > 0, filtered / denominator, torch.zeros_like(filtered),
)
renorm_probs.zero_()
renorm_probs.scatter_(dim=-1, index=top_indices, src=normalized)
```

CUDA 使用 `nvcc -O3 --use_fast_math --extra-device-vectorization` 编译。计分使用 Hydro GPU 算子题的 XPUOJ/SOL 锚点公式；正确测试点等权平均，匹配 PyTorch baseline 为 50 分，达到理论硬件下限为 100 分。
