## 题目描述

实现 Top-p（Nucleus）概率重归一化。对 `probs` 的每一行按降序排序，保留累积概率不超过该行 `top_p` 的前缀（至少保留最大值），将其余概率置零，然后把保留值除以保留值之和并写回原始类别位置。

## CUDA 接口约定

提交 CUDA C++ 源码并导出以下 C 符号：

```cpp
#include <stdint.h>

extern "C" void run_kernel(
    const float* probs,
    const float* top_p,
    float* renorm_probs,
    int64_t batch_size,
    int64_t num_classes
);
```

`probs` 和 `top_p` 是只读的连续 CUDA tensor，`renorm_probs` 是输出 tensor。提交中不要定义 `main`，评测程序会负责 CUDA 同步；`run_kernel` 不得调用 `cudaDeviceSynchronize()`，也不得将主要工作放到评测无法计时的其他 stream。

## TileLang、TIRx 和 Triton

本题同样支持现有 GPU 算子题语言 `tilelang`（0.1.13）、`tirx`（Apache TVM 0.25.0.post1）和 `triton`（3.7.1）。这些提交必须提供同名的 `run_kernel(probs, top_p, renorm_probs, batch_size, num_classes)`，接收当前 GPU 上的连续 `torch.float32` tensor 和 Python `int` 标量，并把工作排入默认 CUDA stream。请缓存 shape 特化后的 JIT kernel；首次编译由不计时预热吸收。禁止显式同步。

## 输入与输出

评测直接在 GPU 上生成参数，不使用标准输入输出。`renorm_probs` 必须写满形状为 `(batch_size, num_classes)` 的 `float32` 矩阵，`probs` 与 `top_p` 不得修改。正确性以 `rtol=1e-2`、`atol=1e-2` 与 PyTorch 参考实现比较。

## 测试点

| 测试点 | `probs` / `renorm_probs` | `top_p` | 预热 | 交错测速 |
| :-: | :-: | :-: | :-: | :-: |
| 1 | `(1536, 11776)` | `(1536,)` | 7 | 146 |
| 2 | `(7936, 4096)` | `(7936,)` | 16 | 316 |
| 3 | `(256, 19712)` | `(256,)` | 16 | 305 |
| 4 | `(2316, 7712)` | `(2316,)` | 12 | 242 |
| 5 | `(33024, 1280)` | `(33024,)` | 11 | 222 |

标量参数分别为对应的 `batch_size` 和 `num_classes`。每个测试点独立执行，目标与 PyTorch baseline 在同一默认 stream 上成对交错测速。

## PyTorch 参考实现

```python
sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
cumsum = torch.cumsum(sorted_probs, dim=-1)
mask = cumsum <= top_p.reshape(batch_size, 1)
mask[:, 0] = True
filtered = sorted_probs * mask.to(sorted_probs.dtype)
denom = filtered.sum(dim=-1, keepdim=True)
renorm_sorted = torch.where(denom > 0, filtered / denom, torch.zeros_like(filtered))
renorm_probs.zero_()
renorm_probs.scatter_(dim=-1, index=sorted_idx, src=renorm_sorted)
```

计分使用 Hydro GPU 算子题的 XPUOJ/SOL 锚点公式；正确测试点等权平均，匹配 PyTorch baseline 为 50 分，达到理论硬件下限为 100 分。
