## 题目描述

实现 bfloat16 精度的批量外积算子，对应 einsum 表达式：

```python
out = torch.einsum('bi,bj->bij', a, b)
```

给定 `a`（形状 `(B, I)`）和 `b`（形状 `(B, J)`），对每个 batch 独立计算：

```text
out[batch, i, j] = a[batch, i] * b[batch, j]
```

`a`、`b`、`out` 均为连续的 `torch.bfloat16` CUDA tensor，`out` 必须完整写入，两个输入不得修改。

## CUDA 接口约定

提交 CUDA C++ 源码并导出以下 C 符号，函数名、参数类型和顺序必须完全一致：

```cpp
#include <stdint.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int64_t B,
    int64_t I,
    int64_t J
);
```

不要定义 `main`。入口函数只需把全部工作排入评测使用的默认 CUDA stream；禁止调用 `cudaDeviceSynchronize()`。

## TileLang、TIRx 和 Triton

TileLang 0.1.13、Apache TVM/TIRx 0.25.0.post1 和 Triton 3.7.1 提交都必须提供同名的 Python 函数：

```python
def run_kernel(a, b, out, B, I, J):
    ...
```

参数是当前 GPU 上的连续 bfloat16 tensor 和 Python `int` 标量。请缓存 shape 特化后的 JIT callable；首次编译由不计时预热吸收。算子必须使用默认 CUDA stream，不得显式同步或把主要工作移到其他 stream。

## 测试点

测试点 shape 与 [XPUOJ 第 139 题 Einsum Outer](https://xpuoj.com/p/139) 完全一致：

| 测试点 | `a` | `b` | `out` | `B` | `I` | `J` |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | `(2, 2048)` | `(2, 2048)` | `(2, 2048, 2048)` | 2 | 2048 | 2048 |
| 2 | `(4, 4096)` | `(4, 4096)` | `(4, 4096, 4096)` | 4 | 4096 | 4096 |
| 3 | `(8, 4096)` | `(8, 4096)` | `(8, 4096, 4096)` | 8 | 4096 | 4096 |
| 4 | `(8, 8192)` | `(8, 4096)` | `(8, 8192, 4096)` | 8 | 8192 | 4096 |
| 5 | `(16, 8192)` | `(16, 4096)` | `(16, 8192, 4096)` | 16 | 8192 | 4096 |
| 6 | `(8, 8192)` | `(8, 8192)` | `(8, 8192, 8192)` | 8 | 8192 | 8192 |
| 7 | `(16, 8192)` | `(16, 8192)` | `(16, 8192, 8192)` | 16 | 8192 | 8192 |
| 8 | `(12, 12288)` | `(12, 8192)` | `(12, 12288, 8192)` | 12 | 12288 | 8192 |
| 9 | `(16, 12288)` | `(16, 8192)` | `(16, 12288, 8192)` | 16 | 12288 | 8192 |
| 10 | `(8, 8193)` | `(8, 12289)` | `(8, 8193, 12289)` | 8 | 8193 | 12289 |

所有测试点预热 3 次并交错测速 15 次。正确性比较使用 `torch.allclose`，`rtol=1e-2`、`atol=1e-2`。

## 参考实现和计分

参考实现先把输入转为 float32 做乘法，再转换回 bfloat16：

```python
def baseline(a, b, out, B, I, J):
    chunk_i = max(1, min(I, 512))
    b_float = b.float()
    for start in range(0, I, chunk_i):
        end = min(start + chunk_i, I)
        product = a[:, start:end].float().unsqueeze(2) * b_float.unsqueeze(1)
        out[:, start:end, :].copy_(product.to(torch.bfloat16))
```

评测使用独立 CUDA Event 在默认 stream 上交错测量 target 和 PyTorch baseline；输入克隆、编译、预热、同步和 checker 均不计入 kernel 时间。计分沿用 XPUOJ 的性能锚点公式，十个测试点等权平均，错误测试点计 0 分。
