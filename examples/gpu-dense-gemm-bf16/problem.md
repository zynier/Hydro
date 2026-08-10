## 题目描述

给定三个位于 GPU 上、按行优先连续存储的 bfloat16 矩阵：

- $A$ 的形状为 $(M, K)$；
- $B$ 的形状为 $(N, K)$；
- 输出 $C$ 的形状为 $(M, N)$。

请计算：

$$
C = A B^T, \qquad C_{i,j}=\sum_{k=0}^{K-1}A_{i,k}B_{j,k}.
$$

本题复现 [XPUOJ 第 2 题 Dense GeMM (bf16)](https://xpuoj.com/p/2) 的算子、数据类型和测试点 shape。

## CUDA 接口约定

CUDA C++ 提交必须导出以下 C 符号：

```cpp
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" void run_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int64_t M,
    int64_t N,
    int64_t K
);
```

提交中不要定义 `main`。`A`、`B` 是只读输入，结果必须写入 `C`。评测程序负责同步；`run_kernel` 不得调用 `cudaDeviceSynchronize()`，也不得将主要工作放到评测无法计时的其他 stream。

## TileLang 接口约定

TileLang 0.1.13 提交是一份 Python 源码，必须提供：

```python
def run_kernel(A, B, C, M, N, K):
    ...
```

`A`、`B`、`C` 是当前 GPU 上的连续 `torch.bfloat16` tensor，`M`、`N`、`K` 是 Python `int`。请缓存 shape 特化后的 JIT callable；每个测试点在独立进程中执行，首次 JIT 由不计时的预热吸收。可参考 [TileLang 矩阵乘教程](https://tilelang.com/deeplearning_operators/matmul.html)，使用 `T.Pipelined`、`T.copy` 和 `T.gemm(..., transpose_B=True)` 实现。

## TIRx 接口约定

TIRx 0.25.0.post1 提交同样必须提供 `run_kernel(A, B, C, M, N, K)`。使用 `tvm.tirx` 构建并缓存 executable，直接接收 PyTorch CUDA tensor；设备编译由不计时的预热吸收。算子必须使用默认 CUDA stream，不得显式同步。

## Triton 接口约定

Triton 3.7.1 提交同样必须提供 `run_kernel(A, B, C, M, N, K)`。请用 `triton.jit` 实现并通过标准 launch 语法排入默认 CUDA stream。首次 JIT 由不计时的预热吸收，不得显式同步。

## 输入与输出

本题没有标准输入输出。评测程序直接生成参数并按 `A, B, C, M, N, K` 的顺序调用 `run_kernel`。`A`、`B` 不允许修改，`C` 必须写满矩阵乘结果。

## 测试点

| 测试点 | $A(M,K)$ | $B(N,K)$ | $C(M,N)$ | 预热 | 交错测速 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | `(6400, 1792)` | `(6400, 1792)` | `(6400, 6400)` | 3 | 30 |
| 2 | `(20480, 3072)` | `(2048, 3072)` | `(20480, 2048)` | 3 | 30 |
| 3 | `(1792, 2816)` | `(18176, 2816)` | `(1792, 18176)` | 3 | 30 |
| 4 | `(9248, 1248)` | `(6176, 1248)` | `(9248, 6176)` | 3 | 30 |
| 5 | `(113408, 5120)` | `(128, 5120)` | `(113408, 128)` | 3 | 30 |

正确性检查使用 PyTorch 的 `C.copy_(torch.matmul(A, B.T))` 作为参考，并以 `rtol=1e-2`、`atol=1e-2` 比较 bfloat16 输出。

## 评测与计分

每个测试点独立执行。target 与 PyTorch baseline 使用独立的 `C` 副本成对交错测速，轮流采用 `target -> baseline` 和 `baseline -> target` 顺序。数据生成、输入克隆、容器启动、编译和 checker 均不计入 kernel 时间。

CUDA 使用 `nvcc -O3 --use_fast_math --extra-device-vectorization` 编译。TileLang、TIRx、Triton 的 JIT 均由预热吸收，之后与 CUDA 使用相同的默认 stream CUDA Event 计时。

设 target、PyTorch baseline 和 roofline 理论下限分别为 $T_k$、$T_b$、$T_h$，单点性能分为：

$$
S(T_k)=\frac{100}{1+\frac{T_k-T_h}{T_b-T_h}}.
$$

$T_k=T_b$ 时为 50 分，$T_k=T_h$ 时为 100 分。超过 100 的原始分数会对数压缩并封顶 150 分；五个测试点等权平均，错误或异常测试点计 0 分。

本题按 $2MNK$ 次浮点运算统计，最低显存流量按读取 $A$、读取 $B$、写入 $C$ 的 bfloat16 字节数统计。

## PyTorch 参考实现

```python
def baseline(A, B, C, M, N, K):
    C.copy_(torch.matmul(A, B.T))
```
