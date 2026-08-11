## 题目描述

给定两个按行优先连续存储的 `float32` 矩阵 `X_real` 和 `X_imag`，形状均为 $(batch,n)$。它们分别表示复数输入的实部和虚部：

$$
X_{b,t}=X_{real,b,t}+iX_{imag,b,t}.
$$

请沿最后一个维度对每一行计算一维离散傅里叶变换：

$$
Y_{b,k}=\sum_{t=0}^{n-1}X_{b,t}e^{-2\pi i k t/n},
$$

并把结果的实部和虚部分别写入 `Y_real` 和 `Y_imag`。四个矩阵的形状均为 $(batch,n)$。

本题复现 [XPUOJ 第 96 题 FFT](https://xpuoj.com/p/96) 的算子、数据类型和全部测试点 shape。

## CUDA 接口约定

提交 CUDA C++ 源码并导出以下 C 符号，函数名、参数类型和顺序必须完全一致：

```cpp
#include <stdint.h>

extern "C" void run_kernel(
    const float* X_real,
    const float* X_imag,
    float* Y_real,
    float* Y_imag,
    int64_t batch,
    int64_t n
);
```

`X_real`、`X_imag` 是只读 CUDA tensor；`Y_real`、`Y_imag` 是待写满的输出 tensor。提交中不要定义 `main`。可以使用 CUDA Toolkit 随附的设备库，但提交仍须通过上述入口把全部工作排入 PyTorch 默认 CUDA stream。禁止调用 `cudaDeviceSynchronize()`，也不得把主要工作放到评测无法计时的其他 stream。

## TileLang 接口约定

TileLang 0.1.13 提交是一份 Python 源码，必须提供：

```python
def run_kernel(X_real, X_imag, Y_real, Y_imag, batch, n):
    ...
```

tensor 参数是当前 GPU 上的连续 `torch.float32` tensor，`batch` 和 `n` 是 Python `int`。请按 shape 缓存 JIT callable；每个测试点在独立进程中运行，首次 JIT 编译由不计时的预热吸收。不得显式同步或使用其他 CUDA stream 规避计时。

## TIRx 接口约定

TIRx 0.25.0.post1 提交同样必须提供 `run_kernel(X_real, X_imag, Y_real, Y_imag, batch, n)`。使用 `tvm.tirx` 构建并缓存 shape 特化的 executable，直接接收 PyTorch CUDA tensor。编译由不计时的预热吸收；算子必须使用默认 CUDA stream，不得显式同步。

## Triton 接口约定

Triton 3.7.1 提交同样必须提供 `run_kernel(X_real, X_imag, Y_real, Y_imag, batch, n)`，并用 `triton.jit` 及标准 launch 语法把工作排入默认 CUDA stream。首次 JIT 由不计时的预热吸收；不得显式同步。

## 输入与输出

本题没有标准输入输出。评测程序直接在 GPU 上生成参数并调用 `run_kernel`。必须写满 `Y_real` 和 `Y_imag`，且不得修改 `X_real` 和 `X_imag`。

## 测试点

| 测试点 | `X_real` / `X_imag` | `Y_real` / `Y_imag` | `batch` | `n` | 预热 | 交错测速 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | `(8192, 257)` | `(8192, 257)` | 8192 | 257 | 3 | 15 |
| 2 | `(6144, 384)` | `(6144, 384)` | 6144 | 384 | 3 | 15 |
| 3 | `(4096, 513)` | `(4096, 513)` | 4096 | 513 | 3 | 15 |
| 4 | `(3072, 640)` | `(3072, 640)` | 3072 | 640 | 3 | 15 |
| 5 | `(2048, 1024)` | `(2048, 1024)` | 2048 | 1024 | 3 | 15 |
| 6 | `(2816, 768)` | `(2816, 768)` | 2816 | 768 | 3 | 15 |
| 7 | `(3584, 511)` | `(3584, 511)` | 3584 | 511 | 3 | 15 |
| 8 | `(2560, 897)` | `(2560, 897)` | 2560 | 897 | 3 | 15 |
| 9 | `(2816, 960)` | `(2816, 960)` | 2816 | 960 | 3 | 15 |
| 10 | `(2304, 1009)` | `(2304, 1009)` | 2304 | 1009 | 3 | 15 |

所有 shape 与 XPUOJ 第 96 题保持一致。正确性检查分别比较输出实部和虚部，使用 `rtol=1e-4`、`atol=1e-4`，并单独检查两个只读输入没有被修改。

## PyTorch 参考实现

```python
def baseline(X_real, X_imag, Y_real, Y_imag, batch, n):
    x = torch.complex(X_real, X_imag).to(torch.complex64)
    y = torch.fft.fft(x, dim=-1)
    Y_real.copy_(y.real.contiguous())
    Y_imag.copy_(y.imag.contiguous())
```

## 评测与计分

每个测试点独立执行。数据生成及输入克隆结束后同步一次，预热结束后同步一次；之后 target 与 PyTorch baseline 使用独立 CUDA Event 对成对交错测速，并轮流交换先后顺序。两者的输出使用独立副本。正确性使用另一组独立副本检查，数据生成、容器启动、编译、JIT 和 checker 均不计入 kernel 时间。

CUDA 使用 `nvcc -O3 --use_fast_math --extra-device-vectorization` 编译；TileLang、TIRx 和 Triton 的 JIT 均由预热吸收。计分采用 XPUOJ 对齐公式：匹配 PyTorch baseline 为 50 分，达到当前 GPU 理论硬件下限为 100 分，超过 100 后对数压缩并封顶 150 分。十个测试点等权平均，错误或异常测试点计 0 分。

FFT 工作量按每个复数一维变换约 $5n\log_2n$ 次实数运算估计；最低显存流量按读取两个输入和写入两个输出计算，共 16 字节/元素。
