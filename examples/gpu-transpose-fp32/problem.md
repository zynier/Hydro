## 题目描述

给定连续存储的二维 `float32` 张量 `input`，其形状为 $(H, W)$。请将它转置到连续存储、形状为 $(W, H)$ 的 `output`：

$$
output[w, h] = input[h, w].
$$

本题复现 [XPUOJ 第 100 题 Transpose](https://xpuoj.com/p/100) 的算子、数据类型和全部测试点 shape。

## CUDA 接口约定

提交 CUDA C++ 源码并导出以下 C 符号，函数名、参数类型和顺序必须完全一致：

```cpp
#include <stdint.h>

extern "C" void run_kernel(
    const float* input,
    float* output,
    int64_t H,
    int64_t W
);
```

`input` 是只读的 `(H, W)` CUDA tensor，`output` 是待写满的 `(W, H)` CUDA tensor。提交中不要定义 `main`。`run_kernel` 需要自行计算 launch 配置并将工作排入默认 CUDA stream；禁止调用 `cudaDeviceSynchronize()`，也不得把主要工作放到其他 stream。

## TileLang 接口约定

TileLang 0.1.13 提交是一份 Python 源码，必须提供 `run_kernel(input, output, H, W)`。tensor 和标量会以普通 PyTorch CUDA tensor / Python `int` 传入。推荐按 shape 缓存 JIT callable，并使用带边界保护的 shared-memory tile 完成合并读写：

```python
import tilelang.language as T
from tilelang import jit

real_kernel = None

@jit
def build_kernel(H, W):
    @T.prim_func
    def kernel(
        input: T.Tensor((H, W), "float32"),
        output: T.Tensor((W, H), "float32"),
    ):
        ...
    return kernel

def run_kernel(input, output, H, W):
    global real_kernel
    if real_kernel is None:
        real_kernel = build_kernel(int(H), int(W))
    real_kernel(input, output)
```

首次 JIT 编译由不计时的预热吸收。请勿显式同步 GPU。

## TIRx 接口约定

TIRx 0.25.0.post1 提交也是 Python 源码，必须使用 `tvm.tirx` 编译链路并提供 `run_kernel(input, output, H, W)`。编译后的 executable 可直接接收 PyTorch CUDA tensor。每个测试点在独立进程中执行，请缓存 shape 特化后的 kernel；首次 JIT 由预热吸收。算子必须使用默认 CUDA stream，不能显式同步。

## Triton 接口约定

Triton 3.7.1 提交必须提供 `run_kernel(input, output, H, W)`，并通过标准 `triton.jit` launch 语法在默认 CUDA stream 上启动 kernel。输入输出均为连续 `torch.float32` CUDA tensor，`H` 和 `W` 为 Python `int`。首次 JIT 由预热吸收；禁止显式同步或使用其他 stream 规避计时。

## 输入与输出

本题没有标准输入输出。评测程序在 GPU 上生成参数并直接调用 `run_kernel`。必须写满 `output` 且不得修改 `input`。转置不改变数值，checker 会逐元素精确比较输出，并单独检查只读输入。

## 测试点

| 测试点 | `input` | `output` | 预热 | 交错测速 |
| :-: | :-: | :-: | :-: | :-: |
| 1 | `(8192, 16384)` | `(16384, 8192)` | 3 | 15 |
| 2 | `(4096, 32768)` | `(32768, 4096)` | 3 | 15 |
| 3 | `(6145, 24577)` | `(24577, 6145)` | 3 | 15 |
| 4 | `(16383, 8193)` | `(8193, 16383)` | 3 | 15 |
| 5 | `(2049, 65537)` | `(65537, 2049)` | 3 | 15 |
| 6 | `(32769, 4097)` | `(4097, 32769)` | 3 | 15 |
| 7 | `(12289, 12287)` | `(12287, 12289)` | 3 | 15 |
| 8 | `(3073, 49153)` | `(49153, 3073)` | 3 | 15 |
| 9 | `(513, 262145)` | `(262145, 513)` | 3 | 15 |
| 10 | `(24577, 6145)` | `(6145, 24577)` | 3 | 15 |

shape 与 XPUOJ 第 100 题保持一致，其中测试点 3-10 专门覆盖 tile 边界。

## PyTorch 参考实现

```python
def baseline(input, output, H, W):
    output.copy_(torch.transpose(input, 0, 1).contiguous())
```

## 评测与计分

每个测试点独立执行。数据生成及输入克隆结束后同步一次，预热结束后同步一次；之后 target 与 PyTorch baseline 使用独立 CUDA Event 对成对交错测速，并轮流交换先后顺序。正确性使用另一组独立副本检查，数据生成、容器启动、JIT/编译与 checker 均不计入 kernel 时间。

CUDA 使用 `nvcc -O3 --use_fast_math --extra-device-vectorization` 编译；TileLang、TIRx 和 Triton 的 JIT 均由预热吸收。计分采用 XPUOJ/SOL 锚点公式：匹配 PyTorch baseline 为 50 分，达到当前 GPU 理论硬件下限为 100 分，超过 100 后对数压缩并封顶 150 分。十个测试点等权平均，错误或异常测试点计 0 分。

理论访存量按读取 `input` 和写入 `output` 计算，共 8 字节/元素。
