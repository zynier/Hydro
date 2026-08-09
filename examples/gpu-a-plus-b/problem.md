## 题目描述

给定 GPU 上两个形状相同、连续存储的 fp16 张量 $A$ 和 $B$，请实现原地逐元素加法：

$$
A_i \leftarrow A_i + B_i
$$

## CUDA 接口约定

提交一份 CUDA C++ 源码，并导出以下 C 符号：

```cpp
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" void run_kernel(__half* A, const __half* B, int64_t numel);
```

- `A`：输入/输出张量的设备指针，必须原地写回。
- `B`：只读输入张量的设备指针，不允许修改。
- `numel`：张量元素总数。

提交代码中不要定义 `main`。`run_kernel` 负责计算 grid/block 并启动 CUDA kernel。评测程序统一处理同步，请不要在入口函数中调用 `cudaDeviceSynchronize()`。

## TileLang 接口约定

提交时也可以选择 TileLang 0.1.13 语言。此时提交内容是一份 Python 源码，不需要定义 `main`，但必须提供可调用的 `run_kernel(A, B, numel)`：

- `A`、`B` 是位于当前 GPU 上的连续 `torch.float16` tensor，形状见下表；
- `numel` 是 Python `int`；
- `run_kernel` 必须把计算排入 PyTorch 的默认 CUDA stream。

下面是与评测接口匹配的 TileLang 提交框架，其中 kernel 主体需要自行实现：

```python
import tilelang
import tilelang.language as T
from tilelang import jit

real_kernel = None

@jit
def build_kernel(...):
    @T.prim_func
    def kernel(
        a: T.Tensor((n,), "float16"),
        b: T.Tensor((n,), "float16"),
    ):
        pass

    return kernel

def run_kernel(a, b, numel):
    global real_kernel
    if real_kernel is None:
        real_kernel = build_kernel(...)
    real_kernel(a.view(-1), b.view(-1))
```

每个测试点在独立进程中执行，评测会多次调用 `run_kernel`。请像上例一样缓存 JIT 后的 callable，不要在每次调用时重新编译。提交阶段只检查 Python 语法；模块导入和 TileLang JIT 在测试点运行阶段完成，第一次调用的 JIT 编译由不计时的预热吸收。请勿调用 `torch.cuda.synchronize()`，也不要把主要计算放到其他 CUDA stream。

## TIRx 接口约定

提交时也可以选择 TIRx 0.25.0.post1 语言。提交内容是一份 Python 源码，必须使用 Apache TVM 中的 `tvm.tirx` 编译链路，并提供与其他语言相同的 `run_kernel(A, B, numel)`：

```python
from __future__ import annotations

import tvm
from tvm.script import tirx as T

compiled_kernel = None


def build_kernel(n):
    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (n,), "float16")
        B = T.match_buffer(B_ptr, (n,), "float16")

        T.device_entry()
        bx = T.cta_id([...])
        tx = T.thread_id([...])
        # 在这里实现算子。

    return tvm.compile(
        tvm.IRModule({"main": kernel}),
        target=tvm.target.Target("cuda"),
        tir_pipeline="tirx",
    )


def run_kernel(A, B, numel):
    global compiled_kernel
    if compiled_kernel is None:
        compiled_kernel = build_kernel(int(numel))
    compiled_kernel(A.view(-1), B.view(-1))
```

评测环境严格固定为 `apache-tvm==0.25.0.post1`，并检查 `tvm.tirx` 可导入。编译后的 TIRx executable 可直接接收 PyTorch CUDA tensor，无需复制。每个测试点都在独立进程中运行，请缓存编译结果；TIRx 编译由不计时的预热吸收。和 TileLang 一样，提交阶段只检查 Python 语法，算子必须使用默认 CUDA stream，不能显式同步，也不能把主要工作放到其他 stream。

## Triton 接口约定

提交时也可以选择 Triton `3.7.1`。提交内容是一份 Python 源码，不需要定义 `main`，但必须提供可调用的 `run_kernel(A, B, numel)`。`A`、`B` 是当前 GPU 上连续的 `torch.float16` tensor，`numel` 是 Python `int`。请使用 `triton.jit` 定义 kernel，并通过 Triton 的标准 launch 语法把工作排入默认 CUDA stream：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(a, b, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a_value = tl.load(a + offsets, mask=mask)
    b_value = tl.load(b + offsets, mask=mask)
    tl.store(a + offsets, a_value + b_value, mask=mask)

def run_kernel(A, B, numel):
    add_kernel[(triton.cdiv(numel, 1024),)](A, B, numel, BLOCK=1024, num_warps=4)
```

评测镜像严格固定为 `triton==3.7.1`，提交阶段只检查 Python 语法；Triton 的首次 JIT 编译在测试点运行阶段完成并由不计时的预热吸收。每个测试点在独立进程中执行，之后会重复调用 `run_kernel`，因此不要在每次调用中显式重新编译。禁止调用 `torch.cuda.synchronize()`，也不要把主要工作放到其他 CUDA stream。

## 输入与输出

本题没有标准输入输出。评测程序直接在 GPU 上生成 $A$、$B$，并调用 `run_kernel`。调用结束后，$A$ 应与 PyTorch 参考实现的结果一致，$B$ 应保持不变。

## 数据范围

| 测试点 | $A$、$B$ 的形状 | `numel` | 预热 | 交错测速 |
| :-: | :-: | :-: | :-: | :-: |
| 1 | `(8192, 8192)` | 67,108,864 | 3 | 30 |
| 2 | `(12288, 8192)` | 100,663,296 | 3 | 30 |
| 3 | `(16384, 8192)` | 134,217,728 | 3 | 30 |
| 4 | `(8192, 12288)` | 100,663,296 | 3 | 30 |
| 5 | `(12288, 12288)` | 150,994,944 | 3 | 30 |

所有张量均为 `torch.float16` 且连续存储。正确性检查使用 `torch.allclose`，容差为 `rtol=1e-5`、`atol=1e-6`。

## 评测与计分

每个测试点独立执行。评测先以不计时的多轮调用完成预热，再用 CUDA Event 对 target 与 PyTorch baseline 成对测速。每轮交替使用 `target -> baseline` 和 `baseline -> target` 顺序，并同时交错分配两者的可变输入，分别取平均时间 $T_k$ 和 $T_b$。数据生成和输入克隆完成后、预热完成后、全部计时 Event 入队后均会同步 GPU；正确性运行与测速运行相互独立。

CUDA 提交使用 `nvcc -O3 --use_fast_math --extra-device-vectorization` 编译。快速数学运算可能采用精度较低但更快的设备指令，最终结果仍须满足本题 checker 的误差要求。TileLang 提交固定为 `tilelang==0.1.13`，TIRx 提交固定为 `apache-tvm==0.25.0.post1`，Triton 提交固定为 `triton==3.7.1`；这些 Python DSL 的 JIT 都由预热吸收，之后的每次 `run_kernel` 调用与 CUDA 提交采用相同的 CUDA Event 计时和正确性检查。

设根据工作量和当前 GPU 峰值规格估算的理论下限为 $T_h$，测试点原始性能分为：

$$
S(T_k)=\frac{100}{1+\frac{T_k-T_h}{T_b-T_h}}
$$

$T_k=T_b$ 时为 50 分，$T_k=T_h$ 时为 100 分。超过 100 的原始分数按对数压缩，最终单点最高 150 分。五个测试点等权取算术平均；错误或异常测试点按 0 分计入平均值。

本题每个元素包含 1 次 fp16 加法；理论访存量按读取 $A$、读取 $B$、写回 $A$，共 6 字节计算。

## PyTorch 参考实现

```python
def baseline(A, B, numel):
    A.add_(B)
```
