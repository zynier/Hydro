# a += b（fp16）

## 题目描述

给定 GPU 上两个形状相同、连续存储的 fp16 张量 $A$ 和 $B$，请实现原地逐元素加法：

$$
A_i \leftarrow A_i + B_i
$$

## 接口约定

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

## 输入与输出

本题没有标准输入输出。评测程序直接在 GPU 上生成 $A$、$B$，并调用 `run_kernel`。调用结束后，$A$ 应与 PyTorch 参考实现的结果一致，$B$ 应保持不变。

## 数据范围

| 测试点 | $A$、$B$ 的形状 | `numel` | 预热 | 交错测速 |
| --- | --- | ---: | ---: | ---: |
| 1 | `(8192, 8192)` | 67,108,864 | 3 | 30 |
| 2 | `(12288, 8192)` | 100,663,296 | 3 | 30 |
| 3 | `(16384, 8192)` | 134,217,728 | 3 | 30 |
| 4 | `(8192, 12288)` | 100,663,296 | 3 | 30 |
| 5 | `(12288, 12288)` | 150,994,944 | 3 | 30 |

所有张量均为 `torch.float16` 且连续存储。正确性检查使用 `torch.allclose`，容差为 `rtol=1e-5`、`atol=1e-6`。

## 评测与计分

每个测试点独立执行。评测先以不计时的多轮调用完成预热，再用 CUDA Event 对 target 与 PyTorch baseline 成对测速。每轮交替使用 `target -> baseline` 和 `baseline -> target` 顺序，并同时交错分配两者的可变输入，分别取平均时间 $T_k$ 和 $T_b$。数据生成和输入克隆完成后、预热完成后、全部计时 Event 入队后均会同步 GPU；正确性运行与测速运行相互独立。

提交使用 `nvcc -O3 --use_fast_math --extra-device-vectorization` 编译。快速数学运算可能采用精度较低但更快的设备指令，最终结果仍须满足本题 checker 的误差要求。

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
