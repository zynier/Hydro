# GPU operator judge

Hydro's `gpu` problem type follows the XPUOJ operator workflow: trusted problem data generates CUDA tensors, defines a PyTorch baseline and checker, and describes the work used for roofline scoring. CUDA submissions export one C entry point, while TileLang 0.1.13 submissions export the same entry point as a Python function. Both are measured on an exclusively scheduled NVIDIA GPU.

## Prerequisites

- NVIDIA driver and NVIDIA Container Toolkit.
- An OCI runtime such as Docker.
- A development image containing PyTorch with CUDA support, Python 3, `nvcc`, and TileLang 0.1.13 when TileLang submissions are enabled.

The default base image is `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel`. Build the bundled image to add the pinned TileLang runtime:

```bash
docker build -t hydro-gpu-tilelang:0.1.13 packages/hydrojudge/gpu
```

Set `hydrojudge.gpu.container_image` to that tag and verify the runtime before enabling submissions:

```bash
docker run --rm --gpus all hydro-gpu-tilelang:0.1.13 \
  python3 -c 'import torch, tilelang; print(torch.__version__, tilelang.__version__, torch.cuda.get_device_name())'
```

To use a different compatible base image:

```bash
docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel \
  -t hydro-gpu-tilelang-custom packages/hydrojudge/gpu
```

Set `hydrojudge.gpu.container_image` to that tag. The other relevant judge settings are `runtime`, `nvidia_smi`, `nvcc`, `lock_dir`, `compile_timeout`, `execution_timeout`, `cpus`, `memory`, and `pids_limit`.

## Problem files

A GPU problem has a `config.yaml` and a trusted Python testcase definition. It does not need `.in` or `.out` files.

```yaml
type: gpu
langs: [cuda, tilelang]
time: 120s
memory: 12g
gpu:
  entry: run_kernel
  testcase: testcase_config.py
  cases:
    - id: 1
    - id: 2
```

`entry` defaults to `run_kernel`, and `testcase` defaults to `testcase_config.py`. `time` limits the complete benchmark after compilation. `memory` is the minimum free GPU memory required by the scheduler. A case or the `gpu` block may override `warmup` and `repeats`; otherwise the Python definition supplies them, falling back to 100 warmups and 2000 measured runs.

Each testcase is equally weighted. There is no static case score in the YAML.

## Testcase definition

`testcase_config.py` uses the same core interface as the XPUOJ problem set:

- `getTestCaseSize()` reads the case id from standard input and returns argument shape descriptions. It may return `(sizes, (warmup, repeats))`.
- `genTestCase(sizes, device)` returns tensors and scalar arguments in entry-point order.
- `baseline(*args)` performs the trusted PyTorch reference computation.
- `check(sizes, original, target, baseline)` compares one independent submission run with one independent baseline run and returns a boolean.
- `getWorkload(sizes)` returns positive `flops`, positive `memory_bytes`, and a `dtype` string.
- `INPUT_CLASS` contains one of `INPUT`, `OUTPUT`, or `INOUT` for every argument. Mutable tensors are cloned for every timed invocation; read-only inputs are shared.

Tensor arguments must be contiguous CUDA tensors. They are passed to CUDA as device pointers. Python `int`, `float`, and `bool` values map to `int64_t`, `float`, and `bool` respectively.

## CUDA entry point

The submission must export an unmangled C symbol. For the bundled fp16 addition problem it is:

```cpp
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" void run_kernel(__half* A, const __half* B, int64_t numel);
```

The runner invokes this symbol directly. Submissions should launch work on the default CUDA stream and should not call `cudaDeviceSynchronize()`; the runner performs synchronization around warmup, timing, and checking.

Submissions are compiled with `nvcc -O3 --use_fast_math --extra-device-vectorization`, `ptxas -O3`, host-side `-O3`, and the exact `sm_XX` plus PTX fallback targets for the assigned GPU. Problem checkers must account for the numerical behavior permitted by CUDA fast math.

## TileLang entry point

TileLang submissions are Python source files and must define the configured entry function (default `run_kernel`) with the same positional arguments as the CUDA ABI. The arguments are ordinary PyTorch CUDA tensors/scalars, so a submission follows the XPUOJ shape:

```python
import tilelang
import tilelang.language as T
from tilelang import jit

real_kernel = None

@jit
def add(*args):
    @T.prim_func
    def kernel(*args):
        ...
    return kernel

def run_kernel(A, B, numel):
    global real_kernel
    if real_kernel is None:
        real_kernel = add(...)
    real_kernel(A, B, numel)
```

The image must contain exactly `tilelang==0.1.13`; the runner checks this before importing the submission. A syntax-only Python compile is performed in the compile container. TileLang's shape-specialized JIT compilation occurs on the first invocation in the execution container and is therefore absorbed by the un-timed warmup. Subsequent invocations use the compiled callable. The function must enqueue work on PyTorch's default CUDA stream and must not call `cudaDeviceSynchronize()`.

## Measurement and correctness

Each case runs in a fresh read-only container so CUDA state, allocator caches, failures, and memory accounting cannot leak into the next case. The runner:

1. Generates deterministic inputs for the case.
2. Prepares paired independent mutable inputs for target and baseline. Allocation order alternates between the two implementations to balance GPU-address placement effects.
3. Synchronizes after input preparation, performs interleaved warmups, then synchronizes again.
4. Measures target and baseline with separate CUDA Event pairs in alternating `target -> baseline` and `baseline -> target` order. After all measured operations are queued, one synchronization makes every Event result readable; `T_k` and `T_b` are the respective sample averages.
5. Releases benchmark inputs, creates fresh correctness inputs, synchronizes their clones, runs the submission and PyTorch baseline once more, synchronizes, then calls the problem's `check()`.

Data generation, cloning, host/device setup, checking, compilation, and container startup are outside the reported kernel time. A failed check or failed execution gives that case zero points.

## Scoring

The theoretical lower bound `T_h` is the larger roofline component:

```text
max(workload.flops / peak_compute_throughput,
    workload.memory_bytes / peak_memory_bandwidth)
```

For a correct case, Hydro uses the XPUOJ/SOL-style anchor formula:

```text
raw = 100 / (1 + (T_k - T_h) / (T_b - T_h))
```

Matching PyTorch is 50 points and matching the hardware estimate is 100 points. Raw scores above 100 are displayed with logarithmic compression (`100 + 10 * log2(raw / 100)`) and capped at 150. The submission score is the arithmetic mean of all case scores, including zeros for failed cases; total time is the sum of all `T_k` values.

The CUDA runtime probe derives memory bandwidth and FP32 throughput from device properties. FP16/BF16 work uses twice the probed FP32 rate, so the roofline is a stable comparison anchor rather than a promise of achievable tensor-core performance.

## Scheduling and isolation

Hydro waits for a matching device with no active compute process and enough free memory. A per-UUID atomic lock prevents concurrent assignment by multiple Hydro processes.

Submission compilation and execution run in a read-only, network-disabled container with capabilities dropped, a PID limit, CPU/memory limits, and access to only the assigned GPU UUID. The ordinary Go Judge path remains unchanged for non-GPU problems.

The displayed memory value is the benchmark process's peak CPU RSS, matching XPUOJ's reporting convention. The YAML `memory` value separately controls GPU scheduling. Result payloads are authenticated with a per-process random nonce and SHA-256 before Hydro accepts their timing or checker data.

The importable example is in `examples/gpu-a-plus-b`:

```bash
HYDRO_PROFILE=conda-local node_modules/.bin/hydrooj cli script gpuExample \
  '{"domainId":"system","pid":"GPU1","owner":1}'
```

The command refuses to overwrite an existing problem unless `"force": true` is supplied.
