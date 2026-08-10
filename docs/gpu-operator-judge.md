# GPU operator judge

Hydro's `gpu` problem type follows the XPUOJ operator workflow: trusted problem data generates CUDA tensors, defines a PyTorch baseline and checker, and describes the work used for roofline scoring. CUDA submissions export one C entry point, while TileLang 0.1.13, Apache TVM/TIRx 0.25.0.post1, and Triton 3.7.1 submissions export the same entry point as a Python function. All languages are measured on an exclusively scheduled NVIDIA GPU.

## Prerequisites

- NVIDIA driver and NVIDIA Container Toolkit.
- An OCI runtime such as Docker.
- A development image containing PyTorch with CUDA support, Python 3, `nvcc`, Nsight Compute 2025.4.1, TileLang 0.1.13, Apache TVM/TIRx 0.25.0.post1, and Triton 3.7.1.

The default base image is `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel`. Build the bundled image to add the pinned Nsight Compute, TileLang, TIRx, and Triton runtimes:

```bash
docker build -t hydro-gpu-tirx:0.25.0.post1 packages/hydrojudge/gpu
```

Set `hydrojudge.gpu.container_image` to that tag and verify the runtime before enabling submissions:

```bash
docker run --rm --gpus all hydro-gpu-tirx:0.25.0.post1 \
  sh -lc 'python3 -c "import torch, tilelang; print(torch.__version__, tilelang.__version__)" && python3 -c "import torch, tvm, tvm.tirx; print(tvm.__version__)" && python3 -c "import torch, triton; print(triton.__version__, torch.cuda.get_device_name())"'
```

TileLang and Apache TVM ship different TVM compiler libraries, so the verification intentionally imports them in separate Python processes. Triton is verified separately as well. The judge loads only the selected language in each fresh testcase container.

To use a different compatible base image:

```bash
docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel \
  -t hydro-gpu-tirx-custom packages/hydrojudge/gpu
```

Set `hydrojudge.gpu.container_image` to that tag. The other relevant judge settings are `runtime`, `nvidia_smi`, `nvcc`, `lock_dir`, `compile_timeout`, `execution_timeout`, `cpus`, `memory`, and `pids_limit`.

Nsight Compute profiling is enabled by default for GPU problems. Its settings are:

- `profile_enabled`: generate profiles after the clean timing pass.
- `profile_ncu`: NCU executable inside the image, defaulting to `ncu`.
- `profile_set`: NCU section set, defaulting to `full`.
- `profile_timeout`: per-testcase profile timeout in milliseconds.
- `profile_measure_run`: measured target invocation to reproduce and profile.
- `profile_max_instances`: maximum raw metric instance values copied into the web summary. The `.ncu-rep` remains complete.
- `profile_tmpfs`: executable temporary filesystem size used by NCU and the TileLang/TIRx/Triton JITs.

## Problem files

A GPU problem has a `config.yaml` and a trusted Python testcase definition. It does not need `.in` or `.out` files.

```yaml
type: gpu
langs: [cuda, tilelang, tirx, triton]
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

## TIRx entry point

TIRx submissions are Python source files with the same configured entry function and ordinary PyTorch CUDA tensor/scalar arguments. The image and runner pin Apache TVM exactly to `apache-tvm==0.25.0.post1` and verify both that version and the `tvm.tirx` module before importing the submission. A typical submission defines a `@T.prim_func` using `from tvm.script import tirx as T`, compiles it with `tvm.compile(..., target=tvm.target.Target("cuda"), tir_pipeline="tirx")`, caches the returned executable, and invokes it directly with the PyTorch tensors.

As with TileLang, the compile container performs a syntax-only check, while device-specific TIRx compilation happens on the first execution-container invocation and is absorbed by warmup. TIRx executables accept PyTorch CUDA tensors through the TVM FFI without a data copy. They must enqueue work on the default CUDA stream and must not synchronize explicitly or move measured work to another stream.

## Triton entry point

Triton submissions are Python source files and must define the configured entry function (default `run_kernel`). The function receives ordinary PyTorch CUDA tensors and Python scalar arguments. A Triton kernel is launched on the default stream through the normal Triton call syntax:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(a, b, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(a + offsets, tl.load(a + offsets, mask=mask) + tl.load(b + offsets, mask=mask), mask=mask)

def run_kernel(A, B, numel):
    add_kernel[(triton.cdiv(numel, 1024),)](A, B, numel, BLOCK=1024, num_warps=4)
```

The image and runner require exactly `triton==3.7.1`. The compile container performs a syntax-only Python check; Triton's device-specific JIT compilation occurs on the first execution-container invocation and is absorbed by the un-timed warmup. The entry function must enqueue work on PyTorch's default CUDA stream, must not call `torch.cuda.synchronize()`, and must not move the measured kernel to another stream. Cache any shape-specialized launch configuration or compiled wrapper when the same testcase is invoked repeatedly.

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

The CUDA runtime probe derives memory bandwidth and FP32 throughput from device properties. For FP16/BF16 workloads, the device compute capability selects an architecture-specific dense Tensor Core rate relative to the probed FP32 rate. Sparse Tensor Core throughput is not used. If the compute capability is unavailable or unknown, Hydro falls back to twice the FP32 rate.

## Nsight Compute profiles

The clean pass completes first and remains the only source of timing and score data. Hydro then keeps the assigned GPU lease and runs one independent NCU pass per testcase. The selected measured target invocation is enclosed in an NVTX push/pop range; baseline work, correctness checking, input generation, and compilation are outside that range. CUDA profile builds add `-lineinfo` but not `-G`. TileLang, TIRx, and Triton compilation are absorbed by the same warmup sequence used by the clean pass before the selected invocation.

The record page exposes a stable profile link for every testcase as soon as clean judging finishes. The profile page refreshes while collection is pending, then shows clean timing, launch configuration, occupancy, full NCU details sections, raw metrics and instance values, rules and focus metrics, source markers, imported source, CUDA/PTX/SASS output, warnings, and session information. The original `.ncu-rep` can be downloaded for Nsight Compute GUI. Reports remain associated with the matching record history revision if a rejudge starts while collection is still running.

Profile artifacts use the judge-only upload endpoint and are stored under the Hydro storage namespace. Web access and signed report downloads reuse record-detail visibility checks. Contest projections remove profile data whenever testcase details are hidden.

## Scheduling and isolation

Hydro waits for a matching device with no active compute process and enough free memory. A per-UUID atomic lock prevents concurrent assignment by multiple Hydro processes.

Submission compilation and execution run in a read-only, network-disabled container with capabilities dropped, a PID limit, CPU/memory limits, and access to only the assigned GPU UUID. Profile execution also mounts trusted code and scripts read-only, with a separate writable artifact directory. The ordinary Go Judge path remains unchanged for non-GPU problems.

The runner detects rootless Docker/Podman automatically. Rootless runtimes use the NVIDIA CDI device name (`nvidia.com/gpu=<UUID>`) and the runtime's mapped root user; rootful runtimes retain the legacy `--gpus device=<UUID>` path and the Hydro service uid/gid. Both modes keep the same capability, filesystem, network, and device isolation.

The displayed memory value is the benchmark process's peak CPU RSS, matching XPUOJ's reporting convention. The YAML `memory` value separately controls GPU scheduling. Result payloads are authenticated with a per-process random nonce and SHA-256 before Hydro accepts their timing or checker data.

The importable example is in `examples/gpu-a-plus-b`:

```bash
HYDRO_PROFILE=conda-local node_modules/.bin/hydrooj cli script gpuExample \
  '{"domainId":"system","pid":"GPU1","owner":1}'
```

The command refuses to overwrite an existing problem unless `"force": true` is supplied.
