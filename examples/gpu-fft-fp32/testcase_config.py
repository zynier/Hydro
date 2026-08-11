from __future__ import annotations

import math
from typing import List, Tuple, Union

import torch

KernelArg = Union[torch.Tensor, int, float]

# XPUOJ problem 96 shapes. Every case uses independent target/baseline outputs.
TESTCASES = [
    (8192, 257, 3, 15),
    (6144, 384, 3, 15),
    (4096, 513, 3, 15),
    (3072, 640, 3, 15),
    (2048, 1024, 3, 15),
    (2816, 768, 3, 15),
    (3584, 511, 3, 15),
    (2560, 897, 3, 15),
    (2816, 960, 3, 15),
    (2304, 1009, 3, 15),
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    testcase_id = int(input())
    if testcase_id < 1 or testcase_id > len(TESTCASES):
        raise ValueError(f"Invalid testcase id: {testcase_id}")
    batch, n, warmup, repeats = TESTCASES[testcase_id - 1]
    shape = (batch, n)
    return [shape, shape, shape, shape, (), ()], (warmup, repeats)


def genTestCase(
    testcase_sizes: List[Tuple[int, ...]],
    device: str = "cuda",
) -> List[KernelArg]:
    x_real_shape, x_imag_shape, y_real_shape, y_imag_shape, batch_shape, n_shape = testcase_sizes
    if batch_shape != () or n_shape != ():
        raise ValueError("batch and n must be scalar arguments")
    if not (x_real_shape == x_imag_shape == y_real_shape == y_imag_shape):
        raise ValueError("All tensor arguments must have the same shape")
    batch, n = x_real_shape
    x_real = torch.randn(x_real_shape, dtype=torch.float32, device=device)
    x_imag = torch.randn(x_imag_shape, dtype=torch.float32, device=device)
    y_real = torch.empty(y_real_shape, dtype=torch.float32, device=device)
    y_imag = torch.empty(y_imag_shape, dtype=torch.float32, device=device)
    return [x_real, x_imag, y_real, y_imag, int(batch), int(n)]


INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "OUTPUT", "INPUT", "INPUT"]


def baseline(x_real, x_imag, y_real, y_imag, batch, n):
    if x_real.shape != (batch, n) or x_imag.shape != (batch, n):
        raise ValueError("Input shapes do not match batch and n")
    x = torch.complex(x_real, x_imag).to(torch.complex64)
    y = torch.fft.fft(x, dim=-1)
    y_real.copy_(y.real.contiguous())
    y_imag.copy_(y.imag.contiguous())


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
) -> bool:
    del testcase_sizes
    original_real, original_imag = original_input_tensors[:2]
    target_real, target_imag, target_y_real, target_y_imag, target_batch, target_n = (
        target_kernel_input_tensors
    )
    _, _, reference_y_real, reference_y_imag, reference_batch, reference_n = baseline_input_tensors

    if target_batch != reference_batch or target_n != reference_n:
        print("batch or n was not preserved")
        return False
    if not torch.equal(target_real, original_real) or not torch.equal(target_imag, original_imag):
        print("A read-only input tensor was modified")
        return False
    if target_y_real.shape != reference_y_real.shape or target_y_imag.shape != reference_y_imag.shape:
        print("Output shape mismatch")
        return False
    if target_y_real.dtype != torch.float32 or target_y_imag.dtype != torch.float32:
        print("Output dtype must be float32")
        return False

    real_ok = torch.allclose(target_y_real, reference_y_real, rtol=1e-4, atol=1e-4)
    imag_ok = torch.allclose(target_y_imag, reference_y_imag, rtol=1e-4, atol=1e-4)
    if not (real_ok and imag_ok):
        real_diff = float((target_y_real - reference_y_real).abs().max().item())
        imag_diff = float((target_y_imag - reference_y_imag).abs().max().item())
        print(f"FFT output differs from the baseline: real={real_diff:.6g}, imag={imag_diff:.6g}")
        return False
    return True


def getWorkload(testcase_sizes: List[Tuple[int, ...]]) -> dict:
    shape = testcase_sizes[0]
    if any(tensor_shape != shape for tensor_shape in testcase_sizes[:4]):
        raise ValueError("Invalid FFT tensor shapes")
    batch, n = shape
    # A complex FFT is conventionally estimated as 5*N*log2(N) real operations.
    return {
        "flops": batch * 5 * n * math.log2(n),
        "memory_bytes": batch * n * 4 * 4,
        "dtype": "fp32",
    }


DESIGNED_VRAM_SIZE = 48
