from __future__ import annotations

from typing import List, Tuple, Union

import torch

KernelArg = Union[torch.Tensor, int, float]

# XPUOJ problem 139 (Einsum Outer) test shapes.
TESTCASES = [
    (2, 2048, 2048, 3, 15),
    (4, 4096, 4096, 3, 15),
    (8, 4096, 4096, 3, 15),
    (8, 8192, 4096, 3, 15),
    (16, 8192, 4096, 3, 15),
    (8, 8192, 8192, 3, 15),
    (16, 8192, 8192, 3, 15),
    (12, 12288, 8192, 3, 15),
    (16, 12288, 8192, 3, 15),
    (8, 8193, 12289, 3, 15),
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    testcase_id = int(input())
    if testcase_id < 1 or testcase_id > len(TESTCASES):
        raise ValueError(f"Invalid testcase id: {testcase_id}")
    batch, size_i, size_j, warmup, repeats = TESTCASES[testcase_id - 1]
    return [(batch, size_i), (batch, size_j), (batch, size_i, size_j), (), (), ()], (warmup, repeats)


def genTestCase(
    testcase_sizes: List[Tuple[int, ...]],
    device: str = "cuda",
) -> List[KernelArg]:
    a_shape, b_shape, out_shape, batch_shape, i_shape, j_shape = testcase_sizes
    if batch_shape != () or i_shape != () or j_shape != ():
        raise ValueError("B, I, and J must be scalar arguments")
    if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 3:
        raise ValueError("Expected a[B,I], b[B,J], and out[B,I,J]")
    batch, size_i = a_shape
    b_batch, size_j = b_shape
    if b_batch != batch or out_shape != (batch, size_i, size_j):
        raise ValueError("Tensor shapes do not match B, I, and J")
    a = torch.randn(a_shape, dtype=torch.bfloat16, device=device)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device=device)
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=device)
    return [a, b, out, int(batch), int(size_i), int(size_j)]


INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"]


def baseline(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    batch: int,
    size_i: int,
    size_j: int,
):
    if a.shape != (batch, size_i) or b.shape != (batch, size_j) or out.shape != (batch, size_i, size_j):
        raise ValueError("Tensor shapes do not match B, I, and J")
    # Match the reference implementation's fp32 multiply before bf16 output conversion.
    chunk_i = max(1, min(size_i, 512))
    b_float = b.float()
    for start in range(0, size_i, chunk_i):
        end = min(start + chunk_i, size_i)
        product = a[:, start:end].float().unsqueeze(2) * b_float.unsqueeze(1)
        out[:, start:end, :].copy_(product.to(torch.bfloat16))


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
) -> bool:
    del testcase_sizes
    original_a, original_b = original_input_tensors[:2]
    target_a, target_b, target_out, target_batch, target_i, target_j = target_kernel_input_tensors
    _, _, reference_out, reference_batch, reference_i, reference_j = baseline_input_tensors
    if (target_batch, target_i, target_j) != (reference_batch, reference_i, reference_j):
        print("B, I, or J was not preserved")
        return False
    if not torch.equal(target_a, original_a) or not torch.equal(target_b, original_b):
        print("A read-only input tensor was modified")
        return False
    if target_out.shape != reference_out.shape or target_out.dtype != reference_out.dtype:
        print(f"Output shape/dtype mismatch: {target_out.shape}/{target_out.dtype} vs {reference_out.shape}/{reference_out.dtype}")
        return False
    if not torch.allclose(target_out, reference_out, rtol=1e-2, atol=1e-2):
        difference = (target_out.float() - reference_out.float()).abs()
        print(f"Output differs from the PyTorch baseline: max_abs_diff={float(difference.max().item()):.6g}")
        return False
    return True


def getWorkload(testcase_sizes: List[Tuple[int, ...]]) -> dict:
    a_shape, b_shape, out_shape, batch_shape, i_shape, j_shape = testcase_sizes
    if batch_shape != () or i_shape != () or j_shape != ():
        raise ValueError("Invalid scalar shape description")
    batch, size_i = a_shape
    b_batch, size_j = b_shape
    if b_batch != batch or out_shape != (batch, size_i, size_j):
        raise ValueError("Invalid outer-product shape description")
    elements = batch * size_i * size_j
    return {
        "flops": elements,
        "memory_bytes": 2 * (batch * size_i + batch * size_j + elements),
        "dtype": "bf16",
    }


DESIGNED_VRAM_SIZE = 48
