from __future__ import annotations

from typing import List, Tuple, Union

import torch

KernelArg = Union[torch.Tensor, int, float]

# (M, N, K, warmup, measured runs), matching XPUOJ problem 2.
TESTCASES = [
    (6400, 6400, 1792, 3, 30),
    (20480, 2048, 3072, 3, 30),
    (1792, 18176, 2816, 3, 30),
    (9248, 6176, 1248, 3, 30),
    (113408, 128, 5120, 3, 30),
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    testcase_id = int(input())
    if testcase_id < 1 or testcase_id > len(TESTCASES):
        raise ValueError(f"Invalid testcase id: {testcase_id}")
    m, n, k, warmup, repeats = TESTCASES[testcase_id - 1]
    return [(m, k), (n, k), (m, n), (), (), ()], (warmup, repeats)


def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
    a_shape, b_shape, c_shape, m_shape, n_shape, k_shape = testcase_sizes
    if any(shape != () for shape in (m_shape, n_shape, k_shape)):
        raise ValueError("M, N, and K must be scalar arguments")
    m, k = a_shape
    n, b_k = b_shape
    if b_k != k or c_shape != (m, n):
        raise ValueError("Expected A[M,K], B[N,K], and C[M,N]")
    a = torch.randn(*a_shape, dtype=torch.bfloat16, device=device)
    b = torch.randn(*b_shape, dtype=torch.bfloat16, device=device)
    c = torch.empty(*c_shape, dtype=torch.bfloat16, device=device)
    return [a, b, c, int(m), int(n), int(k)]


INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"]


def baseline(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, m: int, n: int, k: int):
    if a.shape != (m, k) or b.shape != (n, k) or c.shape != (m, n):
        raise ValueError("Tensor shapes do not match M, N, and K")
    c.copy_(torch.matmul(a, b.T))
    return [a, b, c, m, n, k]


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    del testcase_sizes
    original_a, original_b = original_input_tensors[:2]
    target_a, target_b, target_c = target_kernel_input_tensors[:3]
    baseline_c = baseline_input_tensors[2]
    if not torch.equal(target_a, original_a):
        print("The read-only A tensor was modified")
        return False
    if not torch.equal(target_b, original_b):
        print("The read-only B tensor was modified")
        return False
    if target_c.shape != baseline_c.shape or target_c.dtype != baseline_c.dtype:
        print(
            f"C shape/dtype mismatch: {target_c.shape}/{target_c.dtype} "
            f"vs {baseline_c.shape}/{baseline_c.dtype}"
        )
        return False
    if not torch.allclose(target_c, baseline_c, rtol=rtol, atol=atol):
        difference = (target_c.float() - baseline_c.float()).abs()
        print(
            f"C differs from the PyTorch baseline: max_abs_diff={float(difference.max().item()):.6g} "
            f"(rtol={rtol}, atol={atol})"
        )
        return False
    return True


def getWorkload(testcase_sizes: List[Tuple[int, ...]]) -> dict:
    a_shape, b_shape, c_shape, m_shape, n_shape, k_shape = testcase_sizes
    if any(shape != () for shape in (m_shape, n_shape, k_shape)):
        raise ValueError("Invalid scalar shape description")
    m, k = a_shape
    n, b_k = b_shape
    if b_k != k or c_shape != (m, n):
        raise ValueError("Invalid matrix shape description")
    return {
        "flops": 2 * m * n * k,
        "memory_bytes": 2 * (m * k + n * k + m * n),
        "dtype": "bf16",
    }


DESIGNED_VRAM_SIZE = 48
