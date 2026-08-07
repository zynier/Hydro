from __future__ import annotations

from typing import List, Tuple, Union

import torch

KernelArg = Union[torch.Tensor, int, float]

# (shape, warmup, measured runs)
TESTCASES = [
    ((8192, 8192), 3, 30),
    ((12288, 8192), 3, 30),
    ((16384, 8192), 3, 30),
    ((8192, 12288), 3, 30),
    ((12288, 12288), 3, 30),
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    testcase_id = int(input())
    if testcase_id < 1 or testcase_id > len(TESTCASES):
        raise ValueError(f"Invalid testcase id: {testcase_id}")
    shape, warmup, repeats = TESTCASES[testcase_id - 1]
    return [shape, shape, ()], (warmup, repeats)


def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
    a_shape, b_shape, n_shape = testcase_sizes
    if a_shape != b_shape or n_shape != ():
        raise ValueError("Expected equally shaped A and B tensors followed by a scalar")
    a = torch.randn(*a_shape, dtype=torch.float16, device=device)
    b = torch.randn(*b_shape, dtype=torch.float16, device=device)
    return [a, b, int(a.numel())]


INPUT_CLASS = ["INOUT", "INPUT", "INPUT"]


def baseline(a: torch.Tensor, b: torch.Tensor, numel: int):
    if numel != a.numel():
        raise ValueError("numel does not match A")
    a.add_(b)
    return [a, b, numel]


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    del testcase_sizes
    original_b = original_input_tensors[1]
    target_a, target_b = target_kernel_input_tensors[:2]
    baseline_a = baseline_input_tensors[0]
    if target_a.shape != baseline_a.shape or target_a.dtype != baseline_a.dtype:
        print(f"A shape/dtype mismatch: {target_a.shape}/{target_a.dtype} vs {baseline_a.shape}/{baseline_a.dtype}")
        return False
    if not torch.equal(target_b, original_b):
        print("The read-only B tensor was modified")
        return False
    if not torch.allclose(target_a, baseline_a, rtol=rtol, atol=atol):
        difference = (target_a - baseline_a).abs()
        print(
            f"A differs from the PyTorch baseline: max_abs_diff={float(difference.max().item()):.6g} "
            f"(rtol={rtol}, atol={atol})"
        )
        return False
    return True


def getWorkload(testcase_sizes) -> dict:
    a_shape, b_shape, n_shape = testcase_sizes
    if a_shape != b_shape or n_shape != ():
        raise ValueError("Invalid testcase shape description")
    numel = 1
    for dimension in a_shape:
        numel *= dimension
    return {
        "flops": numel,
        "memory_bytes": numel * 2 * 3,
        "dtype": "fp16",
    }


DESIGNED_VRAM_SIZE = 48
