from __future__ import annotations

from typing import List, Tuple, Union

import torch

KernelArg = Union[torch.Tensor, int, float]

# XPUOJ problem 3 test cases: (batch size, class count, warmup, measured runs).
TESTCASES = [
    (1536, 11776, 7, 146),
    (7936, 4096, 16, 316),
    (256, 19712, 16, 305),
    (2316, 7712, 12, 242),
    (33024, 1280, 11, 222),
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    testcase_id = int(input())
    if testcase_id < 1 or testcase_id > len(TESTCASES):
        raise ValueError(f"Invalid testcase id: {testcase_id}")
    batch_size, num_classes, warmup, repeats = TESTCASES[testcase_id - 1]
    return [
        (batch_size, num_classes),
        (batch_size,),
        (batch_size, num_classes),
        (),
        (),
    ], (warmup, repeats)


def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
    probs_shape, top_p_shape, output_shape, batch_shape, classes_shape = testcase_sizes
    if batch_shape != () or classes_shape != ():
        raise ValueError("batch_size and num_classes must be scalar arguments")
    batch_size, num_classes = probs_shape
    if top_p_shape != (batch_size,) or output_shape != (batch_size, num_classes):
        raise ValueError("Invalid tensor shapes")
    probs = torch.rand(probs_shape, device=device, dtype=torch.float32)
    probs.div_(probs.sum(dim=1, keepdim=True))
    top_p = torch.rand((batch_size,), device=device, dtype=torch.float32) * 0.5 + 0.3
    renorm_probs = torch.empty(output_shape, device=device, dtype=torch.float32)
    return [probs, top_p, renorm_probs, int(batch_size), int(num_classes)]


INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT"]


def baseline(probs, top_p, renorm_probs, batch_size, num_classes):
    if probs.shape != (batch_size, num_classes) or top_p.shape != (batch_size,):
        raise ValueError("Input shapes do not match scalar arguments")
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum <= top_p.reshape(batch_size, 1)
    mask[:, 0] = True
    filtered = sorted_probs * mask.to(sorted_probs.dtype)
    denom = filtered.sum(dim=-1, keepdim=True)
    renorm_sorted = torch.where(denom > 0, filtered / denom, torch.zeros_like(filtered))
    renorm_probs.zero_()
    renorm_probs.scatter_(dim=-1, index=sorted_idx, src=renorm_sorted)


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    del testcase_sizes, original_input_tensors
    probs_t, top_p_t, output_t, batch_t, classes_t = target_kernel_input_tensors
    probs_ref, top_p_ref, output_ref, batch_ref, classes_ref = baseline_input_tensors
    if batch_t != batch_ref or classes_t != classes_ref:
        return False
    if not torch.equal(probs_t, probs_ref) or not torch.equal(top_p_t, top_p_ref):
        print("The read-only input tensor was modified")
        return False
    if output_t.shape != output_ref.shape or output_t.dtype != output_ref.dtype:
        print(f"Output shape/dtype mismatch: {output_t.shape}/{output_t.dtype}")
        return False
    if not torch.isfinite(output_t).all():
        print("Output contains a non-finite value")
        return False
    row_sums = output_t.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-3, atol=1e-3):
        print(f"Output rows are not normalized: max_error={float((row_sums - 1).abs().max().item()):.6g}")
        return False
    support_mismatch = torch.logical_xor(output_t != 0, output_ref != 0).sum(dim=1)
    if int(support_mismatch.max().item()) > 1:
        print(f"Top-p support differs from baseline: max_mismatches={int(support_mismatch.max().item())}")
        return False
    if not torch.allclose(output_t, output_ref, rtol=rtol, atol=atol):
        diff = (output_t - output_ref).abs()
        print(f"Output differs from baseline: max_abs_diff={float(diff.max().item()):.6g}")
        return False
    return True


def getWorkload(testcase_sizes: List[Tuple[int, ...]]) -> dict:
    probs_shape, top_p_shape, output_shape, _, _ = testcase_sizes
    batch_size, num_classes = probs_shape
    if top_p_shape != (batch_size,) or output_shape != (batch_size, num_classes):
        raise ValueError("Invalid workload shapes")
    numel = batch_size * num_classes
    return {
        # Sorting, prefix sum, masking, and normalization are modeled as a
        # constant amount of work per probability, matching XPUOJ problem 3.
        "flops": 20 * numel,
        "memory_bytes": numel * 4 * 2 + batch_size * 4,
        "dtype": "fp32",
    }


DESIGNED_VRAM_SIZE = 48
