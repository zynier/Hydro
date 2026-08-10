from __future__ import annotations

from typing import List, Tuple, Union

import torch

KernelArg = Union[torch.Tensor, int, float]

# XPUOJ problem 4 test cases: (batch size, class count, warmup, measured runs).
TESTCASES = [
    (5120, 1024, 3, 21),
    (17152, 2048, 3, 24),
    (32768, 4096, 3, 39),
    (7144, 4906, 3, 20),
    (70400, 1024, 3, 51),
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
    probs_shape, top_k_shape, output_shape, batch_shape, classes_shape = testcase_sizes
    if batch_shape != () or classes_shape != ():
        raise ValueError("batch_size and num_classes must be scalar arguments")
    batch_size, num_classes = probs_shape
    if top_k_shape != (batch_size,) or output_shape != (batch_size, num_classes):
        raise ValueError("Invalid tensor shapes")

    seed = int((batch_size * 1_000_003 + num_classes) & 0x7FFFFFFF)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    probs = torch.rand(probs_shape, device=device, dtype=torch.float32)
    probs.div_(probs.sum(dim=1, keepdim=True))
    top_k = torch.randint(1, num_classes, (batch_size,), device=device, dtype=torch.int32)
    renorm_probs = torch.empty(output_shape, device=device, dtype=torch.float32)
    return [probs, top_k, renorm_probs, int(batch_size), int(num_classes)]


INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT"]


def baseline(probs, top_k, renorm_probs, batch_size, num_classes):
    if probs.shape != (batch_size, num_classes) or top_k.shape != (batch_size,):
        raise ValueError("Input shapes do not match scalar arguments")
    k = top_k.reshape(batch_size, 1).clamp(min=0, max=num_classes).to(dtype=torch.long)
    max_k = int(k.max().item())
    renorm_probs.zero_()
    if max_k == 0:
        return
    top_values, top_indices = torch.topk(
        probs, k=max_k, dim=-1, largest=True, sorted=True,
    )
    mask = torch.arange(max_k, device=probs.device).reshape(1, max_k) < k
    filtered = top_values * mask.to(top_values.dtype)
    denominator = filtered.sum(dim=-1, keepdim=True)
    normalized = torch.where(
        denominator > 0,
        filtered / denominator,
        torch.zeros_like(filtered),
    )
    renorm_probs.scatter_(dim=-1, index=top_indices, src=normalized)


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
    rtol: float = 2e-3,
    atol: float = 2e-6,
) -> bool:
    del testcase_sizes, original_input_tensors
    probs_t, top_k_t, output_t, batch_t, classes_t = target_kernel_input_tensors
    probs_ref, top_k_ref, _, batch_ref, classes_ref = baseline_input_tensors
    if batch_t != batch_ref or classes_t != classes_ref:
        print("Scalar arguments changed")
        return False
    if not torch.equal(probs_t, probs_ref) or not torch.equal(top_k_t, top_k_ref):
        print("A read-only input tensor was modified")
        return False
    if output_t.shape != probs_t.shape or output_t.dtype != probs_t.dtype:
        print(f"Output shape/dtype mismatch: {output_t.shape}/{output_t.dtype}")
        return False
    if not torch.isfinite(output_t).all():
        print("Output contains a non-finite value")
        return False

    support = torch.count_nonzero(output_t, dim=1)
    if not torch.equal(support, top_k_t.to(dtype=support.dtype)):
        max_error = int((support - top_k_t).abs().max().item())
        print(f"Top-k support size mismatch: max_error={max_error}")
        return False

    selected = output_t != 0
    selected_min = torch.where(selected, probs_t, torch.inf).amin(dim=1)
    unselected_max = torch.where(selected, -torch.inf, probs_t).amax(dim=1)
    if not bool(torch.all(selected_min >= unselected_max)):
        max_order_error = float((unselected_max - selected_min).max().item())
        print(f"Output does not select the largest probabilities: max_error={max_order_error:.6g}")
        return False

    selected_values = torch.where(selected, probs_t, 0.0)
    selected_sum = selected_values.sum(dim=1, keepdim=True)
    expected = selected_values / selected_sum
    row_sums = output_t.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=5e-4, atol=5e-4):
        max_error = float((row_sums - 1).abs().max().item())
        print(f"Output rows are not normalized: max_error={max_error:.6g}")
        return False
    if not torch.allclose(output_t, expected, rtol=rtol, atol=atol):
        max_difference = float((output_t - expected).abs().max().item())
        print(f"Selected probabilities are not correctly normalized: max_abs_diff={max_difference:.6g}")
        return False
    return True


def getWorkload(testcase_sizes: List[Tuple[int, ...]]) -> dict:
    probs_shape, top_k_shape, output_shape, _, _ = testcase_sizes
    batch_size, num_classes = probs_shape
    if top_k_shape != (batch_size,) or output_shape != probs_shape:
        raise ValueError("Invalid testcase shape description")
    numel = batch_size * num_classes
    return {
        "flops": 20 * numel,
        "memory_bytes": numel * 4 * 2 + batch_size * 4,
        "dtype": "fp32",
    }


DESIGNED_VRAM_SIZE = 48
