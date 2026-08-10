from __future__ import annotations

from typing import List, Tuple, Union

import torch

KernelArg = Union[torch.Tensor, int, float]

# XPUOJ problem 100 shapes, with enough interleaved samples for stable B200 timing.
TESTCASES = [
    (8192, 16384, 3, 15),
    (4096, 32768, 3, 15),
    (6145, 24577, 3, 15),
    (16383, 8193, 3, 15),
    (2049, 65537, 3, 15),
    (32769, 4097, 3, 15),
    (12289, 12287, 3, 15),
    (3073, 49153, 3, 15),
    (513, 262145, 3, 15),
    (24577, 6145, 3, 15),
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    testcase_id = int(input())
    if testcase_id < 1 or testcase_id > len(TESTCASES):
        raise ValueError(f"Invalid testcase id: {testcase_id}")
    height, width, warmup, repeats = TESTCASES[testcase_id - 1]
    return [(height, width), (width, height), (), ()], (warmup, repeats)


def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
    input_shape, output_shape, height_shape, width_shape = testcase_sizes
    if height_shape != () or width_shape != ():
        raise ValueError("H and W must be scalar arguments")
    height, width = input_shape
    if output_shape != (width, height):
        raise ValueError("Output shape must be the transpose of the input shape")
    input_tensor = torch.randn(input_shape, dtype=torch.float32, device=device)
    output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)
    return [input_tensor, output_tensor, int(height), int(width)]


INPUT_CLASS = ["INPUT", "OUTPUT", "INPUT", "INPUT"]


def baseline(input_tensor, output_tensor, height, width):
    if input_tensor.shape != (height, width) or output_tensor.shape != (width, height):
        raise ValueError("Tensor shapes do not match H and W")
    output_tensor.copy_(torch.transpose(input_tensor, 0, 1).contiguous())


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
) -> bool:
    del testcase_sizes
    original_input = original_input_tensors[0]
    target_input, target_output, target_height, target_width = target_kernel_input_tensors
    _, baseline_output, baseline_height, baseline_width = baseline_input_tensors
    if target_height != baseline_height or target_width != baseline_width:
        print("H or W was not preserved")
        return False
    if not torch.equal(target_input, original_input):
        print("The read-only input tensor was modified")
        return False
    if target_output.shape != baseline_output.shape or target_output.dtype != baseline_output.dtype:
        print(
            f"Output shape/dtype mismatch: {target_output.shape}/{target_output.dtype} "
            f"vs {baseline_output.shape}/{baseline_output.dtype}"
        )
        return False
    if not torch.equal(target_output, baseline_output):
        difference = (target_output - baseline_output).abs()
        print(f"Output differs from the PyTorch baseline: max_abs_diff={float(difference.max().item()):.6g}")
        return False
    return True


def getWorkload(testcase_sizes: List[Tuple[int, ...]]) -> dict:
    input_shape, output_shape, height_shape, width_shape = testcase_sizes
    if height_shape != () or width_shape != ():
        raise ValueError("Invalid scalar shape description")
    height, width = input_shape
    if output_shape != (width, height):
        raise ValueError("Invalid output shape description")
    numel = height * width
    return {
        "flops": numel,
        "memory_bytes": numel * 4 * 2,
        "dtype": "fp32",
    }


DESIGNED_VRAM_SIZE = 48
