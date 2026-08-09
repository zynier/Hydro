from __future__ import annotations

import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def add_kernel(a_ptr, b_ptr, numel, BLOCK: tl.constexpr):
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    a_value = tl.load(a_ptr + offsets, mask=mask)
    b_value = tl.load(b_ptr + offsets, mask=mask)
    tl.store(a_ptr + offsets, a_value + b_value, mask=mask)


def run_kernel(A, B, numel):
    add_kernel[(triton.cdiv(numel, BLOCK_SIZE),)](
        A,
        B,
        numel,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
    )
