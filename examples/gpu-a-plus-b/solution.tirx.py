from __future__ import annotations

import tvm
from tvm.script import tirx as T

THREADS = 512
VECTOR_SIZE = 8
compiled_kernel = None

ADD8_SOURCE = r"""
__device__ __forceinline__ void hydro_add8(half* a, const half* b) {
    uint4 av = *reinterpret_cast<const uint4*>(a);
    const uint4 bv = *reinterpret_cast<const uint4*>(b);
    half2* av2 = reinterpret_cast<half2*>(&av);
    const half2* bv2 = reinterpret_cast<const half2*>(&bv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        av2[i] = __hadd2(av2[i], bv2[i]);
    }
    *reinterpret_cast<uint4*>(a) = av;
}
"""


def build_kernel(numel: int):
    block_size = THREADS * VECTOR_SIZE
    if numel % block_size:
        raise ValueError(f"GPU1 sizes must be divisible by {block_size}")
    blocks = numel // block_size

    @T.prim_func
    def add(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (numel,), "float16")
        B = T.match_buffer(B_ptr, (numel,), "float16")

        T.device_entry()
        bx = T.cta_id([blocks])
        tx = T.thread_id([THREADS])
        base = (bx * THREADS + tx) * VECTOR_SIZE

        T.cuda.func_call(
            "hydro_add8",
            A.ptr_to([base]),
            B.ptr_to([base]),
            source_code=ADD8_SOURCE,
        )

    return tvm.compile(
        tvm.IRModule({"main": add}),
        target=tvm.target.Target("cuda"),
        tir_pipeline="tirx",
    )


def run_kernel(A, B, numel):
    global compiled_kernel
    if compiled_kernel is None:
        compiled_kernel = build_kernel(int(numel))
    compiled_kernel(A.view(-1), B.view(-1))
