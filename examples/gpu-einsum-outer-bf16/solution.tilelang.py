from __future__ import annotations

import tilelang.language as T
from tilelang import jit


TILE_I = 64
TILE_J = 128

real_kernel = None
compiled_shape = None


@jit
def build_outer(batch: int, size_i: int, size_j: int):
    @T.prim_func
    def kernel(
        a: T.Tensor((batch, size_i), "bfloat16"),
        b: T.Tensor((batch, size_j), "bfloat16"),
        out: T.Tensor((batch, size_i, size_j), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(size_j, TILE_J),
            T.ceildiv(size_i, TILE_I),
            batch,
            threads=256,
        ) as (bx, by, bz):
            a_shared = T.alloc_shared((TILE_I,), "bfloat16")
            b_shared = T.alloc_shared((TILE_J,), "bfloat16")

            for i in T.Parallel(TILE_I):
                row = by * TILE_I + i
                if row < size_i:
                    a_shared[i] = a[bz, row]
            for j in T.Parallel(TILE_J):
                col = bx * TILE_J + j
                if col < size_j:
                    b_shared[j] = b[bz, col]
            T.sync_threads()

            for i, j in T.Parallel(TILE_I, TILE_J):
                row = by * TILE_I + i
                col = bx * TILE_J + j
                if row < size_i and col < size_j:
                    out[bz, row, col] = a_shared[i] * b_shared[j]

    return kernel


def run_kernel(a, b, out, batch, size_i, size_j):
    global compiled_shape, real_kernel
    shape = (int(batch), int(size_i), int(size_j))
    if real_kernel is None or compiled_shape != shape:
        real_kernel = build_outer(*shape)
        compiled_shape = shape
    real_kernel(a, b, out)
