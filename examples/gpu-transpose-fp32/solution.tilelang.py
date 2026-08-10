from __future__ import annotations

import tilelang.language as T
from tilelang import jit


TILE = 64

real_kernel = None
compiled_shape = None


@jit
def build_transpose(height: int, width: int):
    @T.prim_func
    def kernel(
        input_tensor: T.Tensor((height, width), "float32"),
        output_tensor: T.Tensor((width, height), "float32"),
    ):
        with T.Kernel(T.ceildiv(width, TILE), T.ceildiv(height, TILE), threads=256) as (bx, by):
            tile = T.alloc_shared((TILE, TILE + 1), "float32")

            for row, col in T.Parallel(TILE, TILE):
                input_row = by * TILE + row
                input_col = bx * TILE + col
                if input_row < height and input_col < width:
                    tile[row, col] = input_tensor[input_row, input_col]

            T.sync_threads()

            for col, row in T.Parallel(TILE, TILE):
                output_row = bx * TILE + col
                output_col = by * TILE + row
                if output_row < width and output_col < height:
                    output_tensor[output_row, output_col] = tile[row, col]

    return kernel


def run_kernel(input_tensor, output_tensor, height, width):
    global compiled_shape, real_kernel
    shape = (int(height), int(width))
    if real_kernel is None or compiled_shape != shape:
        real_kernel = build_transpose(*shape)
        compiled_shape = shape
    real_kernel(input_tensor, output_tensor)
