from __future__ import annotations

import tilelang.language as T
from tilelang import jit
from tilelang.carver.arch import driver


BLOCK_M = 128
BLOCK_K = 64
NUM_STAGES = 6

real_kernel = None
compiled_shape = None


@jit
def matmul_nt_persistent(m: int, n: int, k: int):
    block_n = 128

    @T.prim_func
    def kernel(
        a: T.Tensor((m, k), "bfloat16"),
        b: T.Tensor((n, k), "bfloat16"),
        c: T.Tensor((m, n), "bfloat16"),
    ):
        sm_num = driver.get_num_sms()
        m_blocks = T.ceildiv(m, BLOCK_M)
        n_blocks = T.ceildiv(n, block_n)
        k_blocks = T.ceildiv(k, BLOCK_K)
        with T.Kernel(sm_num, threads=256) as block_id:
            a_shared = T.alloc_shared((NUM_STAGES, BLOCK_M, BLOCK_K), "bfloat16")
            b_shared = T.alloc_shared((NUM_STAGES, block_n, BLOCK_K), "bfloat16")
            c_tmem_0 = T.alloc_tmem((BLOCK_M, block_n), "float")
            c_tmem_1 = T.alloc_tmem((BLOCK_M, block_n), "float")
            c_local = T.alloc_fragment((BLOCK_M, block_n), "float")
            c_shared = T.alloc_shared((BLOCK_M, 64), "bfloat16")
            loaded = T.alloc_barrier([32] * NUM_STAGES)
            consumed = T.alloc_barrier([1] * NUM_STAGES)
            tmem_full = T.alloc_barrier([1] * 2)
            tmem_empty = T.alloc_barrier([128] * 2)
            tx = T.get_thread_binding()

            if tx < 32:
                scheduler = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=8)
                scheduler.init(block_id)
                while scheduler.valid():
                    bx, by = scheduler.m_idx, scheduler.n_idx
                    for ko in T.serial(k_blocks):
                        phase = scheduler.current_iter * k_blocks + ko
                        T.mbarrier_wait_parity(
                            consumed[phase % NUM_STAGES],
                            ((phase // NUM_STAGES) & 1) ^ 1,
                        )
                        T.tma_copy(
                            a[
                                bx * BLOCK_M : (bx + 1) * BLOCK_M,
                                ko * BLOCK_K : (ko + 1) * BLOCK_K,
                            ],
                            a_shared[phase % NUM_STAGES, :, :],
                            barrier=loaded[phase % NUM_STAGES],
                        )
                        T.tma_copy(
                            b[
                                by * block_n : (by + 1) * block_n,
                                ko * BLOCK_K : (ko + 1) * BLOCK_K,
                            ],
                            b_shared[phase % NUM_STAGES, :, :],
                            barrier=loaded[phase % NUM_STAGES],
                        )
                        T.mbarrier_arrive(loaded[phase % NUM_STAGES])
                    scheduler.next_tile()
            elif tx < 64:
                scheduler = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=8)
                scheduler.init(block_id)
                while scheduler.valid():
                    slot = scheduler.current_iter & 1
                    T.mbarrier_wait_parity(
                        tmem_empty[slot],
                        ((scheduler.current_iter // 2) & 1) ^ 1,
                    )
                    for ko in T.serial(k_blocks):
                        phase = scheduler.current_iter * k_blocks + ko
                        T.mbarrier_wait_parity(
                            loaded[phase % NUM_STAGES],
                            (phase // NUM_STAGES) & 1,
                        )
                        if slot == 0:
                            T.tcgen05_gemm(
                                a_shared[phase % NUM_STAGES, :, :],
                                b_shared[phase % NUM_STAGES, :, :],
                                c_tmem_0,
                                transpose_B=True,
                                mbar=consumed[phase % NUM_STAGES],
                                clear_accum=ko == 0,
                            )
                        else:
                            T.tcgen05_gemm(
                                a_shared[phase % NUM_STAGES, :, :],
                                b_shared[phase % NUM_STAGES, :, :],
                                c_tmem_1,
                                transpose_B=True,
                                mbar=consumed[phase % NUM_STAGES],
                                clear_accum=ko == 0,
                            )
                    T.tcgen05_mma_arrive(tmem_full[slot])
                    scheduler.next_tile()
            elif 128 <= tx < 256:
                scheduler = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=8)
                scheduler.init(block_id)
                while scheduler.valid():
                    bx, by = scheduler.m_idx, scheduler.n_idx
                    slot = scheduler.current_iter & 1
                    T.mbarrier_wait_parity(
                        tmem_full[slot],
                        (scheduler.current_iter // 2) & 1,
                    )
                    if slot == 0:
                        T.copy(c_tmem_0, c_local)
                    else:
                        T.copy(c_tmem_1, c_local)
                    T.mbarrier_arrive(tmem_empty[slot])
                    for ni in T.unroll(T.ceildiv(block_n, 64)):
                        T.copy(c_local[:, ni * 64 : (ni + 1) * 64], c_shared)
                        T.copy(c_shared, c[bx * BLOCK_M, by * block_n + ni * 64])
                    scheduler.next_tile()

    return kernel


@jit
def matmul_nt_persistent_2cta(m: int, n: int, k: int):
    block_n = 256

    @T.prim_func
    def kernel(
        a: T.Tensor((m, k), "bfloat16"),
        b: T.Tensor((n, k), "bfloat16"),
        c: T.Tensor((m, n), "bfloat16"),
    ):
        sm_num = driver.get_num_sms()
        m_blocks = T.ceildiv(m, BLOCK_M)
        n_blocks = T.ceildiv(n, block_n)
        k_blocks = T.ceildiv(k, BLOCK_K)
        with T.ClusterKernel(sm_num, threads=256, cluster_dims=2) as block_id:
            a_shared = T.alloc_shared((NUM_STAGES, BLOCK_M, BLOCK_K), "bfloat16")
            b_shared = T.alloc_shared((NUM_STAGES, block_n // 2, BLOCK_K), "bfloat16")
            c_tmem_0 = T.alloc_tmem((BLOCK_M, block_n), "float")
            c_tmem_1 = T.alloc_tmem((BLOCK_M, block_n), "float")
            c_local = T.alloc_fragment((BLOCK_M, block_n), "float")
            c_shared = T.alloc_shared((BLOCK_M, 64), "bfloat16")
            loaded = T.alloc_cluster_barrier([64] * NUM_STAGES)
            consumed = T.alloc_cluster_barrier([1] * NUM_STAGES)
            tmem_full = T.alloc_cluster_barrier([1] * 2)
            tmem_empty = T.alloc_cluster_barrier([256] * 2)
            tx = T.get_thread_binding()
            cta_id = T.block_rank_in_cluster()
            T.assume(cta_id < 2)

            if tx < 32:
                scheduler = T.PersistentTileScheduler(
                    m_blocks,
                    n_blocks,
                    swizzle_size=8,
                    cluster_size=2,
                )
                scheduler.init(block_id // 2)
                while scheduler.valid():
                    bx = scheduler.m_idx * 2 + cta_id
                    by = scheduler.n_idx
                    for ko in T.serial(k_blocks):
                        phase = scheduler.current_iter * k_blocks + ko
                        T.mbarrier_wait_parity(
                            consumed[phase % NUM_STAGES],
                            ((phase // NUM_STAGES) & 1) ^ 1,
                        )
                        T.tma_copy(
                            a[
                                bx * BLOCK_M : (bx + 1) * BLOCK_M,
                                ko * BLOCK_K : (ko + 1) * BLOCK_K,
                            ],
                            a_shared[phase % NUM_STAGES, :, :],
                            barrier=loaded[phase % NUM_STAGES],
                        )
                        T.tma_copy(
                            b[
                                (by * 2 + cta_id) * (block_n // 2) :
                                (by * 2 + cta_id + 1) * (block_n // 2),
                                ko * BLOCK_K : (ko + 1) * BLOCK_K,
                            ],
                            b_shared[phase % NUM_STAGES, :, :],
                            barrier=loaded[phase % NUM_STAGES],
                        )
                        T.mbarrier_arrive(loaded[phase % NUM_STAGES], 0)
                    scheduler.next_tile()
            elif cta_id == 0 and tx < 64:
                scheduler = T.PersistentTileScheduler(
                    m_blocks,
                    n_blocks,
                    swizzle_size=8,
                    cluster_size=2,
                )
                scheduler.init(block_id // 2)
                while scheduler.valid():
                    slot = scheduler.current_iter & 1
                    T.mbarrier_wait_parity(
                        tmem_empty[slot],
                        ((scheduler.current_iter // 2) & 1) ^ 1,
                    )
                    for ko in T.serial(k_blocks):
                        phase = scheduler.current_iter * k_blocks + ko
                        T.mbarrier_wait_parity(
                            loaded[phase % NUM_STAGES],
                            (phase // NUM_STAGES) & 1,
                        )
                        if slot == 0:
                            T.tcgen05_gemm(
                                a_shared[phase % NUM_STAGES, :, :],
                                b_shared[phase % NUM_STAGES, :, :],
                                c_tmem_0,
                                transpose_B=True,
                                mbar=consumed[phase % NUM_STAGES],
                                clear_accum=ko == 0,
                                use_2cta=True,
                            )
                        else:
                            T.tcgen05_gemm(
                                a_shared[phase % NUM_STAGES, :, :],
                                b_shared[phase % NUM_STAGES, :, :],
                                c_tmem_1,
                                transpose_B=True,
                                mbar=consumed[phase % NUM_STAGES],
                                clear_accum=ko == 0,
                                use_2cta=True,
                            )
                    T.tcgen05_mma_arrive(tmem_full[slot], arrive_2cta=True)
                    scheduler.next_tile()
            elif 128 <= tx < 256:
                scheduler = T.PersistentTileScheduler(
                    m_blocks,
                    n_blocks,
                    swizzle_size=8,
                    cluster_size=2,
                )
                scheduler.init(block_id // 2)
                while scheduler.valid():
                    bx = scheduler.m_idx * 2 + cta_id
                    by = scheduler.n_idx
                    slot = scheduler.current_iter & 1
                    T.mbarrier_wait_parity(
                        tmem_full[slot],
                        (scheduler.current_iter // 2) & 1,
                    )
                    if slot == 0:
                        T.copy(c_tmem_0, c_local)
                    else:
                        T.copy(c_tmem_1, c_local)
                    T.mbarrier_arrive(tmem_empty[slot], 0)
                    for ni in T.unroll(T.ceildiv(block_n, 64)):
                        T.copy(c_local[:, ni * 64 : (ni + 1) * 64], c_shared)
                        T.copy(c_shared, c[bx * BLOCK_M, by * block_n + ni * 64])
                    scheduler.next_tile()

    return kernel


def run_kernel(a, b, c, m, n, k):
    global compiled_shape, real_kernel
    shape = (int(m), int(n), int(k))
    if real_kernel is None:
        compiled_shape = shape
        builder = matmul_nt_persistent if shape[1] == 128 else matmul_nt_persistent_2cta
        real_kernel = builder(*shape)
    elif shape != compiled_shape:
        raise ValueError(f"The cached kernel is specialized for {compiled_shape}, received {shape}")
    real_kernel(a, b, c)
