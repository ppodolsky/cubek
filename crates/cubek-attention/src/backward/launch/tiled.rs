//! Tensor-core FlashAttention backward.
//!
//! Replaces the scalar scaffold kernels for the shapes training uses:
//! `head_dim == val_dim == 64`, sequence a multiple of 64, square
//! self-attention. Two kernels — dQ (query-outer) and dK/dV (key-outer) —
//! recompute `S` and `dP` tiles with bf16 cmma against fp32 accumulators
//! (the same precision profile as the GEMM-chain backward they replace) and
//! never materialize a `[seq, seq]` tensor in global memory. The `D`
//! prepass (`rowsum(dO ⊙ O)`) is a small generic elementwise reduction.
//!
//! Layout conventions inside a block:
//! - 128 threads as 4 planes; every cmma op is plane-scoped.
//! - Operands stage through bf16 shared memory; a plane's `S`/`dP`
//!   accumulator bounces through an fp32 scratch tile so the softmax
//!   recompute (`P = exp(s·scale − lse)`, causal mask, `dS`) runs
//!   element-wise between the two cmma stages.
//! - Loading row-major shared memory as a `ColMajor` B operand yields the
//!   transposed operand without any explicit transpose.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use half::bf16;

use crate::forward::definition::AttentionSetupError;

/// Rows in a cmma tile.
const TILE: usize = 16;
/// Rows a block owns (4 plane tiles).
const BLOCK: usize = 64;
/// Head dimension the kernels are specialized for.
const HEAD_DIM: usize = 64;
const PLANE_DIM: u32 = 32;
const NUM_PLANES: u32 = 4;
const HD_TILES: usize = HEAD_DIM / TILE;
const BLOCK_TILES: usize = BLOCK / TILE;
const TILE_ELEMS: usize = TILE * TILE;
const BLOCK_ELEMS: usize = BLOCK * HEAD_DIM;

#[cube(launch)]
fn flash_backward_prepass_kernel<OF: Float, DF: Float>(
    o: &Tensor<OF>,
    do_: &Tensor<DF>,
    d: &mut Tensor<f32>,
    total_rows: u32,
    #[comptime] head_dim: usize,
) {
    let row = ABSOLUTE_POS;
    if row < total_rows as usize {
        let base = row * head_dim;
        let mut sum = 0.0f32;
        for i in 0..head_dim {
            sum += f32::cast_from(o[base + i]) * f32::cast_from(do_[base + i]);
        }
        d[row] = sum;
    }
}

/// dK/dV: one block per `(batch·head, kv block)`; each plane owns a 16-row
/// slice of the block's keys/values and sweeps the visible query tiles.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn flash_backward_dkdv_kernel<QF: Float, DF: Float>(
    q: &Tensor<QF>,
    k: &Tensor<QF>,
    v: &Tensor<QF>,
    do_: &Tensor<DF>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dk: &mut Tensor<f32>,
    dv: &mut Tensor<f32>,
    seq: u32,
    scale: f32,
    #[comptime] causal: bool,
) {
    let seq = seq as usize;
    let bh = CUBE_POS_X as usize;
    let kv_block = CUBE_POS_Y as usize;
    let plane = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = plane * PLANE_DIM as usize + lane;
    let threads = (PLANE_DIM * NUM_PLANES) as usize;

    let base = bh * seq * HEAD_DIM;
    let row_base = bh * seq;
    let kv0 = kv_block * BLOCK;

    let mut k_smem = Shared::<[bf16]>::new_slice(BLOCK_ELEMS);
    let mut v_smem = Shared::<[bf16]>::new_slice(BLOCK_ELEMS);
    let mut q_smem = Shared::<[bf16]>::new_slice(BLOCK_ELEMS);
    let mut do_smem = Shared::<[bf16]>::new_slice(BLOCK_ELEMS);
    let mut lse_smem = Shared::<[f32]>::new_slice(BLOCK);
    let mut d_smem = Shared::<[f32]>::new_slice(BLOCK);
    let mut s_scratch = Shared::<[f32]>::new_slice(NUM_PLANES as usize * TILE_ELEMS);
    let mut dp_scratch = Shared::<[f32]>::new_slice(NUM_PLANES as usize * TILE_ELEMS);
    let mut p_smem = Shared::<[bf16]>::new_slice(NUM_PLANES as usize * TILE_ELEMS);
    let mut ds_smem = Shared::<[bf16]>::new_slice(NUM_PLANES as usize * TILE_ELEMS);
    let scratch_base = plane * TILE_ELEMS;

    let mut index = tid;
    while index < BLOCK_ELEMS {
        let row = index / HEAD_DIM;
        let col = index % HEAD_DIM;
        let global = base + (kv0 + row) * HEAD_DIM + col;
        k_smem[index] = bf16::cast_from(k[global]);
        v_smem[index] = bf16::cast_from(v[global]);
        index += threads;
    }

    let mut dk_acc = Sequence::<cmma::Matrix<f32>>::new();
    let mut dv_acc = Sequence::<cmma::Matrix<f32>>::new();
    #[unroll]
    for _ in 0..HD_TILES {
        dk_acc.push(cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            TILE,
            TILE,
            TILE,
            cmma::MatrixLayout::Undefined,
            0.0,
        ));
        dv_acc.push(cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            TILE,
            TILE,
            TILE,
            cmma::MatrixLayout::Undefined,
            0.0,
        ));
    }

    let q_blocks = seq / BLOCK;
    let mut q_start = 0usize;
    if causal {
        q_start = kv_block;
    }

    for q_block in q_start..q_blocks {
        let q0 = q_block * BLOCK;
        sync_cube();
        let mut index = tid;
        while index < BLOCK_ELEMS {
            let row = index / HEAD_DIM;
            let col = index % HEAD_DIM;
            let global = base + (q0 + row) * HEAD_DIM + col;
            q_smem[index] = bf16::cast_from(q[global]);
            do_smem[index] = bf16::cast_from(do_[global]);
            index += threads;
        }
        let mut index = tid;
        while index < BLOCK {
            lse_smem[index] = lse[row_base + q0 + index];
            d_smem[index] = d[row_base + q0 + index];
            index += threads;
        }
        sync_cube();

        for q_tile in 0..BLOCK_TILES {
            // dP^T[kv, q] = V · dO^T
            let dp = cmma::Matrix::<f32>::from_value(
                cmma::MatrixIdent::Accumulator,
                TILE,
                TILE,
                TILE,
                cmma::MatrixLayout::Undefined,
                0.0,
            );
            #[unroll]
            for h in 0..HD_TILES {
                let a = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::A,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &v_smem[plane * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                let b = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::B,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::ColMajor,
                    &do_smem[q_tile * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                cmma::execute(&a, &b, &dp, &dp);
            }
            cmma::store(
                &mut dp_scratch[scratch_base..scratch_base + TILE_ELEMS],
                &dp,
                TILE as u32,
                cmma::MatrixLayout::RowMajor,
            );

            // S^T[kv, q] = K · Q^T
            let st = cmma::Matrix::<f32>::from_value(
                cmma::MatrixIdent::Accumulator,
                TILE,
                TILE,
                TILE,
                cmma::MatrixLayout::Undefined,
                0.0,
            );
            #[unroll]
            for h in 0..HD_TILES {
                let a = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::A,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &k_smem[plane * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                let b = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::B,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::ColMajor,
                    &q_smem[q_tile * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                cmma::execute(&a, &b, &st, &st);
            }
            cmma::store(
                &mut s_scratch[scratch_base..scratch_base + TILE_ELEMS],
                &st,
                TILE as u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_plane();

            // P^T = exp(s·scale − lse), dS^T = P^T ⊙ (dP^T − D) · scale,
            // with the causal mask applied per element.
            let mut i = lane;
            while i < TILE_ELEMS {
                let kv_row = i / TILE;
                let q_col = i % TILE;
                let global_q = q0 + q_tile * TILE + q_col;
                let global_k = kv0 + plane * TILE + kv_row;
                let mut visible = true;
                if causal {
                    visible = global_q >= global_k;
                }
                let mut p = 0.0f32;
                if visible {
                    p = (s_scratch[scratch_base + i] * scale - lse_smem[q_tile * TILE + q_col])
                        .exp();
                }
                let ds = p * (dp_scratch[scratch_base + i] - d_smem[q_tile * TILE + q_col]) * scale;
                p_smem[scratch_base + i] = bf16::cast_from(p);
                ds_smem[scratch_base + i] = bf16::cast_from(ds);
                i += PLANE_DIM as usize;
            }
            sync_plane();

            // dV += P^T · dO, dK += dS^T · Q
            #[unroll]
            for h in 0..HD_TILES {
                let a_p = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::A,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &p_smem[scratch_base..scratch_base + TILE_ELEMS],
                    TILE as u32,
                );
                let b_do = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::B,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &do_smem[q_tile * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                cmma::execute(&a_p, &b_do, &dv_acc[h], &dv_acc[h]);

                let a_ds = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::A,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &ds_smem[scratch_base..scratch_base + TILE_ELEMS],
                    TILE as u32,
                );
                let b_q = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::B,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &q_smem[q_tile * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                cmma::execute(&a_ds, &b_q, &dk_acc[h], &dk_acc[h]);
            }
        }
    }

    let out_row = base + (kv0 + plane * TILE) * HEAD_DIM;
    #[unroll]
    for h in 0..HD_TILES {
        cmma::store(
            &mut dk[out_row + h * TILE..base + seq * HEAD_DIM],
            &dk_acc[h],
            HEAD_DIM as u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            &mut dv[out_row + h * TILE..base + seq * HEAD_DIM],
            &dv_acc[h],
            HEAD_DIM as u32,
            cmma::MatrixLayout::RowMajor,
        );
    }
}

/// dQ: one block per `(batch·head, q block)`; each plane owns a 16-row
/// slice of the block's queries and sweeps the visible key tiles.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn flash_backward_dq_kernel<QF: Float, DF: Float>(
    q: &Tensor<QF>,
    k: &Tensor<QF>,
    v: &Tensor<QF>,
    do_: &Tensor<DF>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dq: &mut Tensor<f32>,
    seq: u32,
    scale: f32,
    #[comptime] causal: bool,
) {
    let seq = seq as usize;
    let bh = CUBE_POS_X as usize;
    let q_block = CUBE_POS_Y as usize;
    let plane = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = plane * PLANE_DIM as usize + lane;
    let threads = (PLANE_DIM * NUM_PLANES) as usize;

    let base = bh * seq * HEAD_DIM;
    let row_base = bh * seq;
    let q0 = q_block * BLOCK;

    let mut q_smem = Shared::<[bf16]>::new_slice(BLOCK_ELEMS);
    let mut do_smem = Shared::<[bf16]>::new_slice(BLOCK_ELEMS);
    let mut k_smem = Shared::<[bf16]>::new_slice(BLOCK_ELEMS);
    let mut v_smem = Shared::<[bf16]>::new_slice(BLOCK_ELEMS);
    let mut lse_smem = Shared::<[f32]>::new_slice(BLOCK);
    let mut d_smem = Shared::<[f32]>::new_slice(BLOCK);
    let mut s_scratch = Shared::<[f32]>::new_slice(NUM_PLANES as usize * TILE_ELEMS);
    let mut dp_scratch = Shared::<[f32]>::new_slice(NUM_PLANES as usize * TILE_ELEMS);
    let mut ds_smem = Shared::<[bf16]>::new_slice(NUM_PLANES as usize * TILE_ELEMS);
    let scratch_base = plane * TILE_ELEMS;

    let mut index = tid;
    while index < BLOCK_ELEMS {
        let row = index / HEAD_DIM;
        let col = index % HEAD_DIM;
        let global = base + (q0 + row) * HEAD_DIM + col;
        q_smem[index] = bf16::cast_from(q[global]);
        do_smem[index] = bf16::cast_from(do_[global]);
        index += threads;
    }
    let mut index = tid;
    while index < BLOCK {
        lse_smem[index] = lse[row_base + q0 + index];
        d_smem[index] = d[row_base + q0 + index];
        index += threads;
    }

    let mut dq_acc = Sequence::<cmma::Matrix<f32>>::new();
    #[unroll]
    for _ in 0..HD_TILES {
        dq_acc.push(cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            TILE,
            TILE,
            TILE,
            cmma::MatrixLayout::Undefined,
            0.0,
        ));
    }

    let kv_blocks = seq / BLOCK;
    let kv_end = if causal { q_block + 1 } else { kv_blocks };

    for kv_block in 0..kv_end {
        let kv0 = kv_block * BLOCK;
        sync_cube();
        let mut index = tid;
        while index < BLOCK_ELEMS {
            let row = index / HEAD_DIM;
            let col = index % HEAD_DIM;
            let global = base + (kv0 + row) * HEAD_DIM + col;
            k_smem[index] = bf16::cast_from(k[global]);
            v_smem[index] = bf16::cast_from(v[global]);
            index += threads;
        }
        sync_cube();

        for kv_tile in 0..BLOCK_TILES {
            // dP[q, kv] = dO · V^T
            let dp = cmma::Matrix::<f32>::from_value(
                cmma::MatrixIdent::Accumulator,
                TILE,
                TILE,
                TILE,
                cmma::MatrixLayout::Undefined,
                0.0,
            );
            #[unroll]
            for h in 0..HD_TILES {
                let a = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::A,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &do_smem[plane * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                let b = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::B,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::ColMajor,
                    &v_smem[kv_tile * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                cmma::execute(&a, &b, &dp, &dp);
            }
            cmma::store(
                &mut dp_scratch[scratch_base..scratch_base + TILE_ELEMS],
                &dp,
                TILE as u32,
                cmma::MatrixLayout::RowMajor,
            );

            // S[q, kv] = Q · K^T
            let st = cmma::Matrix::<f32>::from_value(
                cmma::MatrixIdent::Accumulator,
                TILE,
                TILE,
                TILE,
                cmma::MatrixLayout::Undefined,
                0.0,
            );
            #[unroll]
            for h in 0..HD_TILES {
                let a = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::A,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &q_smem[plane * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                let b = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::B,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::ColMajor,
                    &k_smem[kv_tile * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                cmma::execute(&a, &b, &st, &st);
            }
            cmma::store(
                &mut s_scratch[scratch_base..scratch_base + TILE_ELEMS],
                &st,
                TILE as u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_plane();

            // dS = P ⊙ (dP − D) · scale with P = exp(s·scale − lse).
            let mut i = lane;
            while i < TILE_ELEMS {
                let q_row = i / TILE;
                let kv_col = i % TILE;
                let global_q = q0 + plane * TILE + q_row;
                let global_k = kv0 + kv_tile * TILE + kv_col;
                let mut visible = true;
                if causal {
                    visible = global_q >= global_k;
                }
                let mut p = 0.0f32;
                if visible {
                    p = (s_scratch[scratch_base + i] * scale - lse_smem[plane * TILE + q_row])
                        .exp();
                }
                let ds = p * (dp_scratch[scratch_base + i] - d_smem[plane * TILE + q_row]) * scale;
                ds_smem[scratch_base + i] = bf16::cast_from(ds);
                i += PLANE_DIM as usize;
            }
            sync_plane();

            // dQ += dS · K
            #[unroll]
            for h in 0..HD_TILES {
                let a_ds = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::A,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &ds_smem[scratch_base..scratch_base + TILE_ELEMS],
                    TILE as u32,
                );
                let b_k = cmma::Matrix::<bf16>::from_slice(
                    cmma::MatrixIdent::B,
                    TILE,
                    TILE,
                    TILE,
                    cmma::MatrixLayout::RowMajor,
                    &k_smem[kv_tile * TILE * HEAD_DIM + h * TILE..BLOCK_ELEMS],
                    HEAD_DIM as u32,
                );
                cmma::execute(&a_ds, &b_k, &dq_acc[h], &dq_acc[h]);
            }
        }
    }

    let out_row = base + (q0 + plane * TILE) * HEAD_DIM;
    #[unroll]
    for h in 0..HD_TILES {
        cmma::store(
            &mut dq[out_row + h * TILE..base + seq * HEAD_DIM],
            &dq_acc[h],
            HEAD_DIM as u32,
            cmma::MatrixLayout::RowMajor,
        );
    }
}

/// Launches the tensor-core backward: prepass → dQ → dK/dV.
///
/// Inputs are `[B, H, N, 64]` with `N % 64 == 0` and square self-attention;
/// `lse` is `[B·H, N]` FP32 from the forward. `dq`/`dk`/`dv` are FP32 and
/// written cleanly. Returns an error for unsupported shapes so callers can
/// fall back.
#[allow(clippy::too_many_arguments, clippy::result_large_err)]
pub fn flash_attention_backward_tiled<QF: Float + CubePrimitive, DF: Float + CubePrimitive, R: Runtime>(
    client: &ComputeClient<R>,
    q: TensorBinding<R>,
    k: TensorBinding<R>,
    v: TensorBinding<R>,
    o: TensorBinding<R>,
    do_: TensorBinding<R>,
    lse: TensorBinding<R>,
    d: TensorBinding<R>,
    dq: TensorBinding<R>,
    dk: TensorBinding<R>,
    dv: TensorBinding<R>,
    scale: f32,
    causal: bool,
) -> Result<(), AttentionSetupError> {
    let rank = q.shape.len();
    let batch_heads: usize = q.shape[..rank - 2].iter().product();
    let seq = q.shape[rank - 2];
    let head_dim = q.shape[rank - 1];

    if head_dim != HEAD_DIM {
        return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
            "tiled attention backward requires head_dim == {HEAD_DIM}, got {head_dim}"
        ))));
    }
    if seq % BLOCK != 0 || seq == 0 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
            "tiled attention backward requires seq % {BLOCK} == 0, got {seq}"
        ))));
    }
    if k.shape != q.shape || v.shape != q.shape {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "tiled attention backward requires square self-attention".to_string(),
        )));
    }

    let total_rows = (batch_heads * seq) as u32;
    let prepass_threads = 128u32;
    flash_backward_prepass_kernel::launch::<QF, DF, R>(
        client,
        CubeCount::Static(total_rows.div_ceil(prepass_threads), 1, 1),
        CubeDim::new_1d(prepass_threads),
        o.into_tensor_arg(),
        do_.clone().into_tensor_arg(),
        d.clone().into_tensor_arg(),
        total_rows,
        HEAD_DIM,
    );

    let blocks = (seq / BLOCK) as u32;
    let cube_dim = CubeDim::new_2d(PLANE_DIM, NUM_PLANES);

    flash_backward_dq_kernel::launch::<QF, DF, R>(
        client,
        CubeCount::Static(batch_heads as u32, blocks, 1),
        cube_dim,
        q.clone().into_tensor_arg(),
        k.clone().into_tensor_arg(),
        v.clone().into_tensor_arg(),
        do_.clone().into_tensor_arg(),
        lse.clone().into_tensor_arg(),
        d.clone().into_tensor_arg(),
        dq.into_tensor_arg(),
        seq as u32,
        scale,
        causal,
    );

    flash_backward_dkdv_kernel::launch::<QF, DF, R>(
        client,
        CubeCount::Static(batch_heads as u32, blocks, 1),
        cube_dim,
        q.into_tensor_arg(),
        k.into_tensor_arg(),
        v.into_tensor_arg(),
        do_.into_tensor_arg(),
        lse.into_tensor_arg(),
        d.into_tensor_arg(),
        dk.into_tensor_arg(),
        dv.into_tensor_arg(),
        seq as u32,
        scale,
        causal,
    );

    Ok(())
}
