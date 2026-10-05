//! CUDA tensor storage backed by cuTile Rust.
//!
//! cudarc owns device allocations. cuTile borrows the CUDA context and stream
//! for kernels, while cuBLAS uses the same stream for matmul. Work is queued
//! until the caller synchronizes or downloads a tensor.

use std::borrow::Borrow;

use crate::{
    dtype::{DType, WithDType},
    error::{Error, Result},
    layout::Layout,
    storage::{BackendStorage, BinaryOp, CpuStorage, ReduceOp, UnaryOp},
};

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod imp {
    use super::*;
    use std::sync::{Arc, Mutex, OnceLock};

    use cuda_async::device_buffer::DeviceAllocation;
    use cuda_async::device_operation::DeviceOp;
    use cuda_core::{Device as TileDevice, Stream as TileStream};
    use cudarc::cublas::{result as cublas, sys as blas_sys};
    use cudarc::driver::{
        CudaContext, CudaSlice, CudaStream as CudarcStream, DevicePtr, DeviceRepr, ValidAsZeroBits,
    };
    use cutile::{
        tensor::{IntoPartition, Partition, Reshape, Tensor, ToHostVec},
        tile_kernel::TileKernel,
    };
    use half::{bf16, f16};

    /// Tile width for 1-D elementwise kernels. Partial edge tiles are
    /// supported, so one width covers every tensor length.
    /// Elementwise kernels run one thread per element with 1024-thread
    /// blocks, matching the pre-cuTile launch config. Narrower 128-thread
    /// blocks measured ~4x slower per byte (238 vs 841 GB/s on a 67M-elem
    /// BF16 mul), so keep this wide: every `launch_1d` partition and `B`
    /// generic below must stay in sync with it.
    const TILE: usize = 1024;
    /// Pointer gather/scatter kernels (`gather`, `index_select`,
    /// `scatter_add`, `index_add`) hardcode 128-wide tiles and stay narrow:
    /// they run on tiny index tensors where wide blocks gain nothing.
    const GATHER_TILE: usize = 128;

    /// Waits for a host readback. When profiling is active, CUDA events
    /// attribute its device time to the current operation. Event failures
    /// fall back to an untimed readback.
    trait SyncProfiled: DeviceOp {
        fn sync_profiled(
            self,
            rt: &Runtime,
        ) -> std::result::Result<<Self as DeviceOp>::Output, cuda_async::error::DeviceError>
        {
            let scope = if crate::profiler::is_active() {
                crate::profiler::current_scope_id()
            } else {
                None
            };
            let Some(scope) = scope else {
                return self.sync_on(&rt.stream);
            };
            let (Ok(start), Ok(end)) = (rt.device.new_event(), rt.device.new_event()) else {
                return self.sync_on(&rt.stream);
            };
            if start.record(&rt.stream).is_err() {
                return self.sync_on(&rt.stream);
            }
            let out = self.sync_on(&rt.stream)?;
            if end.record(&rt.stream).is_err() {
                return Ok(out);
            }
            let device_ns = end
                .synchronize()
                .ok()
                .and_then(|()| start.elapsed_time(&end).ok())
                .map(|ms| (ms * 1_000_000.0).round() as u64);
            if let Some(ns) = device_ns {
                crate::profiler::record_device_time(scope, ns);
            }
            Ok(out)
        }
    }

    impl<T: DeviceOp> SyncProfiled for T {}

    /// Submits work on the shared stream without waiting, including during
    /// profiling. All kernels, copies, and cuBLAS calls share this stream;
    /// cudarc frees dropped intermediates in submission order. Downloads and
    /// explicit `synchronize` calls still wait for completion.
    trait AsyncProfiled: DeviceOp {
        fn async_profiled(
            self,
            rt: &Runtime,
        ) -> std::result::Result<<Self as DeviceOp>::Output, cuda_async::error::DeviceError>
        {
            let sample = crate::profiler::current_scope_id().and_then(|scope| {
                let (Ok(start), Ok(end)) = (rt.device.new_event(), rt.device.new_event()) else {
                    return None;
                };
                start.record(&rt.stream).ok()?;
                Some((scope, start, end))
            });
            // SAFETY: single-stream submission and stream-ordered frees keep all
            // buffers alive until the queued work completes.
            let out = unsafe { self.async_on(&rt.stream) }?;
            if let Some((scope, start, end)) = sample
                && end.record(&rt.stream).is_ok()
            {
                crate::profiler::queue_cuda_timing(scope, start, end);
            }
            Ok(out)
        }
    }

    impl<T: DeviceOp> AsyncProfiled for T {}

    /// Allocates `len` elements on the runtime stream and wraps them as an
    /// owned cuTile foreign tensor. The tensor holds the cudarc owner alive;
    /// dropping it frees stream-ordered without host synchronization.
    /// Callers needing `Arc` wrap the result, matching the previous
    /// `api::zeros`/`uninitialized` flow where partitioning consumes the
    /// owned tensor.
    fn alloc_foreign<T>(rt: &Arc<Runtime>, len: usize) -> Result<Tensor<T>>
    where
        T: cuda_core::DType + DeviceRepr,
    {
        let slice: CudaSlice<T> = unsafe {
            rt.cudarc_stream
                .alloc(len)
                .map_err(|e| Error::Cuda(format!("cudarc alloc failed: {e}")))?
        };
        let (ptr, guard) = slice.device_ptr(&rt.cudarc_stream);
        // Fresh allocation: no prior work to order against. Drop the guard
        // before moving the slice into its owner.
        drop(guard);
        let owner =
            SliceOwner { ptr, len_bytes: slice.num_bytes(), device_id: rt.ctx.ordinal(), slice };
        let shape =
            vec![i32::try_from(len).map_err(|_| Error::Cuda("tensor length exceeds i32".into()))?];
        let view = unsafe { Tensor::<T>::from_foreign(Arc::new(owner), shape, vec![1]) };
        Ok(view)
    }

    fn alloc_foreign_zeros<T>(rt: &Arc<Runtime>, len: usize) -> Result<Tensor<T>>
    where
        T: cuda_core::DType + DeviceRepr + ValidAsZeroBits,
    {
        let slice: CudaSlice<T> = rt
            .cudarc_stream
            .alloc_zeros(len)
            .map_err(|e| Error::Cuda(format!("cudarc alloc_zeros failed: {e}")))?;
        let (ptr, guard) = slice.device_ptr(&rt.cudarc_stream);
        drop(guard);
        let owner =
            SliceOwner { ptr, len_bytes: slice.num_bytes(), device_id: rt.ctx.ordinal(), slice };
        let shape =
            vec![i32::try_from(len).map_err(|_| Error::Cuda("tensor length exceeds i32".into()))?];
        let view = unsafe { Tensor::<T>::from_foreign(Arc::new(owner), shape, vec![1]) };
        Ok(view)
    }

    #[cutile::module]
    mod kernels {
        use cutile::core::*;

        #[cutile::entry()]
        pub fn cast<E: ElementType, F: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<F, { [-1] }>,
        ) {
            let tile: Tile<F, { [B] }> = x.load_like(out);
            let result: Tile<E, { [B] }> = convert_tile(tile);
            out.store(result);
        }

        #[cutile::entry()]
        pub fn fill_one<E: ElementType, const B: i32>(out: &mut Tensor<E, { [B] }>) {
            let shape: Shape<{ [B] }> = shape![B];
            let one: E = convert_scalar(1i32);
            out.store(broadcast_scalar(one, shape));
        }

        #[cutile::entry()]
        pub fn add<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            y: &Tensor<E, { [-1] }>,
        ) {
            // Native dtype arithmetic, matching Candle's `binary` kernels
            // (`x + y` on `__nv_bfloat16`). Reductions keep f32 accumulation;
            // elementwise ops do not upcast.
            out.store(x.load_like(out) + y.load_like(out));
        }

        #[cutile::entry()]
        pub fn sub<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            y: &Tensor<E, { [-1] }>,
        ) {
            out.store(x.load_like(out) - y.load_like(out));
        }

        #[cutile::entry()]
        pub fn mul<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            y: &Tensor<E, { [-1] }>,
        ) {
            out.store(x.load_like(out) * y.load_like(out));
        }

        #[cutile::entry()]
        pub fn div<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            y: &Tensor<E, { [-1] }>,
        ) {
            out.store(x.load_like(out) / y.load_like(out));
        }

        #[cutile::entry()]
        pub fn kpow<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            y: &Tensor<E, { [-1] }>,
        ) {
            out.store(pow(x.load_like(out), y.load_like(out)));
        }

        #[cutile::entry()]
        pub fn eq<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            y: &Tensor<E, { [-1] }>,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            let one: E = convert_scalar(1i32);
            let zero: E = convert_scalar(0i32);
            out.store(select(
                eq_tile(x.load_like(out), y.load_like(out)),
                broadcast_scalar(one, shape),
                broadcast_scalar(zero, shape),
            ));
        }

        #[cutile::entry()]
        pub fn neg<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            let zero: E = convert_scalar(0i32);
            out.store(broadcast_scalar(zero, shape) - x.load_like(out));
        }

        #[cutile::entry()]
        pub fn kexp<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
        ) {
            out.store(exp(x.load_like(out)));
        }

        #[cutile::entry()]
        pub fn klog<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
        ) {
            out.store(log(x.load_like(out)));
        }

        #[cutile::entry()]
        pub fn ksin<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
        ) {
            out.store(sin(x.load_like(out)));
        }

        #[cutile::entry()]
        pub fn kcos<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
        ) {
            out.store(cos(x.load_like(out)));
        }

        #[cutile::entry()]
        pub fn ktanh<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
        ) {
            out.store(tanh(x.load_like(out)));
        }

        #[cutile::entry()]
        pub fn relu<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            let zero: E = convert_scalar(0i32);
            out.store(max_tile(x.load_like(out), broadcast_scalar(zero, shape)));
        }

        #[cutile::entry()]
        pub fn relu_backward<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            let one: E = convert_scalar(1i32);
            let zero: E = convert_scalar(0i32);
            let positive = gt_tile(x.load_like(out), broadcast_scalar(zero, shape));
            out.store(select(
                positive,
                broadcast_scalar(one, shape),
                broadcast_scalar(zero, shape),
            ));
        }

        #[cutile::entry()]
        pub fn scalar_add<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            s: E,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            out.store(x.load_like(out) + broadcast_scalar(s, shape));
        }

        #[cutile::entry()]
        pub fn scalar_mul<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            s: E,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            out.store(x.load_like(out) * broadcast_scalar(s, shape));
        }

        #[cutile::entry()]
        pub fn scalar_div<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            s: E,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            out.store(x.load_like(out) / broadcast_scalar(s, shape));
        }

        #[cutile::entry()]
        pub fn scalar_pow<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            s: E,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            out.store(pow(x.load_like(out), broadcast_scalar(s, shape)));
        }

        #[cutile::entry()]
        pub fn ne_scalar<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            x: &Tensor<E, { [-1] }>,
            s: E,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            let one: E = convert_scalar(1i32);
            let zero: E = convert_scalar(0i32);
            let diff = ne_tile(x.load_like(out), broadcast_scalar(s, shape));
            out.store(select(diff, broadcast_scalar(one, shape), broadcast_scalar(zero, shape)));
        }

        /// Integer inequality against a scalar. Separate from [`ne_scalar`](Self::ne_scalar)
        /// because the DSL cannot materialize i64 scalar constants generically.
        #[cutile::entry()]
        pub fn ne_scalar_i64<const B: i32>(
            out: &mut Tensor<i64, { [B] }>,
            x: &Tensor<i64, { [-1] }>,
            s: i64,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            let diff = ne_tile(x.load_like(out), broadcast_scalar(s, shape));
            out.store(select(diff, broadcast_scalar(1i64, shape), broadcast_scalar(0i64, shape)));
        }

        /// Integer select on nonzero conditions. Separate from [`where_op`](Self::where_op)
        /// because the DSL cannot materialize i64 scalar constants generically.
        #[cutile::entry()]
        pub fn where_i64<const B: i32>(
            out: &mut Tensor<i64, { [B] }>,
            c: &Tensor<i64, { [-1] }>,
            t: &Tensor<i64, { [-1] }>,
            f: &Tensor<i64, { [-1] }>,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            let cond = ne_tile(c.load_like(out), broadcast_scalar(0i64, shape));
            out.store(select(cond, t.load_like(out), f.load_like(out)));
        }

        #[cutile::entry()]
        pub fn where_op<E: ElementType, const B: i32>(
            out: &mut Tensor<E, { [B] }>,
            c: &Tensor<E, { [-1] }>,
            t: &Tensor<E, { [-1] }>,
            f: &Tensor<E, { [-1] }>,
        ) {
            let shape: Shape<{ [B] }> = shape![B];
            let zero: E = convert_scalar(0i32);
            let cond = ne_tile(c.load_like(out), broadcast_scalar(zero, shape));
            out.store(select(cond, t.load_like(out), f.load_like(out)));
        }

        #[cutile::entry()]
        pub unsafe fn copy_compact<E: ElementType>(
            out_ptr: *mut E,
            in_ptr: *const E,
            len: i32,
            offset: i32,
            s0: i32,
            s1: i32,
            s2: i32,
            s3: i32,
            s4: i32,
            s5: i32,
            s6: i32,
            s7: i32,
            t0: i32,
            t1: i32,
            t2: i32,
            t3: i32,
            t4: i32,
            t5: i32,
            t6: i32,
            t7: i32,
        ) {
            let grid = get_num_tile_blocks();
            let pid = get_tile_block_id();
            let out_base: PointerTile<*mut E, { [] }> = pointer_to_tile(out_ptr);
            let out_1d: PointerTile<*mut E, { [1] }> = out_base.reshape(shape![1]);
            let out_ptrs: PointerTile<*mut E, { [128] }> = out_1d.broadcast(shape![128]);
            let in_base: PointerTile<*const E, { [] }> = pointer_to_tile(in_ptr);
            let in_1d: PointerTile<*const E, { [1] }> = in_base.reshape(shape![1]);
            let in_ptrs: PointerTile<*const E, { [128] }> = in_1d.broadcast(shape![128]);
            let len_tile: Tile<i32, { [128] }> = broadcast_scalar(len, shape![128]);
            let start: i32 = pid.0 * 128i32;
            let step: i32 = grid.0 * 128i32;
            for base in (start..len).step_by(step as usize) {
                let idx: Tile<i32, { [128] }> =
                    iota(shape![128]) + broadcast_scalar(base, shape![128]);
                let mask: Tile<bool, { [128] }> = lt_tile(idx, len_tile);
                // Decompose the output index into strided coordinates,
                // innermost dimension first. Padding dims use shape 1 and
                // stride 0, which contribute nothing.
                let mut rem = idx;
                let mut src_off: Tile<i32, { [128] }> = broadcast_scalar(offset, shape![128]);
                let sh: Tile<i32, { [128] }> = broadcast_scalar(s7, shape![128]);
                let st: Tile<i32, { [128] }> = broadcast_scalar(t7, shape![128]);
                let c = rem % sh;
                rem = rem / sh;
                src_off = src_off + c * st;
                let sh: Tile<i32, { [128] }> = broadcast_scalar(s6, shape![128]);
                let st: Tile<i32, { [128] }> = broadcast_scalar(t6, shape![128]);
                let c = rem % sh;
                rem = rem / sh;
                src_off = src_off + c * st;
                let sh: Tile<i32, { [128] }> = broadcast_scalar(s5, shape![128]);
                let st: Tile<i32, { [128] }> = broadcast_scalar(t5, shape![128]);
                let c = rem % sh;
                rem = rem / sh;
                src_off = src_off + c * st;
                let sh: Tile<i32, { [128] }> = broadcast_scalar(s4, shape![128]);
                let st: Tile<i32, { [128] }> = broadcast_scalar(t4, shape![128]);
                let c = rem % sh;
                rem = rem / sh;
                src_off = src_off + c * st;
                let sh: Tile<i32, { [128] }> = broadcast_scalar(s3, shape![128]);
                let st: Tile<i32, { [128] }> = broadcast_scalar(t3, shape![128]);
                let c = rem % sh;
                rem = rem / sh;
                src_off = src_off + c * st;
                let sh: Tile<i32, { [128] }> = broadcast_scalar(s2, shape![128]);
                let st: Tile<i32, { [128] }> = broadcast_scalar(t2, shape![128]);
                let c = rem % sh;
                rem = rem / sh;
                src_off = src_off + c * st;
                let sh: Tile<i32, { [128] }> = broadcast_scalar(s1, shape![128]);
                let st: Tile<i32, { [128] }> = broadcast_scalar(t1, shape![128]);
                let c = rem % sh;
                rem = rem / sh;
                src_off = src_off + c * st;
                let sh: Tile<i32, { [128] }> = broadcast_scalar(s0, shape![128]);
                let st: Tile<i32, { [128] }> = broadcast_scalar(t0, shape![128]);
                let c = rem % sh;
                src_off = src_off + c * st;
                let src: PointerTile<*const E, { [128] }> = in_ptrs.offset_tile(src_off);
                // SAFETY: every unmasked lane addresses a valid element: src
                // offsets land inside the caller's allocation by construction
                // of the index math, and dst offsets are contiguous output
                // indices below len.
                let loaded: (Tile<E, { [128] }>, Token) = unsafe {
                    load_ptr_tko(
                        src,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(mask),
                        Some(convert_scalar::<E>(0i32)),
                        None,
                        Latency::<0>,
                    )
                };
                let dst: PointerTile<*mut E, { [128] }> = out_ptrs.offset_tile(idx);
                unsafe {
                    store_ptr_tko(
                        dst,
                        loaded.0,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(mask),
                        None,
                        Latency::<0>,
                    );
                }
            }
        }

        #[cutile::entry()]
        pub fn reduce_sum_f32<const N: i32, const B: i32>(
            x: &Tensor<f32, { [-1, N] }>,
            out: &mut Tensor<f32, { [1] }>,
        ) {
            // One output element per tile block.
            let row = get_tile_block_id().0;
            let shape: Shape<{ [1, B] }> = shape![1, B];
            let xp: Partition<f32, { [1, B] }> = x.partition(shape);
            let mut acc: Tile<f32, { [1, B] }> = constant(0.0, shape);
            for j in 0i32..(N / B) {
                acc = acc + xp.load([row, j]);
            }
            let total: Tile<f32, { [1] }> = reduce_sum(acc, 1i32);
            out.partition_mut(shape![1]).store(total, [0i32]);
        }

        #[cutile::entry()]
        pub fn reduce_max_f32<const N: i32, const B: i32>(
            x: &Tensor<f32, { [-1, N] }>,
            out: &mut Tensor<f32, { [1] }>,
        ) {
            let row = get_tile_block_id().0;
            let shape: Shape<{ [1, B] }> = shape![1, B];
            let xp: Partition<f32, { [1, B] }> = x.partition(shape);
            let mut acc: Tile<f32, { [1, B] }> = constant(f32::NEG_INFINITY, shape);
            for j in 0i32..(N / B) {
                acc = max_tile(acc, xp.load([row, j]));
            }
            let total: Tile<f32, { [1] }> = reduce_max(acc, 1i32);
            out.partition_mut(shape![1]).store(total, [0i32]);
        }

        /// Row-wise log-softmax: `dst = x - log(sum(exp(x)))`, stabilized by
        /// subtracting the row max first.
        #[cutile::entry()]
        pub fn log_softmax_fwd<E: ElementType, const N: i32, const B: i32>(
            x: &Tensor<E, { [-1, N] }>,
            out: &mut Tensor<E, { [1, N] }>,
        ) {
            let row = get_tile_block_id().0;
            let shape: Shape<{ [1, B] }> = shape![1, B];
            let xp: Partition<E, { [1, B] }> = x.partition(shape);
            let mut op: PartitionMut<E, { [1, B] }> = out.partition_mut(shape);
            let mut mx: Tile<f32, { [1, B] }> = convert_tile(xp.load([row, 0i32]));
            for j in 0i32..(N / B) {
                let values: Tile<f32, { [1, B] }> = convert_tile(xp.load([row, j]));
                mx = max_tile(mx, values);
            }
            let mx1: Tile<f32, { [1] }> = reduce_max(mx, 1i32);
            let mxb: Tile<f32, { [1, B] }> = mx1.reshape(shape![1, 1]).broadcast(shape);
            let mut se: Tile<f32, { [1, B] }> = constant(0.0, shape);
            for j in 0i32..(N / B) {
                let values: Tile<f32, { [1, B] }> = convert_tile(xp.load([row, j]));
                se = se + exp(values - mxb);
            }
            let se1: Tile<f32, { [1] }> = reduce_sum(se, 1i32);
            let lse: Tile<f32, { [1, B] }> = log(se1).reshape(shape![1, 1]).broadcast(shape) + mxb;
            for j in 0i32..(N / B) {
                let values: Tile<f32, { [1, B] }> = convert_tile(xp.load([row, j]));
                let result: Tile<E, { [1, B] }> = convert_tile(values - lse);
                op.store(result, [0i32, j]);
            }
        }

        /// Gradient of log-softmax given the saved forward output `lsm`:
        /// `dst = grad - exp(lsm) * sum(grad)` per row.
        #[cutile::entry()]
        pub fn log_softmax_bwd_f32<const N: i32, const B: i32>(
            grad: &Tensor<f32, { [-1, N] }>,
            lsm: &Tensor<f32, { [-1, N] }>,
            out: &mut Tensor<f32, { [1, N] }>,
        ) {
            let row = get_tile_block_id().0;
            let shape: Shape<{ [1, B] }> = shape![1, B];
            let gp: Partition<f32, { [1, B] }> = grad.partition(shape);
            let lp: Partition<f32, { [1, B] }> = lsm.partition(shape);
            let mut op: PartitionMut<f32, { [1, B] }> = out.partition_mut(shape);
            // Sums start from zero; the loop always runs at least once.
            let mut sg: Tile<f32, { [1, B] }> = constant(0.0, shape);
            for j in 0i32..(N / B) {
                sg = sg + gp.load([row, j]);
            }
            let sg1: Tile<f32, { [1] }> = reduce_sum(sg, 1i32);
            let sb: Tile<f32, { [1, B] }> = sg1.reshape(shape![1, 1]).broadcast(shape);
            for j in 0i32..(N / B) {
                let lsm_tile: Tile<f32, { [1, B] }> = lp.load([row, j]);
                op.store(gp.load([row, j]) - exp(lsm_tile) * sb, [0i32, j]);
            }
        }

        /// Row-wise log-sum-exp: `log(sum(exp(x - max))) + max`.
        #[cutile::entry()]
        pub fn log_sum_exp_f32<const N: i32, const B: i32>(
            x: &Tensor<f32, { [-1, N] }>,
            out: &mut Tensor<f32, { [1] }>,
        ) {
            let row = get_tile_block_id().0;
            let shape: Shape<{ [1, B] }> = shape![1, B];
            let xp: Partition<f32, { [1, B] }> = x.partition(shape);
            // The loop starts at zero (reloading the seed tile) because the
            // compiler rejects loops that may run zero times when N equals B.
            let mut mx: Tile<f32, { [1, B] }> = xp.load([row, 0i32]);
            for j in 0i32..(N / B) {
                mx = max_tile(mx, xp.load([row, j]));
            }
            let mx1: Tile<f32, { [1] }> = reduce_max(mx, 1i32);
            let mxb: Tile<f32, { [1, B] }> = mx1.reshape(shape![1, 1]).broadcast(shape);
            let mut se: Tile<f32, { [1, B] }> = constant(0.0, shape);
            for j in 0i32..(N / B) {
                se = se + exp(xp.load([row, j]) - mxb);
            }
            let se1: Tile<f32, { [1] }> = reduce_sum(se, 1i32);
            let mx1: Tile<f32, { [1] }> = reduce_max(mx, 1i32);
            let total: Tile<f32, { [1] }> = log(se1) + mx1;
            out.partition_mut(shape![1]).store(total, [0i32]);
        }

        /// Gathers elements along one axis. Indices load through a regular
        /// partition (they are contiguous); only the source side needs a
        /// pointer gather. All shapes arrive as scalars.
        #[cutile::entry()]
        pub unsafe fn gather<E: ElementType>(
            idx: &Tensor<i64, { [-1] }>,
            src_ptr: *const E,
            out: &mut Tensor<E, { [128] }>,
            len: i32,
            dst_dim: i32,
            right: i32,
            src_dim: i32,
        ) {
            // Valid indices are bounded by src_dim, which fits i32.
            let pid = get_tile_block_id();
            let shape128: Shape<{ [128] }> = shape![128];
            let ip: Partition<i64, { [128] }> = idx.partition(shape128);
            let v64: Tile<i64, { [128] }> = ip.load([pid.0]);
            let v: Tile<i32, { [128] }> = trunci(v64, overflow::NoWrap);
            let src_base: PointerTile<*const E, { [] }> = pointer_to_tile(src_ptr);
            let src_1d: PointerTile<*const E, { [1] }> = src_base.reshape(shape![1]);
            let src_ptrs: PointerTile<*const E, { [128] }> = src_1d.broadcast(shape![128]);
            let lane: Tile<i32, { [128] }> =
                iota(shape128) + broadcast_scalar(pid.0 * 128i32, shape128);
            let mask: Tile<bool, { [128] }> = lt_tile(lane, broadcast_scalar(len, shape128));
            let row_len: Tile<i32, { [128] }> = broadcast_scalar(dst_dim * right, shape128);
            let right_tile: Tile<i32, { [128] }> = broadcast_scalar(right, shape128);
            let src_tile: Tile<i32, { [128] }> = broadcast_scalar(src_dim, shape128);
            let left: Tile<i32, { [128] }> = lane / row_len;
            let tmp: Tile<i32, { [128] }> = lane % row_len;
            let rpos: Tile<i32, { [128] }> = tmp % right_tile;
            let src_off: Tile<i32, { [128] }> = (left * src_tile + v) * right_tile + rpos;
            let src: PointerTile<*const E, { [128] }> = src_ptrs.offset_tile(src_off);
            let (vals, _): (Tile<E, { [128] }>, Token) = unsafe {
                load_ptr_tko(
                    src,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(mask),
                    Some(convert_scalar::<E>(0i32)),
                    None,
                    Latency::<0>,
                )
            };
            out.store(vals);
        }

        /// Selects slices along one axis using a 1-D index tensor. The index
        /// positions scatter across the 1-D indices array, so both sides use
        /// pointer gathers.
        #[cutile::entry()]
        pub unsafe fn index_select<E: ElementType>(
            idx_ptr: *const i64,
            src_ptr: *const E,
            out: &mut Tensor<E, { [128] }>,
            len: i32,
            index_len: i32,
            right: i32,
            src_dim: i32,
        ) {
            let pid = get_tile_block_id();
            let shape128: Shape<{ [128] }> = shape![128];
            let idx_base: PointerTile<*const i64, { [] }> = pointer_to_tile(idx_ptr);
            let idx_1d: PointerTile<*const i64, { [1] }> = idx_base.reshape(shape![1]);
            let idx_ptrs: PointerTile<*const i64, { [128] }> = idx_1d.broadcast(shape![128]);
            let src_base: PointerTile<*const E, { [] }> = pointer_to_tile(src_ptr);
            let src_1d: PointerTile<*const E, { [1] }> = src_base.reshape(shape![1]);
            let src_ptrs: PointerTile<*const E, { [128] }> = src_1d.broadcast(shape![128]);
            let lane: Tile<i32, { [128] }> =
                iota(shape128) + broadcast_scalar(pid.0 * 128i32, shape128);
            let mask: Tile<bool, { [128] }> = lt_tile(lane, broadcast_scalar(len, shape128));
            let row_len: Tile<i32, { [128] }> = broadcast_scalar(index_len * right, shape128);
            let right_tile: Tile<i32, { [128] }> = broadcast_scalar(right, shape128);
            let src_tile: Tile<i32, { [128] }> = broadcast_scalar(src_dim, shape128);
            let left: Tile<i32, { [128] }> = lane / row_len;
            let tmp: Tile<i32, { [128] }> = lane % row_len;
            let oc: Tile<i32, { [128] }> = tmp / right_tile;
            let rpos: Tile<i32, { [128] }> = tmp % right_tile;
            let idx_at: PointerTile<*const i64, { [128] }> = idx_ptrs.offset_tile(oc);
            let (v64, _): (Tile<i64, { [128] }>, Token) = unsafe {
                load_ptr_tko(
                    idx_at,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(mask),
                    Some(0i64),
                    None,
                    Latency::<0>,
                )
            };
            // Valid indices are bounded by src_dim, which fits i32.
            let v: Tile<i32, { [128] }> = trunci(v64, overflow::NoWrap);
            let src_off: Tile<i32, { [128] }> = (left * src_tile + v) * right_tile + rpos;
            let src_at: PointerTile<*const E, { [128] }> = src_ptrs.offset_tile(src_off);
            let (vals, _): (Tile<E, { [128] }>, Token) = unsafe {
                load_ptr_tko(
                    src_at,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(mask),
                    Some(convert_scalar::<E>(0i32)),
                    None,
                    Latency::<0>,
                )
            };
            out.store(vals);
        }

        /// Scatter-adds source rows into a zero-initialized destination.
        /// Source and indices share a layout; each lane atomically accumulates
        /// one value so duplicate indices stay correct.
        #[cutile::entry()]
        pub unsafe fn scatter_add<E: ElementType>(
            src: &Tensor<E, { [-1] }>,
            idx: &Tensor<i64, { [-1] }>,
            dst_ptr: *mut E,
            len: i32,
            index_len: i32,
            right: i32,
            dst_dim: i32,
        ) {
            let pid = get_tile_block_id();
            let shape128: Shape<{ [128] }> = shape![128];
            let sp: Partition<E, { [128] }> = src.partition(shape128);
            let sv: Tile<E, { [128] }> = sp.load([pid.0]);
            let ip: Partition<i64, { [128] }> = idx.partition(shape128);
            let v64: Tile<i64, { [128] }> = ip.load([pid.0]);
            // Valid indices are bounded by dst_dim, which fits i32.
            let v: Tile<i32, { [128] }> = trunci(v64, overflow::NoWrap);
            let dst_base: PointerTile<*mut E, { [] }> = pointer_to_tile(dst_ptr);
            let dst_1d: PointerTile<*mut E, { [1] }> = dst_base.reshape(shape![1]);
            let dst_ptrs: PointerTile<*mut E, { [128] }> = dst_1d.broadcast(shape![128]);
            let lane: Tile<i32, { [128] }> =
                iota(shape128) + broadcast_scalar(pid.0 * 128i32, shape128);
            let mask: Tile<bool, { [128] }> = lt_tile(lane, broadcast_scalar(len, shape128));
            let row_len: Tile<i32, { [128] }> = broadcast_scalar(index_len * right, shape128);
            let right_tile: Tile<i32, { [128] }> = broadcast_scalar(right, shape128);
            let dst_tile: Tile<i32, { [128] }> = broadcast_scalar(dst_dim, shape128);
            // Source and indices share a layout, so each lane reads its own
            // index value; only the destination position needs computing.
            let left: Tile<i32, { [128] }> = lane / row_len;
            let tmp: Tile<i32, { [128] }> = lane % row_len;
            let rpos: Tile<i32, { [128] }> = tmp % right_tile;
            let dst_off: Tile<i32, { [128] }> = (left * dst_tile + v) * right_tile + rpos;
            let dst_at: PointerTile<*mut E, { [128] }> = dst_ptrs.offset_tile(dst_off);
            let (_, _): (Tile<E, { [128] }>, Token) = unsafe {
                atomic_rmw_tko(
                    dst_at,
                    sv,
                    atomic::AddF,
                    ordering::Relaxed,
                    scope::Device,
                    Some(mask),
                    None,
                )
            };
        }

        /// Adds source slices into a zero-initialized destination of shape
        /// `dst_shape`, like [`scatter_add`](Self::scatter_add) but with the
        /// source's own indexed dimension.
        #[cutile::entry()]
        pub unsafe fn index_add<E: ElementType>(
            idx_ptr: *const i64,
            src: &Tensor<E, { [-1] }>,
            dst_ptr: *mut E,
            len: i32,
            src_dim: i32,
            right: i32,
            dst_dim: i32,
        ) {
            let pid = get_tile_block_id();
            let shape128: Shape<{ [128] }> = shape![128];
            let sp: Partition<E, { [128] }> = src.partition(shape128);
            let sv: Tile<E, { [128] }> = sp.load([pid.0]);
            let idx_base: PointerTile<*const i64, { [] }> = pointer_to_tile(idx_ptr);
            let idx_1d: PointerTile<*const i64, { [1] }> = idx_base.reshape(shape![1]);
            let idx_ptrs: PointerTile<*const i64, { [128] }> = idx_1d.broadcast(shape![128]);
            let dst_base: PointerTile<*mut E, { [] }> = pointer_to_tile(dst_ptr);
            let dst_1d: PointerTile<*mut E, { [1] }> = dst_base.reshape(shape![1]);
            let dst_ptrs: PointerTile<*mut E, { [128] }> = dst_1d.broadcast(shape![128]);
            let lane: Tile<i32, { [128] }> =
                iota(shape128) + broadcast_scalar(pid.0 * 128i32, shape128);
            let mask: Tile<bool, { [128] }> = lt_tile(lane, broadcast_scalar(len, shape128));
            let row_len: Tile<i32, { [128] }> = broadcast_scalar(src_dim * right, shape128);
            let right_tile: Tile<i32, { [128] }> = broadcast_scalar(right, shape128);
            let dst_tile: Tile<i32, { [128] }> = broadcast_scalar(dst_dim, shape128);
            let left: Tile<i32, { [128] }> = lane / row_len;
            let tmp: Tile<i32, { [128] }> = lane % row_len;
            let sc: Tile<i32, { [128] }> = tmp / right_tile;
            let rpos: Tile<i32, { [128] }> = tmp % right_tile;
            let idx_at: PointerTile<*const i64, { [128] }> = idx_ptrs.offset_tile(sc);
            let (v64, _): (Tile<i64, { [128] }>, Token) = unsafe {
                load_ptr_tko(
                    idx_at,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(mask),
                    Some(0i64),
                    None,
                    Latency::<0>,
                )
            };
            // Valid indices are bounded by dst_dim, which fits i32.
            let v: Tile<i32, { [128] }> = trunci(v64, overflow::NoWrap);
            let dst_off: Tile<i32, { [128] }> = (left * dst_tile + v) * right_tile + rpos;
            let dst_at: PointerTile<*mut E, { [128] }> = dst_ptrs.offset_tile(dst_off);
            let (_, _): (Tile<E, { [128] }>, Token) = unsafe {
                atomic_rmw_tko(
                    dst_at,
                    sv,
                    atomic::AddF,
                    ordering::Relaxed,
                    scope::Device,
                    Some(mask),
                    None,
                )
            };
        }

        #[cutile::entry()]
        pub fn rms_norm<E: ElementType, const N: i32, const B: i32>(
            x: &Tensor<E, { [-1, N] }>,
            w: &Tensor<E, { [N] }>,
            out: &mut Tensor<E, { [1, N] }>,
            eps: f32,
            weighted: i32,
        ) {
            // One output row per tile block. B must divide N.
            let row = get_tile_block_id().0;
            let shape: Shape<{ [1, B] }> = shape![1, B];
            let xp: Partition<E, { [1, B] }> = x.partition(shape);
            let wp: Partition<E, { [B] }> = w.partition(shape![B]);
            let mut op: PartitionMut<E, { [1, B] }> = out.partition_mut(shape);
            // First pass accumulates the sum of squares for the row.
            let mut squares: Tile<f32, { [1, B] }> = constant(0.0, shape);
            for j in 0i32..(N / B) {
                let values: Tile<f32, { [1, B] }> = convert_tile(xp.load([row, j]));
                squares = squares + values * values;
            }
            let sum: Tile<f32, { [1] }> = reduce_sum(squares, 1i32);
            let sum: f32 = tile_to_scalar(sum.reshape(shape![]));
            let n: f32 = convert_scalar(N);
            let inv: f32 = 1.0f32 / (sum / n + eps);
            let inv_tile: Tile<f32, { [] }> =
                sqrt(scalar_to_tile(inv), rounding::NegativeInf, ftz::Disabled);
            let inv: f32 = tile_to_scalar(inv_tile);
            let scale: Tile<f32, { [1, B] }> = broadcast_scalar(inv, shape);
            // Second pass scales the row by the single normalization factor.
            for j in 0i32..(N / B) {
                let values: Tile<f32, { [1, B] }> = convert_tile(xp.load([row, j]));
                let mut result: Tile<f32, { [1, B] }> = values * scale;
                if weighted != 0 {
                    let weights: Tile<f32, { [1, B] }> = convert_tile(wp.load([j]).reshape(shape));
                    result = result * weights;
                }
                let result: Tile<E, { [1, B] }> = convert_tile(result);
                op.store(result, [0i32, j]);
            }
        }
    }

    /// cudarc-owned device allocation backing a cuTile foreign tensor view.
    /// The `CudaSlice` frees stream-ordered on the runtime's single stream,
    /// so dropping it after queued work is safe without host waits.
    struct SliceOwner<T> {
        slice: CudaSlice<T>,
        ptr: u64,
        len_bytes: usize,
        device_id: usize,
    }

    unsafe impl<T: Send + Sync + 'static> DeviceAllocation for SliceOwner<T> {
        fn device_ptr(&self) -> u64 {
            self.ptr
        }
        fn len_bytes(&self) -> usize {
            self.len_bytes
        }
        fn device_id(&self) -> usize {
            self.device_id
        }
    }

    #[derive(Debug)]
    struct Runtime {
        ctx: Arc<CudaContext>,
        cudarc_stream: Arc<CudarcStream>,
        device: Arc<TileDevice>,
        stream: Arc<TileStream>,
        blas: OnceLock<std::result::Result<Mutex<BlasHandle>, String>>,
    }

    #[derive(Debug)]
    struct BlasHandle {
        raw: blas_sys::cublasHandle_t,
        device: Arc<TileDevice>,
    }

    // The handle is used only while its mutex is held. Each call binds its
    // owning context before launch; the shared stream orders later work.
    unsafe impl Send for BlasHandle {}

    impl Drop for BlasHandle {
        fn drop(&mut self) {
            if self.device.bind_to_thread().is_ok() {
                let _ = unsafe { cublas::destroy_handle(self.raw) };
            }
        }
    }

    impl Runtime {
        fn blas(&self) -> Result<&Mutex<BlasHandle>> {
            match self.blas.get_or_init(|| {
                (|| -> std::result::Result<_, String> {
                    self.device.bind_to_thread().map_err(|e| e.to_string())?;
                    let raw = cublas::create_handle().map_err(|e| e.to_string())?;
                    if let Err(e) = unsafe { cublas::set_stream(raw, self.stream.cu_stream() as _) }
                    {
                        let _ = unsafe { cublas::destroy_handle(raw) };
                        return Err(e.to_string());
                    }
                    Ok(Mutex::new(BlasHandle { raw, device: self.device.clone() }))
                })()
            }) {
                Ok(handle) => Ok(handle),
                Err(e) => Err(Error::Cuda(format!("cuBLAS initialization failed: {e}"))),
            }
        }
    }

    static RUNTIME: OnceLock<std::result::Result<Arc<Runtime>, String>> = OnceLock::new();

    fn runtime() -> Result<Arc<Runtime>> {
        match RUNTIME.get_or_init(|| {
            (|| {
                let ctx = CudaContext::new(0)
                    .map_err(|e| Error::Cuda(format!("cudarc context init failed: {e}")))?;
                let cudarc_stream = ctx
                    .new_stream()
                    .map_err(|e| Error::Cuda(format!("cudarc stream init failed: {e}")))?;
                // Borrow the cudarc context and stream for cuTile without
                // transferring ownership, mirroring Candle's CutileContext.
                // All allocations, kernels, copies, and cuBLAS share this
                // single stream, so cudarc's stream-ordered free is safe.
                let device = unsafe {
                    TileDevice::borrow_with_owner(
                        ctx.cu_ctx() as *mut core::ffi::c_void,
                        ctx.cu_device() as core::ffi::c_int,
                        ctx.ordinal(),
                        ctx.clone(),
                    )
                };
                let stream = unsafe {
                    TileStream::borrow_with_owner(
                        cudarc_stream.cu_stream() as *mut core::ffi::c_void,
                        &device,
                        cudarc_stream.clone(),
                    )
                };
                Ok(Arc::new(Runtime { ctx, cudarc_stream, device, stream, blas: OnceLock::new() }))
            })()
            .map_err(|e: Error| e.to_string())
        }) {
            Ok(runtime) => Ok(runtime.clone()),
            Err(err) => Err(Error::Cuda(err.clone())),
        }
    }

    // cuBLAS uses column-major matrices. A row-major [rows, cols] view is
    // therefore a column-major [cols, rows] matrix, with operands reversed
    // when computing the output.
    #[derive(Clone, Copy)]
    struct GemmOperand {
        op: blas_sys::cublasOperation_t,
        ld: i32,
        batch_stride: i64,
        offset: usize,
    }

    impl GemmOperand {
        fn dense(rows: usize, cols: usize) -> Self {
            Self {
                op: blas_sys::cublasOperation_t::CUBLAS_OP_N,
                ld: i32::try_from(cols).expect("GEMM dimension exceeds i32"),
                batch_stride: i64::try_from(rows * cols).expect("GEMM batch stride exceeds i64"),
                offset: 0,
            }
        }

        fn from_layout(layout: &Layout, rows: usize, cols: usize) -> Option<Self> {
            let shape = layout.shape();
            let strides = layout.strides();
            let ndim = layout.ndim();
            let mut expected = rows.checked_mul(cols)?;
            for i in (0..ndim.saturating_sub(2)).rev() {
                if strides[i] != isize::try_from(expected).ok()? {
                    return None;
                }
                expected = expected.checked_mul(shape[i])?;
            }
            let (last, second) = (strides[ndim - 1], strides[ndim - 2]);
            let (op, ld) = if (last == 1 || cols == 1)
                && (second == isize::try_from(cols).ok()? || rows == 1)
            {
                (blas_sys::cublasOperation_t::CUBLAS_OP_N, cols)
            } else if (last == isize::try_from(rows).ok()? || cols == 1)
                && (second == 1 || rows == 1)
            {
                (blas_sys::cublasOperation_t::CUBLAS_OP_T, rows)
            } else {
                return None;
            };
            Some(Self {
                op,
                ld: i32::try_from(ld).ok()?,
                batch_stride: i64::try_from(rows.checked_mul(cols)?).ok()?,
                offset: layout.offset,
            })
        }
    }

    #[derive(Debug)]
    pub enum CudaInner {
        F16(Arc<Tensor<f16>>),
        BF16(Arc<Tensor<bf16>>),
        F32(Arc<Tensor<f32>>),
        I64(Arc<Tensor<i64>>),
    }

    #[derive(Debug, Clone)]
    pub struct CudaStorage {
        inner: CudaInner,
        runtime: Arc<Runtime>,
    }

    impl Clone for CudaInner {
        fn clone(&self) -> Self {
            match self {
                CudaInner::F16(t) => CudaInner::F16(t.clone()),
                CudaInner::BF16(t) => CudaInner::BF16(t.clone()),
                CudaInner::F32(t) => CudaInner::F32(t.clone()),
                CudaInner::I64(t) => CudaInner::I64(t.clone()),
            }
        }
    }

    impl CudaStorage {
        pub fn is_available() -> bool {
            runtime().is_ok()
        }

        fn alloc_tensor<T>(runtime: &Arc<Runtime>, len: usize) -> Result<Arc<Tensor<T>>>
        where
            T: cuda_core::DType + DeviceRepr + ValidAsZeroBits,
        {
            Ok(Arc::new(alloc_foreign_zeros(runtime, len)?))
        }

        /// Converts float storage to f32, cloning f32 inputs. Row-wise kernels
        /// run in f32 exactly like the old CUDA kernels did.
        fn to_f32_storage(&self) -> Result<Arc<Tensor<f32>>> {
            let rt = self.runtime.clone();
            match &self.inner {
                CudaInner::F16(t) => cast_tensor(&rt, t),
                CudaInner::BF16(t) => cast_tensor(&rt, t),
                CudaInner::F32(t) => Ok(t.clone()),
                CudaInner::I64(_) => Err(Error::DTypeMismatch("expected a float dtype".into())),
            }
        }

        /// Wraps a computed f32 tensor back into the given dtype.
        fn from_f32_storage(
            rt: &Arc<Runtime>,
            out: Arc<Tensor<f32>>,
            dtype: DType,
        ) -> Result<CudaInner> {
            match dtype {
                DType::F32 => Ok(CudaInner::F32(out)),
                DType::F16 => Ok(CudaInner::F16(cast_tensor(rt, &out)?)),
                DType::BF16 => Ok(CudaInner::BF16(cast_tensor(rt, &out)?)),
                DType::I64 => Err(Error::DTypeMismatch("expected a float dtype".into())),
            }
        }

        fn for_overwrite(size: usize, dtype: DType) -> Result<Self> {
            fn alloc<T>(rt: &Arc<Runtime>, size: usize) -> Result<Arc<Tensor<T>>>
            where
                T: cuda_core::DType + DeviceRepr,
            {
                // cudarc leaves the memory uninitialized; the caller must
                // overwrite every element before reading the tensor.
                Ok(Arc::new(alloc_foreign(rt, size)?))
            }
            if size == 0 {
                return Self::uninit(0, dtype);
            }
            let runtime = runtime()?;
            let inner = match dtype {
                DType::F16 => CudaInner::F16(alloc(&runtime, size)?),
                DType::BF16 => CudaInner::BF16(alloc(&runtime, size)?),
                DType::F32 => CudaInner::F32(alloc(&runtime, size)?),
                DType::I64 => CudaInner::I64(alloc(&runtime, size)?),
            };
            Ok(Self { inner, runtime })
        }

        fn uninit(size: usize, dtype: DType) -> Result<Self> {
            let runtime = runtime()?;
            let inner = match dtype {
                DType::F16 => CudaInner::F16(Self::alloc_tensor::<f16>(&runtime, size)?),
                DType::BF16 => CudaInner::BF16(Self::alloc_tensor::<bf16>(&runtime, size)?),
                DType::F32 => CudaInner::F32(Self::alloc_tensor::<f32>(&runtime, size)?),
                DType::I64 => CudaInner::I64(Self::alloc_tensor::<i64>(&runtime, size)?),
            };
            Ok(Self { inner, runtime })
        }

        pub fn zeros(size: usize, dtype: DType) -> Self {
            Self::uninit(size, dtype).expect("cuda backend unavailable")
        }

        /// Uploads pageable host data using cudarc's Candle-style copy path.
        fn upload_foreign<T: cuda_core::DType + DeviceRepr>(
            rt: &Arc<Runtime>,
            data: &[T],
        ) -> Result<Tensor<T>> {
            let slice: CudaSlice<T> = rt
                .cudarc_stream
                .clone_htod(data)
                .map_err(|e| Error::Cuda(format!("cudarc upload failed: {e}")))?;
            let (ptr, guard) = slice.device_ptr(&rt.cudarc_stream);
            drop(guard);
            let owner = SliceOwner {
                ptr,
                len_bytes: slice.num_bytes(),
                device_id: rt.ctx.ordinal(),
                slice,
            };
            let shape = vec![
                i32::try_from(data.len())
                    .map_err(|_| Error::Cuda("tensor length exceeds i32".into()))?,
            ];
            let view = unsafe { Tensor::<T>::from_foreign(Arc::new(owner), shape, vec![1]) };
            Ok(view)
        }

        pub fn ones(size: usize, dtype: DType) -> Self {
            let runtime = runtime().expect("cuda backend unavailable");
            let inner = match dtype {
                DType::F16 => {
                    let t = Arc::new(ones_foreign(&runtime, size).expect("ones failed"));
                    CudaInner::F16(t)
                }
                DType::BF16 => {
                    let t = Arc::new(ones_foreign(&runtime, size).expect("ones failed"));
                    CudaInner::BF16(t)
                }
                DType::F32 => {
                    let t = Arc::new(ones_foreign(&runtime, size).expect("ones failed"));
                    CudaInner::F32(t)
                }
                DType::I64 => {
                    let t = Arc::new(ones_foreign(&runtime, size).expect("ones failed"));
                    CudaInner::I64(t)
                }
            };
            Self { inner, runtime }
        }

        pub fn from_cpu_storage(inner: CpuStorage) -> Self {
            fn upload<T: cuda_core::DType + DeviceRepr>(
                runtime: &Arc<Runtime>,
                data: &[T],
            ) -> Arc<Tensor<T>> {
                Arc::new(CudaStorage::upload_foreign(runtime, data).expect("cuda upload failed"))
            }
            let runtime = runtime().expect("cuda backend unavailable");
            let inner = match inner {
                CpuStorage::F16(data) => CudaInner::F16(upload(&runtime, &data)),
                CpuStorage::BF16(data) => CudaInner::BF16(upload(&runtime, &data)),
                CpuStorage::F32(data) => CudaInner::F32(upload(&runtime, &data)),
                CpuStorage::I64(data) => CudaInner::I64(upload(&runtime, &data)),
            };
            Self { inner, runtime }
        }

        pub(crate) fn copy_to_device(&self, layout: &Layout) -> Result<Self> {
            let compact = self.compact(layout)?;
            let out = Self::uninit(compact.len(), compact.dtype())?;
            out.copy_from(&compact)?;
            Ok(out)
        }

        pub(crate) fn copy_to_cpu(&self, layout: &Layout) -> Result<CpuStorage> {
            let compact = self.compact(layout)?;
            let err = |e| Error::Cuda(format!("cuda download failed: {e}"));
            match &compact.inner {
                CudaInner::F16(t) => {
                    let v: Vec<f16> =
                        t.to_host_vec().sync_profiled(&compact.runtime).map_err(err)?;
                    Ok(CpuStorage::F16(v))
                }
                CudaInner::BF16(t) => {
                    let v: Vec<bf16> =
                        t.to_host_vec().sync_profiled(&compact.runtime).map_err(err)?;
                    Ok(CpuStorage::BF16(v))
                }
                CudaInner::F32(t) => {
                    let v: Vec<f32> =
                        t.to_host_vec().sync_profiled(&compact.runtime).map_err(err)?;
                    Ok(CpuStorage::F32(v))
                }
                CudaInner::I64(t) => {
                    let v: Vec<i64> =
                        t.to_host_vec().sync_profiled(&compact.runtime).map_err(err)?;
                    Ok(CpuStorage::I64(v))
                }
            }
        }

        pub(crate) fn copy_from_cpu(src: &CpuStorage, layout: &Layout) -> Result<Self> {
            let runtime = runtime()?;
            let inner = match src {
                CpuStorage::F16(data) => {
                    let staged = CpuStorage::borrow_or_compact(data, src, layout);
                    CudaInner::F16(Arc::new(Self::upload_foreign(&runtime, &staged)?))
                }
                CpuStorage::BF16(data) => {
                    let staged = CpuStorage::borrow_or_compact(data, src, layout);
                    CudaInner::BF16(Arc::new(Self::upload_foreign(&runtime, &staged)?))
                }
                CpuStorage::F32(data) => {
                    let staged = CpuStorage::borrow_or_compact(data, src, layout);
                    CudaInner::F32(Arc::new(Self::upload_foreign(&runtime, &staged)?))
                }
                CpuStorage::I64(data) => {
                    let staged = CpuStorage::borrow_or_compact(data, src, layout);
                    CudaInner::I64(Arc::new(Self::upload_foreign(&runtime, &staged)?))
                }
            };
            Ok(Self { inner, runtime })
        }

        pub fn cat(parts: &[(&CudaStorage, usize)]) -> Result<Self> {
            if parts.is_empty() {
                return Err(Error::LayoutMismatch("cat: empty parts".into()));
            }
            let dtype = parts[0].0.dtype();
            let total_len: usize = parts.iter().map(|(_, len)| *len).sum();
            let out = Self::uninit(total_len, dtype)?;
            let mut offset = 0usize;
            for (part, len) in parts {
                if *len > 0 {
                    let src = part.compact_full(*len)?;
                    out.copy_from_offset(&src, offset)?;
                }
                offset += *len;
            }
            Ok(out)
        }

        fn len(&self) -> usize {
            match &self.inner {
                CudaInner::F16(t) => t.size(),
                CudaInner::BF16(t) => t.size(),
                CudaInner::F32(t) => t.size(),
                CudaInner::I64(t) => t.size(),
            }
        }

        fn compact(&self, layout: &Layout) -> Result<Self> {
            if layout.is_compact() && layout.offset == 0 && layout.size() == self.len() {
                return Ok(self.clone());
            }
            let mut out = Self::for_overwrite(layout.size(), self.dtype())?;
            self.copy_compact(layout, &mut out)?;
            Ok(out)
        }

        fn compact_full(&self, len: usize) -> Result<Self> {
            if len == self.len() {
                return Ok(self.clone());
            }
            let out = Self::uninit(len, self.dtype())?;
            out.copy_prefix(self, len)?;
            Ok(out)
        }

        fn copy_from(&self, src: &Self) -> Result<()> {
            self.copy_prefix(src, src.len())
        }

        fn copy_prefix(&self, src: &Self, len: usize) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            use cuda_core::memcpy_dtod_async;
            let err = |e| {
                Error::Cuda(format!(
                    "cuda device copy failed (len {len}, src {} dst {}): {e}",
                    src.len(),
                    self.len()
                ))
            };
            let stream = &self.runtime.stream;
            match (&src.inner, &self.inner) {
                (CudaInner::F16(s), CudaInner::F16(d)) => unsafe {
                    memcpy_dtod_async::<f16>(
                        d.device_pointer().cu_deviceptr(),
                        s.device_pointer().cu_deviceptr(),
                        len,
                        stream,
                    )
                    .map_err(err)?;
                },
                (CudaInner::BF16(s), CudaInner::BF16(d)) => unsafe {
                    memcpy_dtod_async::<bf16>(
                        d.device_pointer().cu_deviceptr(),
                        s.device_pointer().cu_deviceptr(),
                        len,
                        stream,
                    )
                    .map_err(err)?;
                },
                (CudaInner::F32(s), CudaInner::F32(d)) => unsafe {
                    memcpy_dtod_async::<f32>(
                        d.device_pointer().cu_deviceptr(),
                        s.device_pointer().cu_deviceptr(),
                        len,
                        stream,
                    )
                    .map_err(err)?;
                },
                (CudaInner::I64(s), CudaInner::I64(d)) => unsafe {
                    memcpy_dtod_async::<i64>(
                        d.device_pointer().cu_deviceptr(),
                        s.device_pointer().cu_deviceptr(),
                        len,
                        stream,
                    )
                    .map_err(err)?;
                },
                _ => return Err(Error::DTypeMismatch("device copy dtype mismatch".into())),
            }
            // Single-stream cudarc ownership: the queued copy precedes the
            // stream-ordered free, so no host wait is needed.
            Ok(())
        }

        fn copy_from_offset(&self, src: &Self, offset: usize) -> Result<()> {
            use cuda_core::memcpy_dtod_async;
            let err = |e| Error::Cuda(format!("cuda cat copy failed: {e}"));
            let stream = &self.runtime.stream;
            let shift_bytes = |dptr: cuda_core::sys::CUdeviceptr, elems: usize, bytes: usize| {
                dptr + (elems * bytes) as u64
            };
            match (&src.inner, &self.inner) {
                (CudaInner::F16(s), CudaInner::F16(d)) => unsafe {
                    let dst =
                        shift_bytes(d.device_pointer().cu_deviceptr(), offset, size_of::<f16>());
                    memcpy_dtod_async::<f16>(
                        dst,
                        s.device_pointer().cu_deviceptr(),
                        s.size(),
                        stream,
                    )
                    .map_err(err)?;
                },
                (CudaInner::BF16(s), CudaInner::BF16(d)) => unsafe {
                    let dst =
                        shift_bytes(d.device_pointer().cu_deviceptr(), offset, size_of::<bf16>());
                    memcpy_dtod_async::<bf16>(
                        dst,
                        s.device_pointer().cu_deviceptr(),
                        s.size(),
                        stream,
                    )
                    .map_err(err)?;
                },
                (CudaInner::F32(s), CudaInner::F32(d)) => unsafe {
                    let dst =
                        shift_bytes(d.device_pointer().cu_deviceptr(), offset, size_of::<f32>());
                    memcpy_dtod_async::<f32>(
                        dst,
                        s.device_pointer().cu_deviceptr(),
                        s.size(),
                        stream,
                    )
                    .map_err(err)?;
                },
                (CudaInner::I64(s), CudaInner::I64(d)) => unsafe {
                    let dst =
                        shift_bytes(d.device_pointer().cu_deviceptr(), offset, size_of::<i64>());
                    memcpy_dtod_async::<i64>(
                        dst,
                        s.device_pointer().cu_deviceptr(),
                        s.size(),
                        stream,
                    )
                    .map_err(err)?;
                },
                _ => return Err(Error::DTypeMismatch("cat: mixed dtypes".into())),
            }
            // Single-stream cudarc ownership: the queued copy precedes the
            // stream-ordered free, so no host wait is needed.
            Ok(())
        }
    }

    fn log_softmax_tensor<E: cuda_core::DType + DeviceRepr>(
        rt: &Arc<Runtime>,
        src: &Arc<Tensor<E>>,
        outer_size: usize,
        inner_size: usize,
        block: usize,
    ) -> Result<Arc<Tensor<E>>> {
        fn err(e: impl std::fmt::Display) -> Error {
            Error::Cuda(format!("cuTile log_softmax failed: {e}"))
        }
        let n = i32::try_from(inner_size).map_err(err)?;
        let b = i32::try_from(block).map_err(err)?;
        let x = src.reshape(&[outer_size, inner_size]).map_err(err)?;
        if outer_size == 0 {
            return Ok(Arc::new(alloc_foreign(rt, 0)?));
        }
        // Every row and column is written by the kernel before this output is read.
        let out = alloc_foreign(rt, outer_size * inner_size)?;
        let out = out.reshape(&[outer_size, inner_size]).map_err(err)?;
        let generics = vec![E::DTYPE.as_str().to_string(), n.to_string(), b.to_string()];
        let (_x, out) = kernels::log_softmax_fwd(x, out.partition([1, inner_size]))
            .generics(generics)
            .async_profiled(rt)
            .map_err(err)?;
        Ok(Arc::new(out.unpartition().reshape(&[outer_size * inner_size]).map_err(err)?))
    }

    fn rms_norm_tensor<E: cuda_core::DType + DeviceRepr + ValidAsZeroBits>(
        rt: &Arc<Runtime>,
        src: &Arc<Tensor<E>>,
        weight: Option<&Arc<Tensor<E>>>,
        outer_size: usize,
        inner_size: usize,
        eps: f32,
        block: usize,
    ) -> Result<Arc<Tensor<E>>> {
        fn err(e: impl std::fmt::Display) -> Error {
            Error::Cuda(format!("cuTile rms_norm failed: {e}"))
        }
        let n = i32::try_from(inner_size).map_err(err)?;
        let b = i32::try_from(block).map_err(err)?;
        let x = src.reshape(&[outer_size, inner_size]).map_err(err)?;
        let w = match weight {
            Some(w) => w.reshape(&[inner_size]).map_err(err)?,
            None => Arc::new(alloc_foreign_zeros(rt, inner_size)?),
        };
        let weighted = i32::from(weight.is_some());
        // The kernel writes each element of every row before the output is read.
        let out = alloc_foreign(rt, outer_size * inner_size)?;
        let out = out.reshape(&[outer_size, inner_size]).map_err(err)?;
        let (_x, _w, out, _, _) =
            kernels::rms_norm(x, w, out.partition([1, inner_size]), eps, weighted)
                .generics(vec![E::DTYPE.as_str().to_string(), n.to_string(), b.to_string()])
                .async_profiled(rt)
                .map_err(err)?;
        Ok(Arc::new(out.unpartition().reshape(&[outer_size * inner_size]).map_err(err)?))
    }

    fn log_softmax_storage(
        storage: &CudaStorage,
        layout: &Layout,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<CudaStorage> {
        let compact = storage.compact(layout)?;
        let rt = compact.runtime.clone();
        let block = [1024usize, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            .into_iter()
            .find(|&b| inner_size.is_multiple_of(b))
            .unwrap();
        let inner = match &compact.inner {
            CudaInner::F16(t) => {
                CudaInner::F16(log_softmax_tensor(&rt, t, outer_size, inner_size, block)?)
            }
            CudaInner::BF16(t) => {
                CudaInner::BF16(log_softmax_tensor(&rt, t, outer_size, inner_size, block)?)
            }
            CudaInner::F32(t) => {
                CudaInner::F32(log_softmax_tensor(&rt, t, outer_size, inner_size, block)?)
            }
            CudaInner::I64(_) => {
                return Err(Error::DTypeMismatch("log_softmax requires floats".into()));
            }
        };
        Ok(CudaStorage { inner, runtime: rt })
    }

    fn launch_err(e: impl std::fmt::Display) -> Error {
        Error::Cuda(format!("cuTile launch failed: {e}"))
    }

    /// Allocates an output tensor and runs `launch` against its `TILE`-wide
    /// partition. Each caller must write every valid output element before returning
    /// the partition. Empty inputs skip the launch entirely.
    /// Casts a foreign tensor to another dtype via the `cast` kernel.
    fn cast_tensor<F, T>(rt: &Arc<Runtime>, src: &Arc<Tensor<F>>) -> Result<Arc<Tensor<T>>>
    where
        F: cuda_core::DType + DeviceRepr,
        T: cuda_core::DType + DeviceRepr,
    {
        let len = src.size();
        if len == 0 {
            return Ok(Arc::new(alloc_foreign(rt, 0)?));
        }
        let out = alloc_foreign(rt, len)?;
        let (o, _) = kernels::cast(out.partition([TILE]), src.clone())
            .generics(vec![
                T::DTYPE.as_str().to_string(),
                F::DTYPE.as_str().to_string(),
                TILE.to_string(),
            ])
            .async_profiled(rt)
            .map_err(launch_err)?;
        Ok(Arc::new(o.unpartition()))
    }

    /// Device-side ones without host synchronization (hot-path safe).
    fn ones_foreign<T>(rt: &Arc<Runtime>, size: usize) -> Result<Tensor<T>>
    where
        T: cuda_core::DType + DeviceRepr,
    {
        if size == 0 {
            return alloc_foreign(rt, 0);
        }
        let out: Tensor<T> = alloc_foreign(rt, size)?;
        let (o,) = kernels::fill_one(out.partition([TILE]))
            .generics(vec![T::DTYPE.as_str().to_string(), TILE.to_string()])
            .async_profiled(rt)
            .map_err(launch_err)?;
        Ok(o.unpartition())
    }

    fn launch_1d<T>(
        rt: &Arc<Runtime>,
        len: usize,
        launch: impl FnOnce(Partition<Tensor<T>>) -> Result<Partition<Tensor<T>>>,
    ) -> Result<Arc<Tensor<T>>>
    where
        T: cuda_core::DType + DeviceRepr,
    {
        if len == 0 {
            return Ok(Arc::new(alloc_foreign(rt, 0)?));
        }
        let out: Tensor<T> = alloc_foreign(rt, len)?;
        // Every valid lane is overwritten by the launched elementwise kernel.
        Ok(Arc::new(launch(out.partition([TILE]))?.unpartition()))
    }

    impl BackendStorage for CudaStorage {
        fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
            let compact = self.compact(l)?;
            let rt = compact.runtime.clone();
            let inner = match &compact.inner {
                CudaInner::F16(src) => CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                    let (o, _, _) = kernels::scalar_pow(part, src.clone(), f16::from_f32(e as f32))
                        .generics(vec!["f16".into(), TILE.to_string()])
                        .async_profiled(&rt)
                        .map_err(launch_err)?;
                    Ok(o)
                })?),
                CudaInner::BF16(src) => CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                    let (o, _, _) =
                        kernels::scalar_pow(part, src.clone(), bf16::from_f32(e as f32))
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                    Ok(o)
                })?),
                CudaInner::F32(src) => CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                    let (o, _, _) = kernels::scalar_pow(part, src.clone(), e as f32)
                        .generics(vec!["f32".into(), TILE.to_string()])
                        .async_profiled(&rt)
                        .map_err(launch_err)?;
                    Ok(o)
                })?),
                CudaInner::I64(_) => {
                    return Err(Error::NotImplemented(
                        "cuda scalar powf for i64 is not implemented",
                    ));
                }
            };
            Ok(Self { inner, runtime: rt })
        }
        fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
            let compact = self.compact(l)?;
            let rt = compact.runtime.clone();
            let inner = match (&compact.inner, O::KERNEL) {
                (CudaInner::F16(src), "Neg") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::neg(part, src.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "Neg") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::neg(part, src.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "Neg") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::neg(part, src.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "exp") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::kexp(part, src.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "exp") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::kexp(part, src.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "exp") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::kexp(part, src.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "log") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::klog(part, src.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "log") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::klog(part, src.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "log") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::klog(part, src.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "sin") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::ksin(part, src.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "sin") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::ksin(part, src.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "sin") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::ksin(part, src.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "cos") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::kcos(part, src.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "cos") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::kcos(part, src.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "cos") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::kcos(part, src.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "tanh") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::ktanh(part, src.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "tanh") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::ktanh(part, src.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "tanh") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::ktanh(part, src.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "relu") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::relu(part, src.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "relu") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::relu(part, src.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "relu") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::relu(part, src.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "relu_backward") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::relu_backward(part, src.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "relu_backward") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::relu_backward(part, src.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "relu_backward") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _) = kernels::relu_backward(part, src.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "scalar_add") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _, _) =
                            kernels::scalar_add(part, src.clone(), f16::from_f32(op.f32(0.0)))
                                .generics(vec!["f16".into(), TILE.to_string()])
                                .async_profiled(&rt)
                                .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "scalar_add") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _, _) =
                            kernels::scalar_add(part, src.clone(), bf16::from_f32(op.f32(0.0)))
                                .generics(vec!["bf16".into(), TILE.to_string()])
                                .async_profiled(&rt)
                                .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "scalar_add") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _, _) = kernels::scalar_add(part, src.clone(), op.f32(0.0))
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "scalar_mul") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _, _) =
                            kernels::scalar_mul(part, src.clone(), f16::from_f32(op.f32(1.0)))
                                .generics(vec!["f16".into(), TILE.to_string()])
                                .async_profiled(&rt)
                                .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "scalar_mul") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _, _) =
                            kernels::scalar_mul(part, src.clone(), bf16::from_f32(op.f32(1.0)))
                                .generics(vec!["bf16".into(), TILE.to_string()])
                                .async_profiled(&rt)
                                .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "scalar_mul") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _, _) = kernels::scalar_mul(part, src.clone(), op.f32(1.0))
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(src), "scalar_div") => {
                    CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                        let (o, _, _) = kernels::scalar_div(
                            part,
                            src.clone(),
                            f16::from_f32(1.0 / op.f32(1.0)),
                        )
                        .generics(vec!["f16".into(), TILE.to_string()])
                        .async_profiled(&rt)
                        .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(src), "scalar_div") => {
                    CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                        let (o, _, _) = kernels::scalar_div(
                            part,
                            src.clone(),
                            bf16::from_f32(1.0 / op.f32(1.0)),
                        )
                        .generics(vec!["bf16".into(), TILE.to_string()])
                        .async_profiled(&rt)
                        .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(src), "scalar_div") => {
                    CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                        let (o, _, _) = kernels::scalar_div(part, src.clone(), 1.0 / op.f32(1.0))
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::I64(_), _) => {
                    return Err(Error::NotImplemented(
                        "cuda unary ops for i64 are not implemented",
                    ));
                }
                _ => return Err(Error::NotImplemented("cuda unary op is not implemented")),
            };
            Ok(Self { inner, runtime: rt })
        }
        fn binary_op<O: BinaryOp>(
            &self,
            layout: &Layout,
            other: &Self,
            layout_other: &Layout,
        ) -> Result<Self> {
            let lhs = self.compact(layout)?;
            let rhs = other.compact(layout_other)?;
            let rt = lhs.runtime.clone();
            let inner = match (&lhs.inner, &rhs.inner, O::KERNEL) {
                (CudaInner::F16(a), CudaInner::F16(b), "add") => {
                    CudaInner::F16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::add(part, a.clone(), b.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "add") => {
                    CudaInner::BF16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::add(part, a.clone(), b.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "add") => {
                    CudaInner::F32(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::add(part, a.clone(), b.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "sub") => {
                    CudaInner::F16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::sub(part, a.clone(), b.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "sub") => {
                    CudaInner::BF16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::sub(part, a.clone(), b.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "sub") => {
                    CudaInner::F32(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::sub(part, a.clone(), b.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "mul") => {
                    CudaInner::F16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::mul(part, a.clone(), b.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "mul") => {
                    CudaInner::BF16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::mul(part, a.clone(), b.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "mul") => {
                    CudaInner::F32(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::mul(part, a.clone(), b.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "div") => {
                    CudaInner::F16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::div(part, a.clone(), b.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "div") => {
                    CudaInner::BF16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::div(part, a.clone(), b.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "div") => {
                    CudaInner::F32(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::div(part, a.clone(), b.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "powf") => {
                    CudaInner::F16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::kpow(part, a.clone(), b.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "powf") => {
                    CudaInner::BF16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::kpow(part, a.clone(), b.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "powf") => {
                    CudaInner::F32(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::kpow(part, a.clone(), b.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "eq") => {
                    CudaInner::F16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::eq(part, a.clone(), b.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "eq") => {
                    CudaInner::BF16(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::eq(part, a.clone(), b.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "eq") => {
                    CudaInner::F32(launch_1d(&rt, a.size(), |part| {
                        let (o, _, _) = kernels::eq(part, a.clone(), b.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                _ => {
                    return Err(Error::NotImplemented(
                        "cuda binary op is not implemented for this dtype",
                    ));
                }
            };
            Ok(Self { inner, runtime: rt })
        }
        fn ne_scalar(&self, layout: &Layout, scalar: f64) -> Result<Self> {
            let compact = self.compact(layout)?;
            let rt = compact.runtime.clone();
            let inner = match &compact.inner {
                CudaInner::F16(src) => CudaInner::F16(launch_1d(&rt, src.size(), |part| {
                    let (o, _, _) =
                        kernels::ne_scalar(part, src.clone(), f16::from_f32(scalar as f32))
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                    Ok(o)
                })?),
                CudaInner::BF16(src) => CudaInner::BF16(launch_1d(&rt, src.size(), |part| {
                    let (o, _, _) =
                        kernels::ne_scalar(part, src.clone(), bf16::from_f32(scalar as f32))
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                    Ok(o)
                })?),
                CudaInner::F32(src) => CudaInner::F32(launch_1d(&rt, src.size(), |part| {
                    let (o, _, _) = kernels::ne_scalar(part, src.clone(), scalar as f32)
                        .generics(vec!["f32".into(), TILE.to_string()])
                        .async_profiled(&rt)
                        .map_err(launch_err)?;
                    Ok(o)
                })?),
                CudaInner::I64(_) => {
                    return Err(Error::DTypeMismatch("ne_scalar requires float dtype".into()));
                }
            };
            Ok(Self { inner, runtime: rt })
        }
        fn ne_scalar_i64(&self, layout: &Layout, scalar: i64) -> Result<Self> {
            let compact = self.compact(layout)?;
            let rt = compact.runtime.clone();
            let err = |e| Error::Cuda(format!("cuTile launch failed: {e}"));
            let inner = match &compact.inner {
                CudaInner::I64(src) => {
                    let len = src.size();
                    let out_owned: Tensor<i64> = alloc_foreign_zeros(&rt, len)?;
                    let part = out_owned.partition([TILE]);
                    let (out_part, _, _) = kernels::ne_scalar_i64(part, src.clone(), scalar)
                        .generics(vec![TILE.to_string()])
                        .async_profiled(&rt)
                        .map_err(err)?;
                    CudaInner::I64(Arc::new(out_part.unpartition()))
                }
                _ => return Err(Error::DTypeMismatch("ne_scalar_i64 requires i64 dtype".into())),
            };
            Ok(Self { inner, runtime: rt })
        }
        fn select(
            &self,
            cond_layout: &Layout,
            on_true: &Self,
            true_layout: &Layout,
            on_false: &Self,
            false_layout: &Layout,
        ) -> Result<Self> {
            let cond = self.compact(cond_layout)?;
            let t = on_true.compact(true_layout)?;
            let f = on_false.compact(false_layout)?;
            let rt = cond.runtime.clone();
            let inner = match (&cond.inner, &t.inner, &f.inner) {
                (CudaInner::F16(c), CudaInner::F16(t), CudaInner::F16(f)) => {
                    CudaInner::F16(launch_1d(&rt, c.size(), |part| {
                        let (o, _, _, _) = kernels::where_op(part, c.clone(), t.clone(), f.clone())
                            .generics(vec!["f16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::BF16(c), CudaInner::BF16(t), CudaInner::BF16(f)) => {
                    CudaInner::BF16(launch_1d(&rt, c.size(), |part| {
                        let (o, _, _, _) = kernels::where_op(part, c.clone(), t.clone(), f.clone())
                            .generics(vec!["bf16".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::F32(c), CudaInner::F32(t), CudaInner::F32(f)) => {
                    CudaInner::F32(launch_1d(&rt, c.size(), |part| {
                        let (o, _, _, _) = kernels::where_op(part, c.clone(), t.clone(), f.clone())
                            .generics(vec!["f32".into(), TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(launch_err)?;
                        Ok(o)
                    })?)
                }
                (CudaInner::I64(c), CudaInner::I64(t), CudaInner::I64(f)) => {
                    let len = c.size();
                    let out_owned: Tensor<i64> = alloc_foreign_zeros(&rt, len)
                        .map_err(|e| Error::Cuda(format!("cuTile launch failed: {e}")))?;
                    let part = out_owned.partition([TILE]);
                    let (out_part, _, _, _) =
                        kernels::where_i64(part, c.clone(), t.clone(), f.clone())
                            .generics(vec![TILE.to_string()])
                            .async_profiled(&rt)
                            .map_err(|e| Error::Cuda(format!("cuTile launch failed: {e}")))?;
                    CudaInner::I64(Arc::new(out_part.unpartition()))
                }
                _ => {
                    return Err(Error::NotImplemented(
                        "cuda select is not implemented for this dtype",
                    ));
                }
            };
            Ok(Self { inner, runtime: rt })
        }
        fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
            fn err(e: impl std::fmt::Display) -> Error {
                Error::Cuda(format!("cuTile reduce failed: {e}"))
            }
            let compact = self.compact(layout)?;
            let rt = compact.runtime.clone();
            let outer_size = dst.len();
            let reduce_size = compact.len() / outer_size;
            let block = [128usize, 64, 32, 16, 8, 4, 2, 1]
                .into_iter()
                .find(|&b| reduce_size.is_multiple_of(b))
                .unwrap();
            let n = i32::try_from(reduce_size).map_err(err)?;
            let b = i32::try_from(block).map_err(err)?;
            if compact.dtype() == DType::I64 {
                return Err(Error::NotImplemented("cuda reduce for i64 is not implemented"));
            }
            let sum = O::KERNEL == "reduce_sum";
            if !sum && O::KERNEL != "reduce_max" {
                return Err(Error::NotImplemented("cuda reduction is not implemented"));
            }
            let src_f32 = compact.to_f32_storage()?;
            let dtype = compact.dtype();
            let x2 = src_f32.reshape(&[outer_size, reduce_size]).map_err(err)?;
            let out_owned: Tensor<f32> = alloc_foreign_zeros(&rt, outer_size)?;
            let generics = vec![n.to_string(), b.to_string()];
            let out_f32 = if sum {
                let (_x, _o) = kernels::reduce_sum_f32(x2, out_owned.partition([1]))
                    .generics(generics)
                    .async_profiled(&rt)
                    .map_err(err)?;
                Arc::new(_o.unpartition())
            } else {
                let (_x, _o) = kernels::reduce_max_f32(x2, out_owned.partition([1]))
                    .generics(generics)
                    .async_profiled(&rt)
                    .map_err(err)?;
                Arc::new(_o.unpartition())
            };
            let reduced = Self::from_f32_storage(&rt, out_f32, dtype)?;
            *dst = Self { inner: reduced, runtime: rt };
            Ok(())
        }
        fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
            fn err(e: impl std::fmt::Display) -> Error {
                Error::Cuda(format!("cuTile matmul failed: {e}"))
            }
            let ndim = layout.ndim();
            let m = layout.shape()[ndim - 2];
            let k = layout.shape()[ndim - 1];
            let n = layout_other.shape()[ndim - 1];
            let batch: usize =
                if ndim > 2 { layout.shape().iter().take(ndim - 2).product() } else { 1 };
            if m == 0 || n == 0 || batch == 0 {
                return Self::uninit(0, self.dtype());
            }
            if k == 0 {
                return Self::uninit(batch * m * n, self.dtype());
            }
            let lhs_params = GemmOperand::from_layout(layout, m, k);
            let rhs_params = GemmOperand::from_layout(layout_other, k, n);
            let lhs_compact = if lhs_params.is_none() { Some(self.compact(layout)?) } else { None };
            let rhs_compact =
                if rhs_params.is_none() { Some(other.compact(layout_other)?) } else { None };
            let lhs = lhs_compact.as_ref().unwrap_or(self);
            let rhs = rhs_compact.as_ref().unwrap_or(other);
            let lhs_params = lhs_params.unwrap_or_else(|| GemmOperand::dense(m, k));
            let rhs_params = rhs_params.unwrap_or_else(|| GemmOperand::dense(k, n));
            let rt = lhs.runtime.clone();
            let inner = match (&lhs.inner, &rhs.inner) {
                (CudaInner::F32(a), CudaInner::F32(b)) => {
                    CudaInner::F32(blas_gemm(&rt, a, b, lhs_params, rhs_params, m, n, k, batch)?)
                }
                (CudaInner::F16(a), CudaInner::F16(b)) => {
                    CudaInner::F16(blas_gemm(&rt, a, b, lhs_params, rhs_params, m, n, k, batch)?)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b)) => {
                    CudaInner::BF16(blas_gemm(&rt, a, b, lhs_params, rhs_params, m, n, k, batch)?)
                }
                _ => return Err(Error::DTypeMismatch("matmul dtype mismatch".into())),
            };
            Ok(Self { inner, runtime: rt })
        }
        fn gather(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
        ) -> Result<Self> {
            fn err(e: impl std::fmt::Display) -> Error {
                Error::Cuda(format!("cuTile gather failed: {e}"))
            }
            let src = self.compact(layout)?;
            let idx = indices.compact(indices_layout)?;
            let rt = src.runtime.clone();
            let src_dim = layout.shape()[dim];
            let dst_dim = indices_layout.shape()[dim];
            let right_len: usize = layout.shape().iter().skip(dim + 1).product();
            let len = indices_layout.size();
            if len == 0 {
                return Self::uninit(0, src.dtype());
            }
            let idx_i64: &Arc<Tensor<i64>> = match &idx.inner {
                CudaInner::I64(t) => t,
                _ => {
                    return Err(Error::DTypeMismatch(
                        "gather requires floating source and i64 indices".into(),
                    ));
                }
            };
            let len_i32 = i32::try_from(len).map_err(err)?;
            let dst_i32 = i32::try_from(dst_dim).map_err(err)?;
            let right_i32 = i32::try_from(right_len).map_err(err)?;
            let src_i32 = i32::try_from(src_dim).map_err(err)?;
            let inner = match &src.inner {
                CudaInner::F32(s) => {
                    let out_owned: Tensor<f32> = alloc_foreign_zeros(&rt, len)?;
                    let (_idx, _src_ptr, _out, _, _, _, _) = unsafe {
                        kernels::gather(
                            idx_i64.clone(),
                            s.device_pointer(),
                            out_owned.partition([GATHER_TILE]),
                            len_i32,
                            dst_i32,
                            right_i32,
                            src_i32,
                        )
                    }
                    .generics(vec!["f32".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::F32(Arc::new(_out.unpartition()))
                }
                CudaInner::F16(s) => {
                    let out_owned: Tensor<f16> = alloc_foreign_zeros(&rt, len)?;
                    let (_idx, _src_ptr, _out, _, _, _, _) = unsafe {
                        kernels::gather(
                            idx_i64.clone(),
                            s.device_pointer(),
                            out_owned.partition([GATHER_TILE]),
                            len_i32,
                            dst_i32,
                            right_i32,
                            src_i32,
                        )
                    }
                    .generics(vec!["f16".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::F16(Arc::new(_out.unpartition()))
                }
                CudaInner::BF16(s) => {
                    let out_owned: Tensor<bf16> = alloc_foreign_zeros(&rt, len)?;
                    let (_idx, _src_ptr, _out, _, _, _, _) = unsafe {
                        kernels::gather(
                            idx_i64.clone(),
                            s.device_pointer(),
                            out_owned.partition([GATHER_TILE]),
                            len_i32,
                            dst_i32,
                            right_i32,
                            src_i32,
                        )
                    }
                    .generics(vec!["bf16".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::BF16(Arc::new(_out.unpartition()))
                }
                _ => {
                    return Err(Error::DTypeMismatch(
                        "gather requires floating source and i64 indices".into(),
                    ));
                }
            };
            Ok(Self { inner, runtime: rt })
        }
        fn scatter_add(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
            dst_shape: &[usize],
        ) -> Result<Self> {
            fn err(e: impl std::fmt::Display) -> Error {
                Error::Cuda(format!("cuTile scatter_add failed: {e}"))
            }
            let src = self.compact(layout)?;
            let idx = indices.compact(indices_layout)?;
            let rt = src.runtime.clone();
            let dst_dim = dst_shape[dim];
            let index_len = indices_layout.shape()[dim];
            let right_len: usize = dst_shape[dim + 1..].iter().product();
            let len = src.len();
            if len == 0 {
                return Self::uninit(0, src.dtype());
            }
            let idx_i64: &Arc<Tensor<i64>> = match &idx.inner {
                CudaInner::I64(t) => t,
                _ => {
                    return Err(Error::DTypeMismatch(
                        "scatter_add requires floating source and i64 indices".into(),
                    ));
                }
            };
            let len_i32 = i32::try_from(len).map_err(err)?;
            let index_i32 = i32::try_from(index_len).map_err(err)?;
            let right_i32 = i32::try_from(right_len).map_err(err)?;
            let dst_i32 = i32::try_from(dst_dim).map_err(err)?;
            let blocks = len.div_ceil(128) as u32;
            let inner = match &src.inner {
                CudaInner::F32(s) => {
                    let out_owned: Tensor<f32> =
                        alloc_foreign_zeros(&rt, dst_shape.iter().product()).map_err(err)?;
                    unsafe {
                        kernels::scatter_add(
                            s.clone(),
                            idx_i64.clone(),
                            out_owned.device_pointer(),
                            len_i32,
                            index_i32,
                            right_i32,
                            dst_i32,
                        )
                    }
                    .grid((blocks, 1, 1))
                    .generics(vec!["f32".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::F32(Arc::new(out_owned))
                }
                CudaInner::F16(s) => {
                    let out_owned: Tensor<f16> =
                        alloc_foreign_zeros(&rt, dst_shape.iter().product()).map_err(err)?;
                    unsafe {
                        kernels::scatter_add(
                            s.clone(),
                            idx_i64.clone(),
                            out_owned.device_pointer(),
                            len_i32,
                            index_i32,
                            right_i32,
                            dst_i32,
                        )
                    }
                    .grid((blocks, 1, 1))
                    .generics(vec!["f16".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::F16(Arc::new(out_owned))
                }
                CudaInner::BF16(s) => {
                    let out_owned: Tensor<bf16> =
                        alloc_foreign_zeros(&rt, dst_shape.iter().product()).map_err(err)?;
                    unsafe {
                        kernels::scatter_add(
                            s.clone(),
                            idx_i64.clone(),
                            out_owned.device_pointer(),
                            len_i32,
                            index_i32,
                            right_i32,
                            dst_i32,
                        )
                    }
                    .grid((blocks, 1, 1))
                    .generics(vec!["bf16".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::BF16(Arc::new(out_owned))
                }
                _ => {
                    return Err(Error::DTypeMismatch(
                        "scatter_add requires floating source and i64 indices".into(),
                    ));
                }
            };
            Ok(Self { inner, runtime: rt })
        }
        fn index_select(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
        ) -> Result<Self> {
            fn err(e: impl std::fmt::Display) -> Error {
                Error::Cuda(format!("cuTile index_select failed: {e}"))
            }
            let src = self.compact(layout)?;
            let idx = indices.compact(indices_layout)?;
            let rt = src.runtime.clone();
            let left_len: usize = layout.shape().iter().take(dim).product();
            let index_len = indices_layout.shape()[0];
            let src_dim = layout.shape()[dim];
            let right_len: usize = layout.shape().iter().skip(dim + 1).product();
            let len = left_len * index_len * right_len;
            if len == 0 {
                return Self::uninit(0, src.dtype());
            }
            let idx_i64: &Arc<Tensor<i64>> = match &idx.inner {
                CudaInner::I64(t) => t,
                _ => {
                    return Err(Error::DTypeMismatch(
                        "index_select requires floating source and i64 indices".into(),
                    ));
                }
            };
            let len_i32 = i32::try_from(len).map_err(err)?;
            let index_i32 = i32::try_from(index_len).map_err(err)?;
            let right_i32 = i32::try_from(right_len).map_err(err)?;
            let src_i32 = i32::try_from(src_dim).map_err(err)?;
            let inner = match &src.inner {
                CudaInner::F32(s) => {
                    let out_owned: Tensor<f32> = alloc_foreign_zeros(&rt, len)?;
                    let (_, _, _out, _, _, _, _) = unsafe {
                        kernels::index_select(
                            idx_i64.device_pointer(),
                            s.device_pointer(),
                            out_owned.partition([GATHER_TILE]),
                            len_i32,
                            index_i32,
                            right_i32,
                            src_i32,
                        )
                    }
                    .generics(vec!["f32".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::F32(Arc::new(_out.unpartition()))
                }
                CudaInner::F16(s) => {
                    let out_owned: Tensor<f16> = alloc_foreign_zeros(&rt, len)?;
                    let (_, _, _out, _, _, _, _) = unsafe {
                        kernels::index_select(
                            idx_i64.device_pointer(),
                            s.device_pointer(),
                            out_owned.partition([GATHER_TILE]),
                            len_i32,
                            index_i32,
                            right_i32,
                            src_i32,
                        )
                    }
                    .generics(vec!["f16".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::F16(Arc::new(_out.unpartition()))
                }
                CudaInner::BF16(s) => {
                    let out_owned: Tensor<bf16> = alloc_foreign_zeros(&rt, len)?;
                    let (_, _, _out, _, _, _, _) = unsafe {
                        kernels::index_select(
                            idx_i64.device_pointer(),
                            s.device_pointer(),
                            out_owned.partition([GATHER_TILE]),
                            len_i32,
                            index_i32,
                            right_i32,
                            src_i32,
                        )
                    }
                    .generics(vec!["bf16".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::BF16(Arc::new(_out.unpartition()))
                }
                _ => {
                    return Err(Error::DTypeMismatch(
                        "index_select requires floating source and i64 indices".into(),
                    ));
                }
            };
            Ok(Self { inner, runtime: rt })
        }
        fn index_add(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
            dst_shape: &[usize],
        ) -> Result<Self> {
            fn err(e: impl std::fmt::Display) -> Error {
                Error::Cuda(format!("cuTile index_add failed: {e}"))
            }
            let src = self.compact(layout)?;
            let idx = indices.compact(indices_layout)?;
            let rt = src.runtime.clone();
            let src_dim = layout.shape()[dim];
            let dst_dim = dst_shape[dim];
            let right_len: usize = dst_shape[dim + 1..].iter().product();
            let len = src.len();
            if len == 0 {
                return Self::uninit(dst_shape.iter().product(), src.dtype());
            }
            let idx_i64: &Arc<Tensor<i64>> = match &idx.inner {
                CudaInner::I64(t) => t,
                _ => {
                    return Err(Error::DTypeMismatch(
                        "index_add requires floating source and i64 indices".into(),
                    ));
                }
            };
            let len_i32 = i32::try_from(len).map_err(err)?;
            let src_i32 = i32::try_from(src_dim).map_err(err)?;
            let right_i32 = i32::try_from(right_len).map_err(err)?;
            let dst_i32 = i32::try_from(dst_dim).map_err(err)?;
            let blocks = len.div_ceil(128) as u32;
            let inner = match &src.inner {
                CudaInner::F32(s) => {
                    let out_owned: Tensor<f32> =
                        alloc_foreign_zeros(&rt, dst_shape.iter().product()).map_err(err)?;
                    unsafe {
                        kernels::index_add(
                            idx_i64.device_pointer(),
                            s.clone(),
                            out_owned.device_pointer(),
                            len_i32,
                            src_i32,
                            right_i32,
                            dst_i32,
                        )
                    }
                    .grid((blocks, 1, 1))
                    .generics(vec!["f32".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::F32(Arc::new(out_owned))
                }
                CudaInner::F16(s) => {
                    let out_owned: Tensor<f16> =
                        alloc_foreign_zeros(&rt, dst_shape.iter().product()).map_err(err)?;
                    unsafe {
                        kernels::index_add(
                            idx_i64.device_pointer(),
                            s.clone(),
                            out_owned.device_pointer(),
                            len_i32,
                            src_i32,
                            right_i32,
                            dst_i32,
                        )
                    }
                    .grid((blocks, 1, 1))
                    .generics(vec!["f16".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::F16(Arc::new(out_owned))
                }
                CudaInner::BF16(s) => {
                    let out_owned: Tensor<bf16> =
                        alloc_foreign_zeros(&rt, dst_shape.iter().product()).map_err(err)?;
                    unsafe {
                        kernels::index_add(
                            idx_i64.device_pointer(),
                            s.clone(),
                            out_owned.device_pointer(),
                            len_i32,
                            src_i32,
                            right_i32,
                            dst_i32,
                        )
                    }
                    .grid((blocks, 1, 1))
                    .generics(vec!["bf16".to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
                    CudaInner::BF16(Arc::new(out_owned))
                }
                _ => {
                    return Err(Error::DTypeMismatch(
                        "index_add requires floating source and i64 indices".into(),
                    ));
                }
            };
            Ok(Self { inner, runtime: rt })
        }
        /// Runs a row-wise f32 kernel, converting half-precision inputs
        /// through f32 exactly like the old CUDA kernels did.
        fn log_sum_exp(
            &self,
            layout: &Layout,
            outer_size: usize,
            reduce_size: usize,
        ) -> Result<Self> {
            fn err(e: impl std::fmt::Display) -> Error {
                Error::Cuda(format!("cuTile log_sum_exp failed: {e}"))
            }
            let compact = self.compact(layout)?;
            let rt = compact.runtime.clone();
            let dtype = compact.dtype();
            if compact.dtype() == DType::I64 {
                return Err(Error::NotImplemented("cuda log_sum_exp for i64 is not implemented"));
            }
            let src_f32 = compact.to_f32_storage()?;
            let block = [128usize, 64, 32, 16, 8, 4, 2, 1]
                .into_iter()
                .find(|&b| reduce_size.is_multiple_of(b))
                .unwrap();
            let n = i32::try_from(reduce_size).map_err(err)?;
            let b = i32::try_from(block).map_err(err)?;
            let x2 = src_f32.reshape(&[outer_size, reduce_size]).map_err(err)?;
            let out_owned: Tensor<f32> = alloc_foreign_zeros(&rt, outer_size)?;
            let (_x, _o) = kernels::log_sum_exp_f32(x2, out_owned.partition([1]))
                .generics(vec![n.to_string(), b.to_string()])
                .async_profiled(&rt)
                .map_err(err)?;
            let out_f32 = Arc::new(_o.unpartition());
            let inner = Self::from_f32_storage(&rt, out_f32, dtype)?;
            Ok(Self { inner, runtime: rt })
        }
        fn log_softmax_fwd(
            &self,
            layout: &Layout,
            outer_size: usize,
            inner_size: usize,
        ) -> Result<Self> {
            log_softmax_storage(self, layout, outer_size, inner_size)
        }
        fn log_softmax_bwd(
            &self,
            grad_layout: &Layout,
            lsm: &Self,
            lsm_layout: &Layout,
            outer_size: usize,
            inner_size: usize,
        ) -> Result<Self> {
            fn err(e: impl std::fmt::Display) -> Error {
                Error::Cuda(format!("cuTile log_softmax failed: {e}"))
            }
            let grad = self.compact(grad_layout)?;
            let lsm_c = lsm.compact(lsm_layout)?;
            let rt = grad.runtime.clone();
            let dtype = grad.dtype();
            if lsm_c.dtype() != dtype {
                return Err(Error::DTypeMismatch(
                    "log_softmax_bwd: dtype mismatch between grad and lsm".into(),
                ));
            }
            if dtype == DType::I64 {
                return Err(Error::NotImplemented(
                    "cuda log_softmax_bwd for i64 is not implemented",
                ));
            }
            let grad_f32 = grad.to_f32_storage()?;
            let lsm_f32 = lsm_c.to_f32_storage()?;
            let block = [128usize, 64, 32, 16, 8, 4, 2, 1]
                .into_iter()
                .find(|&b| inner_size.is_multiple_of(b))
                .unwrap();
            let n = i32::try_from(inner_size).map_err(err)?;
            let b = i32::try_from(block).map_err(err)?;
            let cols = n;
            let g2 = grad_f32.reshape(&[outer_size, inner_size]).map_err(err)?;
            let l2 = lsm_f32.reshape(&[outer_size, inner_size]).map_err(err)?;
            let out_owned: Tensor<f32> = alloc_foreign_zeros(&rt, outer_size * inner_size)?;
            let o2 = out_owned.reshape(&[outer_size, inner_size]).map_err(err)?;
            let (_g, _l, _o) =
                kernels::log_softmax_bwd_f32(g2, l2, o2.partition([1, cols as usize]))
                    .generics(vec![n.to_string(), b.to_string()])
                    .async_profiled(&rt)
                    .map_err(err)?;
            let flat = _o.unpartition().reshape(&[outer_size * inner_size]).map_err(err)?;
            let inner = match dtype {
                DType::F32 => CudaInner::F32(Arc::new(flat)),
                DType::F16 => CudaInner::F16(cast_tensor(&rt, &Arc::new(flat)).map_err(err)?),
                DType::BF16 => CudaInner::BF16(cast_tensor(&rt, &Arc::new(flat)).map_err(err)?),
                DType::I64 => {
                    return Err(Error::NotImplemented(
                        "cuda log_softmax_bwd for i64 is not implemented",
                    ));
                }
            };
            Ok(Self { inner, runtime: rt })
        }
        fn rms_norm_fwd(
            &self,
            layout: &Layout,
            weight: Option<(&Self, &Layout)>,
            outer_size: usize,
            inner_size: usize,
            eps: f32,
        ) -> Result<Self> {
            let src = self.compact(layout)?;
            let rt = src.runtime.clone();
            let weight_c = weight.map(|(w, wl)| w.compact(wl)).transpose()?;
            if weight_c.as_ref().is_some_and(|w| w.dtype() != src.dtype()) {
                return Err(Error::DTypeMismatch(
                    "rms_norm_fwd: dtype mismatch between input and weight".into(),
                ));
            }
            let block = [256usize, 128, 64, 32, 16, 8, 4, 2, 1]
                .into_iter()
                .find(|&b| inner_size.is_multiple_of(b))
                .unwrap();
            let inner =
                match (&src.inner, weight_c.as_ref().map(|w| &w.inner)) {
                    (CudaInner::F16(x), None) => CudaInner::F16(rms_norm_tensor(
                        &rt, x, None, outer_size, inner_size, eps, block,
                    )?),
                    (CudaInner::F16(x), Some(CudaInner::F16(w))) => CudaInner::F16(
                        rms_norm_tensor(&rt, x, Some(w), outer_size, inner_size, eps, block)?,
                    ),
                    (CudaInner::BF16(x), None) => CudaInner::BF16(rms_norm_tensor(
                        &rt, x, None, outer_size, inner_size, eps, block,
                    )?),
                    (CudaInner::BF16(x), Some(CudaInner::BF16(w))) => CudaInner::BF16(
                        rms_norm_tensor(&rt, x, Some(w), outer_size, inner_size, eps, block)?,
                    ),
                    (CudaInner::F32(x), None) => CudaInner::F32(rms_norm_tensor(
                        &rt, x, None, outer_size, inner_size, eps, block,
                    )?),
                    (CudaInner::F32(x), Some(CudaInner::F32(w))) => CudaInner::F32(
                        rms_norm_tensor(&rt, x, Some(w), outer_size, inner_size, eps, block)?,
                    ),
                    _ => return Err(Error::DTypeMismatch("rms_norm_fwd requires floats".into())),
                };
            Ok(Self { inner, runtime: rt })
        }
        fn dtype(&self) -> DType {
            match &self.inner {
                CudaInner::F16(_) => DType::F16,
                CudaInner::BF16(_) => DType::BF16,
                CudaInner::F32(_) => DType::F32,
                CudaInner::I64(_) => DType::I64,
            }
        }
        fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
            if self.dtype() == dtype {
                return self.copy_to_device(layout);
            }
            let compact = self.compact(layout)?;
            let rt = compact.runtime.clone();
            fn convert<F: cuda_core::DType + DeviceRepr, T: cuda_core::DType + DeviceRepr>(
                rt: &Arc<Runtime>,
                src: &Arc<Tensor<F>>,
            ) -> Result<Arc<Tensor<T>>> {
                cast_tensor(rt, src).map_err(|e| Error::Cuda(format!("cuTile cast failed: {e}")))
            }
            let inner = match (&compact.inner, dtype) {
                (CudaInner::F16(src), DType::F32) => CudaInner::F32(convert(&rt, src)?),
                (CudaInner::F16(src), DType::BF16) => CudaInner::BF16(convert(&rt, src)?),
                (CudaInner::F16(src), DType::I64) => CudaInner::I64(convert(&rt, src)?),
                (CudaInner::BF16(src), DType::F16) => CudaInner::F16(convert(&rt, src)?),
                (CudaInner::BF16(src), DType::F32) => CudaInner::F32(convert(&rt, src)?),
                (CudaInner::BF16(src), DType::I64) => CudaInner::I64(convert(&rt, src)?),
                (CudaInner::F32(src), DType::F16) => CudaInner::F16(convert(&rt, src)?),
                (CudaInner::F32(src), DType::BF16) => CudaInner::BF16(convert(&rt, src)?),
                (CudaInner::F32(src), DType::I64) => CudaInner::I64(convert(&rt, src)?),
                (CudaInner::I64(src), DType::F16) => CudaInner::F16(convert(&rt, src)?),
                (CudaInner::I64(src), DType::BF16) => CudaInner::BF16(convert(&rt, src)?),
                (CudaInner::I64(src), DType::F32) => CudaInner::F32(convert(&rt, src)?),
                _ => return self.copy_to_device(layout),
            };
            Ok(Self { inner, runtime: rt })
        }
        fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
            let layout = layout.borrow();
            let cpu = self.copy_to_cpu(layout).expect("cuda to_vec failed");
            D::to_vec(&cpu)
        }
        fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()> {
            // A compact layout may still cover fewer elements than the
            // storage holds (e.g. a narrowed view), so copy exactly the
            // layout's size rather than the whole buffer.
            if src_layout.is_compact() && src_layout.offset == 0 {
                return dst.copy_prefix(self, src_layout.size());
            }
            assert!(src_layout.ndim() <= 8, "cuda backend supports at most 8 dims");
            fn err(e: impl std::fmt::Display) -> Error {
                Error::Cuda(format!("cuTile copy_compact failed: {e}"))
            }
            let len = src_layout.size();
            if len == 0 {
                return Ok(());
            }
            // Pad missing dimensions with shape 1 and stride 0: they address
            // element zero repeatedly, which contributes nothing to the sum.
            let mut shape = [1i32; 8];
            let mut strides = [0i32; 8];
            for (i, dim) in src_layout.shape().iter().enumerate() {
                shape[i] = i32::try_from(*dim).map_err(err)?;
            }
            for (i, stride) in src_layout.strides().iter().enumerate() {
                strides[i] = i32::try_from(*stride).map_err(err)?;
            }
            let len_i32 = i32::try_from(len).map_err(err)?;
            let offset = i32::try_from(src_layout.offset).map_err(err)?;
            let blocks = len.div_ceil(128) as u32;
            match (&self.inner, &dst.inner) {
                (CudaInner::F32(s), CudaInner::F32(d)) => {
                    unsafe {
                        kernels::copy_compact(
                            d.device_pointer(),
                            s.device_pointer(),
                            len_i32,
                            offset,
                            shape[0],
                            shape[1],
                            shape[2],
                            shape[3],
                            shape[4],
                            shape[5],
                            shape[6],
                            shape[7],
                            strides[0],
                            strides[1],
                            strides[2],
                            strides[3],
                            strides[4],
                            strides[5],
                            strides[6],
                            strides[7],
                        )
                    }
                    .grid((blocks, 1, 1))
                    .async_profiled(&self.runtime)
                    .map_err(err)?;
                }
                (CudaInner::F16(s), CudaInner::F16(d)) => {
                    unsafe {
                        kernels::copy_compact(
                            d.device_pointer(),
                            s.device_pointer(),
                            len_i32,
                            offset,
                            shape[0],
                            shape[1],
                            shape[2],
                            shape[3],
                            shape[4],
                            shape[5],
                            shape[6],
                            shape[7],
                            strides[0],
                            strides[1],
                            strides[2],
                            strides[3],
                            strides[4],
                            strides[5],
                            strides[6],
                            strides[7],
                        )
                    }
                    .grid((blocks, 1, 1))
                    .async_profiled(&self.runtime)
                    .map_err(err)?;
                }
                (CudaInner::BF16(s), CudaInner::BF16(d)) => {
                    unsafe {
                        kernels::copy_compact(
                            d.device_pointer(),
                            s.device_pointer(),
                            len_i32,
                            offset,
                            shape[0],
                            shape[1],
                            shape[2],
                            shape[3],
                            shape[4],
                            shape[5],
                            shape[6],
                            shape[7],
                            strides[0],
                            strides[1],
                            strides[2],
                            strides[3],
                            strides[4],
                            strides[5],
                            strides[6],
                            strides[7],
                        )
                    }
                    .grid((blocks, 1, 1))
                    .async_profiled(&self.runtime)
                    .map_err(err)?;
                }
                (CudaInner::I64(s), CudaInner::I64(d)) => {
                    unsafe {
                        kernels::copy_compact(
                            d.device_pointer(),
                            s.device_pointer(),
                            len_i32,
                            offset,
                            shape[0],
                            shape[1],
                            shape[2],
                            shape[3],
                            shape[4],
                            shape[5],
                            shape[6],
                            shape[7],
                            strides[0],
                            strides[1],
                            strides[2],
                            strides[3],
                            strides[4],
                            strides[5],
                            strides[6],
                            strides[7],
                        )
                    }
                    .grid((blocks, 1, 1))
                    .async_profiled(&self.runtime)
                    .map_err(err)?;
                }
                _ => return Err(Error::DTypeMismatch("copy_compact dtype mismatch".into())),
            }
            Ok(())
        }
    }

    trait BlasDType: cuda_core::DType {
        const BLAS_TYPE: blas_sys::cudaDataType;
    }

    impl BlasDType for f32 {
        const BLAS_TYPE: blas_sys::cudaDataType = blas_sys::cudaDataType_t::CUDA_R_32F;
    }
    impl BlasDType for f16 {
        const BLAS_TYPE: blas_sys::cudaDataType = blas_sys::cudaDataType_t::CUDA_R_16F;
    }
    impl BlasDType for bf16 {
        const BLAS_TYPE: blas_sys::cudaDataType = blas_sys::cudaDataType_t::CUDA_R_16BF;
    }

    #[allow(clippy::too_many_arguments)]
    fn blas_gemm<T: BlasDType + DeviceRepr>(
        rt: &Arc<Runtime>,
        lhs: &Arc<Tensor<T>>,
        rhs: &Arc<Tensor<T>>,
        lhs_view: GemmOperand,
        rhs_view: GemmOperand,
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<Arc<Tensor<T>>> {
        let err = |e: &dyn std::fmt::Display| Error::Cuda(format!("cuBLAS matmul failed: {e}"));
        let m = i32::try_from(m).map_err(|e| err(&e))?;
        let n = i32::try_from(n).map_err(|e| err(&e))?;
        let k = i32::try_from(k).map_err(|e| err(&e))?;
        let batch = i32::try_from(batch).map_err(|e| err(&e))?;
        let stride_c = i64::from(m) * i64::from(n);
        // cuBLAS with beta=0 writes every output element before we expose it.
        let output: Tensor<T> =
            alloc_foreign(rt, stride_c as usize * batch as usize).map_err(|e| err(&e))?;
        let lhs_ptr = (lhs.device_pointer().cu_deviceptr() as usize
            + lhs_view.offset * std::mem::size_of::<T>())
            as *const std::ffi::c_void;
        let rhs_ptr = (rhs.device_pointer().cu_deviceptr() as usize
            + rhs_view.offset * std::mem::size_of::<T>())
            as *const std::ffi::c_void;
        let out_ptr = output.device_pointer().cu_deviceptr() as usize as *mut std::ffi::c_void;
        let blas = rt.blas()?.lock().expect("cuBLAS handle mutex poisoned");
        rt.device.bind_to_thread().map_err(|e| err(&e))?;
        let scope = crate::profiler::current_scope_id();
        let mut events = if scope.is_some() {
            rt.device.new_event().ok().zip(rt.device.new_event().ok())
        } else {
            None
        };
        if events.as_ref().is_some_and(|(start, _)| start.record(&rt.stream).is_err()) {
            events = None;
        }
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // All buffers and the handle remain live until the stream completes.
        // The row-major multiplication A(m,k) B(k,n) is the column-major
        // multiplication B^T(n,k) A^T(k,m) with reversed operands.
        let launch = unsafe {
            cublas::gemm_strided_batched_ex(
                blas.raw,
                rhs_view.op,
                lhs_view.op,
                n,
                m,
                k,
                &alpha as *const f32 as *const _,
                rhs_ptr,
                T::BLAS_TYPE,
                rhs_view.ld,
                rhs_view.batch_stride,
                lhs_ptr,
                T::BLAS_TYPE,
                lhs_view.ld,
                lhs_view.batch_stride,
                &beta as *const f32 as *const _,
                out_ptr,
                T::BLAS_TYPE,
                n,
                stride_c,
                batch,
                blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };
        if events.as_ref().is_some_and(|(_, end)| end.record(&rt.stream).is_err()) {
            events = None;
        }
        // Dropped buffers remain valid for queued cuBLAS work until the
        // stream-ordered free. Collect event times after the profiler's final
        // stream synchronization, without waiting here.
        launch.map_err(|e| err(&e))?;
        if let (Some(id), Some((start, end))) = (scope, events) {
            crate::profiler::queue_cuda_timing(id, start, end);
        }
        Ok(Arc::new(output))
    }

    pub fn synchronize() {
        if let Ok(runtime) = runtime() {
            let _ = runtime.device.bind_to_thread();
            unsafe {
                let _ = runtime.stream.synchronize();
            }
        }
    }

    pub fn availability() -> Result<()> {
        runtime().map(|_| ())
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
mod imp {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct CudaStorage;

    impl CudaStorage {
        pub fn is_available() -> bool {
            false
        }
        pub fn zeros(_size: usize, _dtype: DType) -> Self {
            panic!("cuda backend unavailable")
        }
        pub fn ones(_size: usize, _dtype: DType) -> Self {
            panic!("cuda backend unavailable")
        }
        pub fn from_cpu_storage(_inner: CpuStorage) -> Self {
            panic!("cuda backend unavailable")
        }
        pub(crate) fn copy_to_device(&self, _: &Layout) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        pub(crate) fn copy_to_cpu(&self, _: &Layout) -> Result<CpuStorage> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        pub(crate) fn copy_from_cpu(_src: &CpuStorage, _: &Layout) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        pub fn cat(_parts: &[(&CudaStorage, usize)]) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
    }

    impl BackendStorage for CudaStorage {
        fn ewise_powf(&self, _: f64, _: &Layout) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn unary_op<O: UnaryOp>(&self, _: O, _: &Layout) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn binary_op<O: BinaryOp>(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn ne_scalar(&self, _: &Layout, _: f64) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn ne_scalar_i64(&self, _: &Layout, _: i64) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn select(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn reduce<O: ReduceOp>(&self, _: &Layout, _: &mut Self) -> Result<()> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn matmul(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn gather(&self, _: &Layout, _: usize, _: &Self, _: &Layout) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn scatter_add(
            &self,
            _: &Layout,
            _: usize,
            _: &Self,
            _: &Layout,
            _: &[usize],
        ) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn index_select(&self, _: &Layout, _: usize, _: &Self, _: &Layout) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn index_add(
            &self,
            _: &Layout,
            _: usize,
            _: &Self,
            _: &Layout,
            _: &[usize],
        ) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn log_sum_exp(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn log_softmax_fwd(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn log_softmax_bwd(
            &self,
            _: &Layout,
            _: &Self,
            _: &Layout,
            _: usize,
            _: usize,
        ) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn rms_norm_fwd(
            &self,
            _: &Layout,
            _: Option<(&Self, &Layout)>,
            _: usize,
            _: usize,
            _: f32,
        ) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn dtype(&self) -> DType {
            panic!("cuda backend unavailable")
        }
        fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
        fn to_vec<D: WithDType>(&self, _: impl Borrow<Layout>) -> Vec<D> {
            panic!("cuda backend unavailable")
        }
        fn copy_compact(&self, _: &Layout, _: &mut Self) -> Result<()> {
            Err(Error::Cuda("cuda backend unavailable".into()))
        }
    }

    pub fn synchronize() {}

    pub fn availability() -> Result<()> {
        Err(Error::Cuda("cuda backend unavailable".into()))
    }
}

pub use imp::*;
