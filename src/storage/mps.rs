use std::borrow::Borrow;

use crate::{
    dtype::{DType, WithDType},
    error::Result,
    layout::Layout,
    storage::{BackendStorage, BinaryOp, CpuStorage, ReduceOp, UnaryOp},
};

// Metadata structs passed to Metal kernels via set_bytes.
// These must match the MSL struct layouts in kernels.metal exactly (repr(C)).

const MAX_DIMS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
struct StridedMeta {
    ndim: u32,
    offset: u32,
    size: u32,
    pad: u32,
    shape: [u32; MAX_DIMS],
    strides: [u32; MAX_DIMS],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScalarMeta {
    input: StridedMeta,
    scalar: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BinaryMeta {
    lhs: StridedMeta,
    rhs: StridedMeta,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReduceMeta {
    outer_size: u32,
    reduce_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulMeta {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GatherMeta {
    rows: u32,
    cols: u32,
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use metal::{
        Buffer, CommandBuffer, CommandQueue, CompileOptions, ComputePipelineState, Device,
        MTLResourceOptions, MTLSize,
    };
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::fmt;
    use std::sync::{Arc, Mutex, OnceLock};

    const KERNELS: &str = include_str!("kernels.metal");

    const ALL_KERNELS: &[&str] = &[
        "neg_f32",
        "exp_f32",
        "log_f32",
        "tanh_f32",
        "relu_f32",
        "relu_backward_f32",
        "scalar_add_f32",
        "scalar_mul_f32",
        "scalar_div_f32",
        "scalar_powf_f32",
        "copy_compact_f32",
        "copy_compact_u32",
        "add_f32",
        "sub_f32",
        "mul_f32",
        "div_f32",
        "pow_f32",
        "eq_f32",
        "reduce_sum_f32",
        "reduce_max_f32",
        "matmul_f32",
        "gather_f32",
        "scatter_f32",
    ];

    #[derive(Debug)]
    struct MpsContext {
        device: Device,
        queue: CommandQueue,
        pipelines: HashMap<&'static str, ComputePipelineState>,
        active_command_buffer: Mutex<Option<CommandBuffer>>,
    }

    impl MpsContext {
        fn shared() -> Arc<Self> {
            static CTX: OnceLock<Arc<MpsContext>> = OnceLock::new();
            CTX.get_or_init(|| Arc::new(Self::new())).clone()
        }

        fn new() -> Self {
            let device = Device::system_default().expect("MPS requires a Metal device");
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(KERNELS, &options)
                .expect("failed to compile Metal kernels");
            let queue = device.new_command_queue();

            // Pre-cache all pipelines at init — no locks needed on the hot path
            let mut pipelines = HashMap::new();
            for &name in ALL_KERNELS {
                let function = library
                    .get_function(name, None)
                    .unwrap_or_else(|_| panic!("missing Metal function {name}"));
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .unwrap_or_else(|_| panic!("failed to build Metal pipeline {name}"));
                pipelines.insert(name, pipeline);
            }

            Self { device, queue, pipelines, active_command_buffer: Mutex::new(None) }
        }

        fn command_buffer(&self) -> std::sync::MutexGuard<'_, Option<CommandBuffer>> {
            let mut guard = self.active_command_buffer.lock().unwrap();
            if guard.is_none() {
                *guard = Some(self.queue.new_command_buffer().to_owned());
            }
            guard
        }

        fn pipeline(&self, name: &'static str) -> &ComputePipelineState {
            self.pipelines.get(name).unwrap_or_else(|| panic!("missing pipeline {name}"))
        }

        fn buffer_from_f32(&self, data: &[f32]) -> Buffer {
            let byte_len = std::mem::size_of_val(data) as u64;
            self.device.new_buffer_with_data(
                data.as_ptr().cast::<c_void>(),
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
        }

        fn buffer_from_u32(&self, data: &[u32]) -> Buffer {
            let byte_len = std::mem::size_of_val(data) as u64;
            self.device.new_buffer_with_data(
                data.as_ptr().cast::<c_void>(),
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
        }

        fn empty_f32_buffer(&self, len: usize) -> Buffer {
            self.device.new_buffer(
                (len * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        fn empty_u32_buffer(&self, len: usize) -> Buffer {
            self.device.new_buffer(
                (len * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        fn synchronize(&self) {
            let mut guard = self.active_command_buffer.lock().unwrap();
            if let Some(cb) = guard.take() {
                cb.commit();
                cb.wait_until_completed();
            } else {
                // Nothing pending — still need a fence for shared-memory buffers
                let cb = self.queue.new_command_buffer();
                cb.commit();
                cb.wait_until_completed();
            }
        }

        fn set_params<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
            encoder.set_bytes(
                index,
                std::mem::size_of::<T>() as u64,
                (value as *const T).cast::<c_void>(),
            );
        }

        fn dispatch_1d(
            &self,
            pipeline_name: &'static str,
            total_threads: usize,
            configure: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) {
            let pipeline = self.pipeline(pipeline_name);
            let guard = self.command_buffer();
            let cb = guard.as_ref().unwrap();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            configure(&encoder);
            let width = total_threads.max(1) as u64;
            let tg = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
            encoder.dispatch_threads(MTLSize::new(width, 1, 1), MTLSize::new(tg.max(1), 1, 1));
            encoder.end_encoding();
        }

        fn dispatch_2d(
            &self,
            pipeline_name: &'static str,
            width: usize,
            height: usize,
            configure: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) {
            self.dispatch_3d(pipeline_name, width, height, 1, configure);
        }

        fn dispatch_3d(
            &self,
            pipeline_name: &'static str,
            width: usize,
            height: usize,
            depth: usize,
            configure: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) {
            let pipeline = self.pipeline(pipeline_name);
            let guard = self.command_buffer();
            let cb = guard.as_ref().unwrap();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            configure(&encoder);
            encoder.dispatch_threads(
                MTLSize::new(width as u64, height as u64, depth as u64),
                MTLSize::new(16, 16, 1),
            );
            encoder.end_encoding();
        }
    }

    #[derive(Clone)]
    enum MpsInner {
        Accelerated { ctx: Arc<MpsContext>, buffer: Buffer, len: usize, dtype: DType },
        Cpu(CpuStorage),
    }

    #[derive(Clone)]
    pub struct MpsStorage {
        inner: MpsInner,
    }

    impl fmt::Debug for MpsStorage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match &self.inner {
                MpsInner::Accelerated { len, dtype, .. } => f
                    .debug_struct("MpsStorage")
                    .field("backend", &"metal")
                    .field("dtype", dtype)
                    .field("len", len)
                    .finish(),
                MpsInner::Cpu(storage) => f.debug_tuple("MpsStorage").field(storage).finish(),
            }
        }
    }

    impl MpsStorage {
        pub fn empty(len: usize, dtype: DType) -> Self {
            let ctx = MpsContext::shared();
            let buffer = match dtype {
                DType::F32 => ctx.empty_f32_buffer(len),
                DType::U32 => ctx.empty_u32_buffer(len),
                DType::F64 => todo!(),
                DType::F16 => todo!(),
            };
            Self { inner: MpsInner::Accelerated { ctx, buffer, len, dtype } }
        }

        pub fn zeros(len: usize, dtype: DType) -> Self {
            let storage = Self::empty(len, dtype);
            let (buffer, byte_len) = match &storage.inner {
                MpsInner::Accelerated { buffer, len, dtype, .. } => {
                    let elem_size = match dtype {
                        DType::F32 => std::mem::size_of::<f32>(),
                        DType::U32 => std::mem::size_of::<u32>(),
                        DType::F64 => todo!(),
                        DType::F16 => todo!(),
                    };
                    (buffer, len * elem_size)
                }
                MpsInner::Cpu(_) => unreachable!(),
            };
            unsafe {
                std::ptr::write_bytes(buffer.contents(), 0, byte_len);
            }
            storage
        }

        pub fn ones(len: usize, dtype: DType) -> Self {
            let storage = Self::empty(len, dtype);
            match &storage.inner {
                MpsInner::Accelerated { buffer, len, dtype: DType::F32, .. } => {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), *len)
                    };
                    slice.fill(1.0);
                }
                MpsInner::Accelerated { buffer, len, dtype: DType::U32, .. } => {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer.contents().cast::<u32>(), *len)
                    };
                    slice.fill(1);
                }
                MpsInner::Accelerated { dtype, .. } => todo!("MPS ones for {dtype:?}"),
                MpsInner::Cpu(_) => unreachable!(),
            }
            storage
        }

        pub fn from_cpu_storage(inner: CpuStorage) -> Self {
            match inner {
                CpuStorage::F32(data) => {
                    let ctx = MpsContext::shared();
                    let buffer = ctx.buffer_from_f32(&data);
                    Self {
                        inner: MpsInner::Accelerated {
                            ctx,
                            buffer,
                            len: data.len(),
                            dtype: DType::F32,
                        },
                    }
                }
                CpuStorage::U32(data) => {
                    let ctx = MpsContext::shared();
                    let buffer = ctx.buffer_from_u32(&data);
                    Self {
                        inner: MpsInner::Accelerated {
                            ctx,
                            buffer,
                            len: data.len(),
                            dtype: DType::U32,
                        },
                    }
                }
                storage => Self { inner: MpsInner::Cpu(storage) },
            }
        }

        /// Concatenates compact MPS storages by memcpy into a single output buffer.
        pub fn cat(parts: &[(&MpsStorage, usize)]) -> MpsStorage {
            assert!(!parts.is_empty());
            let total_len: usize = parts.iter().map(|(_, len)| *len).sum();
            let dtype = match &parts[0].0.inner {
                MpsInner::Accelerated { dtype, .. } => *dtype,
                MpsInner::Cpu(_) => panic!("cat requires accelerated MPS storage"),
            };
            let ctx = MpsContext::shared();
            ctx.synchronize();
            let elem_size = match dtype {
                DType::F32 => std::mem::size_of::<f32>(),
                DType::U32 => std::mem::size_of::<u32>(),
                _ => todo!("MPS cat for {dtype:?}"),
            };
            let out_buffer = match dtype {
                DType::F32 => ctx.empty_f32_buffer(total_len),
                DType::U32 => ctx.empty_u32_buffer(total_len),
                _ => unreachable!(),
            };
            let mut byte_offset = 0usize;
            for (storage, len) in parts {
                let src_buffer = match &storage.inner {
                    MpsInner::Accelerated { buffer, .. } => buffer,
                    MpsInner::Cpu(_) => panic!("cat requires accelerated MPS storage"),
                };
                let byte_len = len * elem_size;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_buffer.contents().cast::<u8>(),
                        out_buffer.contents().cast::<u8>().add(byte_offset),
                        byte_len,
                    );
                }
                byte_offset += byte_len;
            }
            MpsStorage {
                inner: MpsInner::Accelerated { ctx, buffer: out_buffer, len: total_len, dtype },
            }
        }

        pub fn into_cpu(self) -> CpuStorage {
            match self.inner {
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::F32 } => {
                    ctx.synchronize();
                    let slice =
                        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len) };
                    CpuStorage::F32(slice.to_vec())
                }
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::U32 } => {
                    ctx.synchronize();
                    let slice =
                        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u32>(), len) };
                    CpuStorage::U32(slice.to_vec())
                }
                MpsInner::Accelerated { dtype, .. } => todo!("MPS readback for {dtype:?}"),
                MpsInner::Cpu(storage) => storage,
            }
        }

        fn as_cpu_storage(&self) -> CpuStorage {
            self.clone().into_cpu()
        }

        fn strided_meta(layout: &Layout) -> StridedMeta {
            assert!(layout.ndim() <= MAX_DIMS);
            let mut shape = [1u32; MAX_DIMS];
            let mut strides = [0u32; MAX_DIMS];
            for (i, dim) in layout.shape().iter().copied().enumerate() {
                shape[i] = dim as u32;
            }
            for (i, stride) in layout.strides().iter().copied().enumerate() {
                strides[i] = stride as u32;
            }
            StridedMeta {
                ndim: layout.ndim() as u32,
                offset: layout.offset as u32,
                size: layout.size() as u32,
                pad: 0,
                shape,
                strides,
            }
        }

        fn unary_kernel_name<O: UnaryOp>() -> Option<&'static str> {
            match O::KERNEL {
                "Neg" => Some("neg_f32"),
                "exp" => Some("exp_f32"),
                "log" => Some("log_f32"),
                "tanh" => Some("tanh_f32"),
                "relu" => Some("relu_f32"),
                "relu_backward" => Some("relu_backward_f32"),
                _ => None,
            }
        }

        fn scalar_kernel_name<O: UnaryOp>() -> Option<&'static str> {
            match O::KERNEL {
                "scalar_add" => Some("scalar_add_f32"),
                "scalar_mul" => Some("scalar_mul_f32"),
                "scalar_div" => Some("scalar_div_f32"),
                _ => None,
            }
        }

        fn binary_kernel_name<O: BinaryOp>() -> Option<&'static str> {
            match O::KERNEL {
                "add" => Some("add_f32"),
                "sub" => Some("sub_f32"),
                "mul" => Some("mul_f32"),
                "div" => Some("div_f32"),
                "powf" => Some("pow_f32"),
                "eq" => Some("eq_f32"),
                _ => None,
            }
        }

        fn accelerated(&self, dtype: DType) -> Option<(&Arc<MpsContext>, &Buffer, usize)> {
            match &self.inner {
                MpsInner::Accelerated { ctx, buffer, len, dtype: inner_dtype }
                    if *inner_dtype == dtype =>
                {
                    Some((ctx, buffer, *len))
                }
                MpsInner::Cpu(_) => None,
                MpsInner::Accelerated { .. } => None,
            }
        }
    }

    impl From<Vec<f32>> for MpsStorage {
        fn from(value: Vec<f32>) -> Self {
            Self::from_cpu_storage(CpuStorage::from(value))
        }
    }

    impl From<Vec<f64>> for MpsStorage {
        fn from(value: Vec<f64>) -> Self {
            Self::from_cpu_storage(CpuStorage::from(value))
        }
    }

    impl BackendStorage for MpsStorage {
        fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
            if let Some((ctx, input, _)) = self.accelerated(DType::F32) {
                let out = ctx.empty_f32_buffer(l.size());
                let params = ScalarMeta {
                    input: Self::strided_meta(l),
                    scalar: e as f32,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                };
                ctx.dispatch_1d("scalar_powf_f32", l.size(), |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(&out), 0);
                    MpsContext::set_params(encoder, 2, &params);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: l.size(),
                        dtype: DType::F32,
                    },
                });
            }

            let inner = self.as_cpu_storage().ewise_powf(e, l)?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
            if let Some((ctx, input, _)) = self.accelerated(DType::F32) {
                if let Some(kernel) = Self::unary_kernel_name::<O>() {
                    let out = ctx.empty_f32_buffer(l.size());
                    let meta = Self::strided_meta(l);
                    ctx.dispatch_1d(kernel, l.size(), |encoder| {
                        encoder.set_buffer(0, Some(input), 0);
                        encoder.set_buffer(1, Some(&out), 0);
                        MpsContext::set_params(encoder, 2, &meta);
                    });
                    return Ok(Self {
                        inner: MpsInner::Accelerated {
                            ctx: ctx.clone(),
                            buffer: out,
                            len: l.size(),
                            dtype: DType::F32,
                        },
                    });
                }

                if let Some(kernel) = Self::scalar_kernel_name::<O>() {
                    let scalar = match O::KERNEL {
                        "scalar_add" => op.f32(0.0),
                        "scalar_mul" => op.f32(1.0),
                        "scalar_div" => 1.0 / op.f32(1.0),
                        _ => unreachable!(),
                    };
                    let out = ctx.empty_f32_buffer(l.size());
                    let params = ScalarMeta {
                        input: Self::strided_meta(l),
                        scalar,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    };
                    ctx.dispatch_1d(kernel, l.size(), |encoder| {
                        encoder.set_buffer(0, Some(input), 0);
                        encoder.set_buffer(1, Some(&out), 0);
                        MpsContext::set_params(encoder, 2, &params);
                    });
                    return Ok(Self {
                        inner: MpsInner::Accelerated {
                            ctx: ctx.clone(),
                            buffer: out,
                            len: l.size(),
                            dtype: DType::F32,
                        },
                    });
                }
            }

            let inner = self.as_cpu_storage().unary_op(op, l)?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn binary_op<O: BinaryOp>(
            &self,
            layout: &Layout,
            other: &Self,
            other_layout: &Layout,
        ) -> Result<Self> {
            if let (Some((ctx, lhs, _)), Some((_, rhs, _))) =
                (self.accelerated(DType::F32), other.accelerated(DType::F32))
            {
                if let Some(kernel) = Self::binary_kernel_name::<O>() {
                    let out = ctx.empty_f32_buffer(layout.size());
                    let meta = BinaryMeta {
                        lhs: Self::strided_meta(layout),
                        rhs: Self::strided_meta(other_layout),
                    };
                    ctx.dispatch_1d(kernel, layout.size(), |encoder| {
                        encoder.set_buffer(0, Some(lhs), 0);
                        encoder.set_buffer(1, Some(rhs), 0);
                        encoder.set_buffer(2, Some(&out), 0);
                        MpsContext::set_params(encoder, 3, &meta);
                    });
                    return Ok(Self {
                        inner: MpsInner::Accelerated {
                            ctx: ctx.clone(),
                            buffer: out,
                            len: layout.size(),
                            dtype: DType::F32,
                        },
                    });
                }
            }

            let inner = self.as_cpu_storage().binary_op::<O>(
                layout,
                &other.as_cpu_storage(),
                other_layout,
            )?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
            if let (Some((ctx, input, _)), Some((_, output, out_len))) =
                (self.accelerated(DType::F32), dst.accelerated(DType::F32))
            {
                assert!(layout.is_compact());
                let reduce_size = layout.size() / out_len;
                let meta =
                    ReduceMeta { outer_size: out_len as u32, reduce_size: reduce_size as u32 };
                let kernel = match O::KERNEL {
                    "reduce_sum" => "reduce_sum_f32",
                    "reduce_max" => "reduce_max_f32",
                    _ => unreachable!(),
                };
                ctx.dispatch_1d(kernel, out_len, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(output), 0);
                    MpsContext::set_params(encoder, 2, &meta);
                });
                return Ok(());
            }

            self.as_cpu_storage().reduce::<O>(layout, &mut dst.clone().into_cpu())?;
            *dst = Self::from_cpu_storage(dst.clone().into_cpu());
            Ok(())
        }

        fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
            if let (Some((ctx, lhs, _)), Some((_, rhs, _))) =
                (self.accelerated(DType::F32), other.accelerated(DType::F32))
            {
                assert!(layout.is_compact() && layout_other.is_compact());
                let ndim = layout.ndim();
                let m = layout.shape()[ndim - 2];
                let k = layout.shape()[ndim - 1];
                let n = layout_other.shape()[ndim - 1];
                let batch: usize = (0..ndim - 2).map(|i| layout.shape()[i]).product();
                let total = batch * m * n;
                let out = ctx.empty_f32_buffer(total);
                let meta =
                    MatmulMeta { m: m as u32, k: k as u32, n: n as u32, batch: batch as u32 };
                // Matmul uses tiled algorithm — must dispatch full threadgroups
                // so all 16x16 threads cooperate on tile loading.
                let pipeline = ctx.pipeline("matmul_f32");
                let guard = ctx.command_buffer();
                let cb = guard.as_ref().unwrap();
                let encoder = cb.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(lhs), 0);
                encoder.set_buffer(1, Some(rhs), 0);
                encoder.set_buffer(2, Some(&out), 0);
                MpsContext::set_params(&encoder, 3, &meta);
                let tile = 16u64;
                let groups = MTLSize::new(
                    (n as u64 + tile - 1) / tile,
                    (m as u64 + tile - 1) / tile,
                    batch as u64,
                );
                let tg_size = MTLSize::new(tile, tile, 1);
                encoder.dispatch_thread_groups(groups, tg_size);
                encoder.end_encoding();
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: total,
                        dtype: DType::F32,
                    },
                });
            }

            let inner =
                self.as_cpu_storage().matmul(layout, &other.as_cpu_storage(), layout_other)?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn gather(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
        ) -> Result<Self> {
            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F32), indices.accelerated(DType::U32))
            {
                assert!(layout.is_compact());
                assert!(indices_layout.is_compact());
                assert_eq!(dim, 1);
                let rows = layout.shape()[0];
                let cols = layout.shape()[1];
                let out = ctx.empty_f32_buffer(rows);
                let meta = GatherMeta { rows: rows as u32, cols: cols as u32 };
                ctx.dispatch_1d("gather_f32", rows, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: rows,
                        dtype: DType::F32,
                    },
                });
            }

            let inner = self.as_cpu_storage().gather(
                layout,
                dim,
                &indices.as_cpu_storage(),
                indices_layout,
            )?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn scatter(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
            full_shape: &[usize],
        ) -> Result<Self> {
            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F32), indices.accelerated(DType::U32))
            {
                assert!(layout.is_compact());
                assert!(indices_layout.is_compact());
                assert_eq!(dim, 1);
                let rows = full_shape[0];
                let cols = full_shape[1];
                let out = ctx.empty_f32_buffer(rows * cols);
                unsafe {
                    std::ptr::write_bytes(
                        out.contents(),
                        0,
                        rows * cols * std::mem::size_of::<f32>(),
                    );
                }
                let meta = GatherMeta { rows: rows as u32, cols: cols as u32 };
                ctx.dispatch_1d("scatter_f32", rows, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: rows * cols,
                        dtype: DType::F32,
                    },
                });
            }

            let inner = self.as_cpu_storage().scatter(
                layout,
                dim,
                &indices.as_cpu_storage(),
                indices_layout,
                full_shape,
            )?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn dtype(&self) -> DType {
            match &self.inner {
                MpsInner::Accelerated { dtype, .. } => *dtype,
                MpsInner::Cpu(storage) => storage.dtype(),
            }
        }

        fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
            let layout = layout.borrow();
            match &self.inner {
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::F32 } => {
                    ctx.synchronize();
                    let data = read_f32(ctx, buffer, *len, layout);
                    D::to_vec(&CpuStorage::F32(data))
                }
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::U32 } => {
                    ctx.synchronize();
                    let data = read_u32(ctx, buffer, *len, layout);
                    D::to_vec(&CpuStorage::U32(data))
                }
                MpsInner::Accelerated { dtype, .. } => todo!("MPS to_vec for {dtype:?}"),
                MpsInner::Cpu(storage) => storage.to_vec(layout),
            }
        }

        fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()> {
            if let (Some((ctx, input, _)), Some((_, output, out_len))) =
                (self.accelerated(DType::F32), dst.accelerated(DType::F32))
            {
                assert_eq!(src_layout.size(), out_len);
                let meta = Self::strided_meta(src_layout);
                ctx.dispatch_1d("copy_compact_f32", out_len, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(output), 0);
                    MpsContext::set_params(encoder, 2, &meta);
                });
                return Ok(());
            }

            if let (Some((ctx, input, _)), Some((_, output, out_len))) =
                (self.accelerated(DType::U32), dst.accelerated(DType::U32))
            {
                assert_eq!(src_layout.size(), out_len);
                let meta = Self::strided_meta(src_layout);
                ctx.dispatch_1d("copy_compact_u32", out_len, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(output), 0);
                    MpsContext::set_params(encoder, 2, &meta);
                });
                return Ok(());
            }

            let src = self.as_cpu_storage();
            let mut cpu_dst = dst.as_cpu_storage();
            src.copy_compact(src_layout, &mut cpu_dst)?;
            *dst = Self::from_cpu_storage(cpu_dst);
            Ok(())
        }
    }

    fn read_f32(ctx: &Arc<MpsContext>, buffer: &Buffer, len: usize, layout: &Layout) -> Vec<f32> {
        if layout.is_compact() && layout.size() == len {
            return unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len) }
                .to_vec();
        }

        let tmp = MpsStorage {
            inner: MpsInner::Accelerated {
                ctx: ctx.clone(),
                buffer: buffer.to_owned(),
                len,
                dtype: DType::F32,
            },
        };
        let mut dst = MpsStorage {
            inner: MpsInner::Accelerated {
                ctx: ctx.clone(),
                buffer: ctx.empty_f32_buffer(layout.size()),
                len: layout.size(),
                dtype: DType::F32,
            },
        };
        tmp.copy_compact(layout, &mut dst).unwrap();
        ctx.synchronize();

        match dst.inner {
            MpsInner::Accelerated { buffer, len, .. } => {
                unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len) }.to_vec()
            }
            MpsInner::Cpu(_) => unreachable!(),
        }
    }

    fn read_u32(ctx: &Arc<MpsContext>, buffer: &Buffer, len: usize, layout: &Layout) -> Vec<u32> {
        if layout.is_compact() && layout.size() == len {
            return unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u32>(), len) }
                .to_vec();
        }

        let tmp = MpsStorage {
            inner: MpsInner::Accelerated {
                ctx: ctx.clone(),
                buffer: buffer.to_owned(),
                len,
                dtype: DType::U32,
            },
        };
        let mut dst = MpsStorage {
            inner: MpsInner::Accelerated {
                ctx: ctx.clone(),
                buffer: ctx.empty_u32_buffer(layout.size()),
                len: layout.size(),
                dtype: DType::U32,
            },
        };
        tmp.copy_compact(layout, &mut dst).unwrap();
        ctx.synchronize();

        match dst.inner {
            MpsInner::Accelerated { buffer, len, .. } => {
                unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u32>(), len) }.to_vec()
            }
            MpsInner::Cpu(_) => unreachable!(),
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct MpsStorage(());

    impl MpsStorage {
        fn unavailable() -> ! {
            panic!("MPS backend requires macOS — use Device::Cpu instead")
        }

        pub fn empty(_len: usize, _dtype: DType) -> Self {
            Self::unavailable()
        }
        pub fn zeros(_len: usize, _dtype: DType) -> Self {
            Self::unavailable()
        }
        pub fn ones(_len: usize, _dtype: DType) -> Self {
            Self::unavailable()
        }
        pub fn from_cpu_storage(_inner: CpuStorage) -> Self {
            Self::unavailable()
        }
        pub fn cat(_parts: &[(&MpsStorage, usize)]) -> Self {
            Self::unavailable()
        }
        pub fn into_cpu(self) -> CpuStorage {
            Self::unavailable()
        }
    }

    impl From<Vec<f32>> for MpsStorage {
        fn from(_: Vec<f32>) -> Self {
            Self::unavailable()
        }
    }

    impl From<Vec<f64>> for MpsStorage {
        fn from(_: Vec<f64>) -> Self {
            Self::unavailable()
        }
    }

    impl BackendStorage for MpsStorage {
        fn ewise_powf(&self, _: f64, _: &Layout) -> Result<Self> {
            Self::unavailable()
        }
        fn unary_op<O: UnaryOp>(&self, _: O, _: &Layout) -> Result<Self> {
            Self::unavailable()
        }
        fn binary_op<O: BinaryOp>(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
            Self::unavailable()
        }
        fn reduce<O: ReduceOp>(&self, _: &Layout, _: &mut Self) -> Result<()> {
            Self::unavailable()
        }
        fn matmul(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
            Self::unavailable()
        }
        fn gather(&self, _: &Layout, _: usize, _: &Self, _: &Layout) -> Result<Self> {
            Self::unavailable()
        }
        fn scatter(&self, _: &Layout, _: usize, _: &Self, _: &Layout, _: &[usize]) -> Result<Self> {
            Self::unavailable()
        }
        fn dtype(&self) -> DType {
            Self::unavailable()
        }
        fn to_vec<D: WithDType>(&self, _: impl Borrow<Layout>) -> Vec<D> {
            Self::unavailable()
        }
        fn copy_compact(&self, _: &Layout, _: &mut Self) -> Result<()> {
            Self::unavailable()
        }
    }
}

pub use imp::MpsStorage;
