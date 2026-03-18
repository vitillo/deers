use std::borrow::Borrow;

use crate::{
    dtype::{DType, WithDType},
    error::Result,
    layout::Layout,
    storage::{BackendStorage, BinaryOp, CpuStorage, ReduceOp, UnaryOp},
};

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

    const MAX_DIMS: usize = 8;

    const KERNELS: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        constant uint MAX_DIMS = 8;

        struct StridedMeta {
            uint ndim;
            uint offset;
            uint size;
            uint pad;
            uint shape[MAX_DIMS];
            uint strides[MAX_DIMS];
        };

        struct ScalarMeta {
            StridedMeta input;
            float scalar;
            uint pad0;
            uint pad1;
            uint pad2;
        };

        struct BinaryMeta {
            StridedMeta lhs;
            StridedMeta rhs;
        };

        struct ReduceMeta {
            uint outer_size;
            uint reduce_size;
        };

        struct MatmulMeta {
            uint m;
            uint n;
            uint p;
        };

        struct GatherMeta {
            uint rows;
            uint cols;
        };

        uint linear_to_offset(uint linear, constant StridedMeta& meta) {
            uint offset = meta.offset;
            uint remaining = linear;
            for (uint dim = meta.ndim; dim > 0; dim--) {
                uint i = dim - 1;
                uint coord = remaining % meta.shape[i];
                remaining /= meta.shape[i];
                offset += coord * meta.strides[i];
            }
            return offset;
        }

        kernel void neg_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant StridedMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.size) return;
            output[id] = -input[linear_to_offset(id, meta)];
        }

        kernel void exp_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant StridedMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.size) return;
            output[id] = exp(input[linear_to_offset(id, meta)]);
        }

        kernel void log_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant StridedMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.size) return;
            output[id] = log(input[linear_to_offset(id, meta)]);
        }

        kernel void relu_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant StridedMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.size) return;
            float v = input[linear_to_offset(id, meta)];
            output[id] = max(v, 0.0f);
        }

        kernel void relu_backward_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant StridedMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.size) return;
            output[id] = input[linear_to_offset(id, meta)] > 0.0f ? 1.0f : 0.0f;
        }

        kernel void scalar_add_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant ScalarMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.input.size) return;
            output[id] = input[linear_to_offset(id, meta.input)] + meta.scalar;
        }

        kernel void scalar_mul_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant ScalarMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.input.size) return;
            output[id] = input[linear_to_offset(id, meta.input)] * meta.scalar;
        }

        kernel void scalar_div_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant ScalarMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.input.size) return;
            output[id] = input[linear_to_offset(id, meta.input)] / meta.scalar;
        }

        kernel void scalar_powf_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant ScalarMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.input.size) return;
            output[id] = pow(input[linear_to_offset(id, meta.input)], meta.scalar);
        }

        kernel void copy_compact_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant StridedMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.size) return;
            output[id] = input[linear_to_offset(id, meta)];
        }

        kernel void copy_compact_u32(
            device const uint* input [[buffer(0)]],
            device uint* output [[buffer(1)]],
            constant StridedMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.size) return;
            output[id] = input[linear_to_offset(id, meta)];
        }

        kernel void add_f32(
            device const float* lhs [[buffer(0)]],
            device const float* rhs [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant BinaryMeta& meta [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.lhs.size) return;
            output[id] = lhs[linear_to_offset(id, meta.lhs)] + rhs[linear_to_offset(id, meta.rhs)];
        }

        kernel void sub_f32(
            device const float* lhs [[buffer(0)]],
            device const float* rhs [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant BinaryMeta& meta [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.lhs.size) return;
            output[id] = lhs[linear_to_offset(id, meta.lhs)] - rhs[linear_to_offset(id, meta.rhs)];
        }

        kernel void mul_f32(
            device const float* lhs [[buffer(0)]],
            device const float* rhs [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant BinaryMeta& meta [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.lhs.size) return;
            output[id] = lhs[linear_to_offset(id, meta.lhs)] * rhs[linear_to_offset(id, meta.rhs)];
        }

        kernel void div_f32(
            device const float* lhs [[buffer(0)]],
            device const float* rhs [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant BinaryMeta& meta [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.lhs.size) return;
            output[id] = lhs[linear_to_offset(id, meta.lhs)] / rhs[linear_to_offset(id, meta.rhs)];
        }

        kernel void pow_f32(
            device const float* lhs [[buffer(0)]],
            device const float* rhs [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant BinaryMeta& meta [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.lhs.size) return;
            output[id] = pow(lhs[linear_to_offset(id, meta.lhs)], rhs[linear_to_offset(id, meta.rhs)]);
        }

        kernel void eq_f32(
            device const float* lhs [[buffer(0)]],
            device const float* rhs [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant BinaryMeta& meta [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.lhs.size) return;
            output[id] = lhs[linear_to_offset(id, meta.lhs)] == rhs[linear_to_offset(id, meta.rhs)] ? 1.0f : 0.0f;
        }

        kernel void reduce_sum_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant ReduceMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.outer_size) return;
            uint start = id * meta.reduce_size;
            float acc = 0.0f;
            for (uint i = 0; i < meta.reduce_size; i++) {
                acc += input[start + i];
            }
            output[id] = acc;
        }

        kernel void reduce_max_f32(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant ReduceMeta& meta [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.outer_size) return;
            uint start = id * meta.reduce_size;
            float acc = input[start];
            for (uint i = 1; i < meta.reduce_size; i++) {
                acc = max(acc, input[start + i]);
            }
            output[id] = acc;
        }

        constant uint TILE = 16;

        kernel void matmul_f32(
            device const float* lhs [[buffer(0)]],
            device const float* rhs [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant MatmulMeta& meta [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            threadgroup float tileA[TILE][TILE];
            threadgroup float tileB[TILE][TILE];

            uint row = tgid.y * TILE + tid.y;
            uint col = tgid.x * TILE + tid.x;
            float acc = 0.0f;

            uint num_tiles = (meta.n + TILE - 1) / TILE;
            for (uint t = 0; t < num_tiles; t++) {
                uint ak = t * TILE + tid.x;
                uint bk = t * TILE + tid.y;
                tileA[tid.y][tid.x] = (row < meta.m && ak < meta.n) ? lhs[row * meta.n + ak] : 0.0f;
                tileB[tid.y][tid.x] = (bk < meta.n && col < meta.p) ? rhs[bk * meta.p + col] : 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint k = 0; k < TILE; k++) {
                    acc += tileA[tid.y][k] * tileB[k][tid.x];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row < meta.m && col < meta.p) {
                output[row * meta.p + col] = acc;
            }
        }

        kernel void gather_f32(
            device const float* input [[buffer(0)]],
            device const uint* indices [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant GatherMeta& meta [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.rows) return;
            output[id] = input[id * meta.cols + indices[id]];
        }

        kernel void scatter_f32(
            device const float* input [[buffer(0)]],
            device const uint* indices [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant GatherMeta& meta [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= meta.rows) return;
            output[id * meta.cols + indices[id]] = input[id];
        }
    "#;

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
        n: u32,
        p: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct GatherMeta {
        rows: u32,
        cols: u32,
    }

    const ALL_KERNELS: &[&str] = &[
        "neg_f32", "exp_f32", "log_f32", "relu_f32", "relu_backward_f32",
        "scalar_add_f32", "scalar_mul_f32", "scalar_div_f32", "scalar_powf_f32",
        "copy_compact_f32", "copy_compact_u32",
        "add_f32", "sub_f32", "mul_f32", "div_f32", "pow_f32", "eq_f32",
        "reduce_sum_f32", "reduce_max_f32",
        "matmul_f32", "gather_f32", "scatter_f32",
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

            Self {
                device,
                queue,
                pipelines,
                active_command_buffer: Mutex::new(None),
            }
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

        fn param_buffer<T>(&self, value: &T) -> Buffer {
            self.device.new_buffer_with_data(
                (value as *const T).cast::<c_void>(),
                std::mem::size_of::<T>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
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
            let pipeline = self.pipeline(pipeline_name);
            let guard = self.command_buffer();
            let cb = guard.as_ref().unwrap();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            configure(&encoder);
            encoder.dispatch_threads(
                MTLSize::new(width as u64, height as u64, 1),
                MTLSize::new(16, 16, 1),
            );
            encoder.end_encoding();
        }
    }

    #[derive(Clone)]
    enum MpsInner {
        Accelerated {
            ctx: Arc<MpsContext>,
            buffer: Buffer,
            len: usize,
            dtype: DType,
        },
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
            Self {
                inner: MpsInner::Accelerated {
                    ctx,
                    buffer,
                    len,
                    dtype,
                },
            }
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
                MpsInner::Accelerated {
                    buffer,
                    len,
                    dtype: DType::F32,
                    ..
                } => {
                    let slice =
                        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), *len) };
                    slice.fill(1.0);
                }
                MpsInner::Accelerated {
                    buffer,
                    len,
                    dtype: DType::U32,
                    ..
                } => {
                    let slice =
                        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u32>(), *len) };
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
                storage => Self {
                    inner: MpsInner::Cpu(storage),
                },
            }
        }

        pub fn into_cpu(self) -> CpuStorage {
            match self.inner {
                MpsInner::Accelerated {
                    ctx,
                    buffer,
                    len,
                    dtype: DType::F32,
                } => {
                    ctx.synchronize();
                    let slice = unsafe {
                        std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len)
                    };
                    CpuStorage::F32(slice.to_vec())
                }
                MpsInner::Accelerated {
                    ctx,
                    buffer,
                    len,
                    dtype: DType::U32,
                } => {
                    ctx.synchronize();
                    let slice = unsafe {
                        std::slice::from_raw_parts(buffer.contents().cast::<u32>(), len)
                    };
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
                MpsInner::Accelerated {
                    ctx,
                    buffer,
                    len,
                    dtype: inner_dtype,
                } if *inner_dtype == dtype => Some((ctx, buffer, *len)),
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
                let params = ctx.param_buffer(&params);
                ctx.dispatch_1d("scalar_powf_f32", l.size(), |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(&out), 0);
                    encoder.set_buffer(2, Some(&params), 0);
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
                    let meta = ctx.param_buffer(&Self::strided_meta(l));
                    ctx.dispatch_1d(kernel, l.size(), |encoder| {
                        encoder.set_buffer(0, Some(input), 0);
                        encoder.set_buffer(1, Some(&out), 0);
                        encoder.set_buffer(2, Some(&meta), 0);
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
                    let params = ctx.param_buffer(&params);
                    ctx.dispatch_1d(kernel, l.size(), |encoder| {
                        encoder.set_buffer(0, Some(input), 0);
                        encoder.set_buffer(1, Some(&out), 0);
                        encoder.set_buffer(2, Some(&params), 0);
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
                    let meta = ctx.param_buffer(&meta);
                    ctx.dispatch_1d(kernel, layout.size(), |encoder| {
                        encoder.set_buffer(0, Some(lhs), 0);
                        encoder.set_buffer(1, Some(rhs), 0);
                        encoder.set_buffer(2, Some(&out), 0);
                        encoder.set_buffer(3, Some(&meta), 0);
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

            let inner = self
                .as_cpu_storage()
                .binary_op::<O>(layout, &other.as_cpu_storage(), other_layout)?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
            if let (Some((ctx, input, _)), Some((_, output, out_len))) =
                (self.accelerated(DType::F32), dst.accelerated(DType::F32))
            {
                assert!(layout.is_compact());
                let reduce_size = layout.size() / out_len;
                let meta = ReduceMeta {
                    outer_size: out_len as u32,
                    reduce_size: reduce_size as u32,
                };
                let meta = ctx.param_buffer(&meta);
                let kernel = match O::KERNEL {
                    "reduce_sum" => "reduce_sum_f32",
                    "reduce_max" => "reduce_max_f32",
                    _ => unreachable!(),
                };
                ctx.dispatch_1d(kernel, out_len, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(output), 0);
                    encoder.set_buffer(2, Some(&meta), 0);
                });
                return Ok(());
            }

            self.as_cpu_storage()
                .reduce::<O>(layout, &mut dst.clone().into_cpu())?;
            *dst = Self::from_cpu_storage(dst.clone().into_cpu());
            Ok(())
        }

        fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
            if let (Some((ctx, lhs, _)), Some((_, rhs, _))) =
                (self.accelerated(DType::F32), other.accelerated(DType::F32))
            {
                assert!(layout.is_compact() && layout_other.is_compact());
                let n = layout.shape()[layout.ndim() - 1];
                let m = layout.size() / n;
                let p = layout_other.size() / n;
                let out = ctx.empty_f32_buffer(m * p);
                let meta = MatmulMeta {
                    m: m as u32,
                    n: n as u32,
                    p: p as u32,
                };
                let meta = ctx.param_buffer(&meta);
                ctx.dispatch_2d("matmul_f32", p, m, |encoder| {
                    encoder.set_buffer(0, Some(lhs), 0);
                    encoder.set_buffer(1, Some(rhs), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    encoder.set_buffer(3, Some(&meta), 0);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: m * p,
                        dtype: DType::F32,
                    },
                });
            }

            let inner = self
                .as_cpu_storage()
                .matmul(layout, &other.as_cpu_storage(), layout_other)?;
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
                let meta = GatherMeta {
                    rows: rows as u32,
                    cols: cols as u32,
                };
                let meta = ctx.param_buffer(&meta);
                ctx.dispatch_1d("gather_f32", rows, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    encoder.set_buffer(3, Some(&meta), 0);
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

            let inner = self
                .as_cpu_storage()
                .gather(layout, dim, &indices.as_cpu_storage(), indices_layout)?;
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
                let meta = GatherMeta {
                    rows: rows as u32,
                    cols: cols as u32,
                };
                let meta = ctx.param_buffer(&meta);
                ctx.dispatch_1d("scatter_f32", rows, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    encoder.set_buffer(3, Some(&meta), 0);
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

            let inner = self
                .as_cpu_storage()
                .scatter(layout, dim, &indices.as_cpu_storage(), indices_layout, full_shape)?;
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
                MpsInner::Accelerated {
                    ctx,
                    buffer,
                    len,
                    dtype: DType::F32,
                } => {
                    ctx.synchronize();
                    let data = read_f32(ctx, buffer, *len, layout);
                    D::to_vec(&CpuStorage::F32(data))
                }
                MpsInner::Accelerated {
                    ctx,
                    buffer,
                    len,
                    dtype: DType::U32,
                } => {
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
                let meta = ctx.param_buffer(&Self::strided_meta(src_layout));
                ctx.dispatch_1d("copy_compact_f32", out_len, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(output), 0);
                    encoder.set_buffer(2, Some(&meta), 0);
                });
                return Ok(());
            }

            if let (Some((ctx, input, _)), Some((_, output, out_len))) =
                (self.accelerated(DType::U32), dst.accelerated(DType::U32))
            {
                assert_eq!(src_layout.size(), out_len);
                let meta = ctx.param_buffer(&Self::strided_meta(src_layout));
                ctx.dispatch_1d("copy_compact_u32", out_len, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(output), 0);
                    encoder.set_buffer(2, Some(&meta), 0);
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
            MpsInner::Accelerated { buffer, len, .. } => unsafe {
                std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len)
            }
            .to_vec(),
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
            MpsInner::Accelerated { buffer, len, .. } => unsafe {
                std::slice::from_raw_parts(buffer.contents().cast::<u32>(), len)
            }
            .to_vec(),
            MpsInner::Cpu(_) => unreachable!(),
        }
    }

}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::*;

    /// Non-macOS fallback that keeps the backend/API shape intact.
    #[derive(Debug, Clone)]
    pub struct MpsStorage {
        inner: CpuStorage,
    }

    impl MpsStorage {
        pub fn from_cpu_storage(inner: CpuStorage) -> Self {
            Self { inner }
        }

        pub fn into_cpu(self) -> CpuStorage {
            self.inner
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
            Ok(Self::from_cpu_storage(self.inner.ewise_powf(e, l)?))
        }

        fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
            Ok(Self::from_cpu_storage(self.inner.unary_op(op, l)?))
        }

        fn binary_op<O: BinaryOp>(
            &self,
            layout: &Layout,
            other: &Self,
            other_layout: &Layout,
        ) -> Result<Self> {
            Ok(Self::from_cpu_storage(
                self.inner
                    .binary_op::<O>(layout, &other.inner, other_layout)?,
            ))
        }

        fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
            self.inner.reduce::<O>(layout, &mut dst.inner)
        }

        fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
            Ok(Self::from_cpu_storage(
                self.inner.matmul(layout, &other.inner, layout_other)?,
            ))
        }

        fn gather(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
        ) -> Result<Self> {
            Ok(Self::from_cpu_storage(
                self.inner.gather(layout, dim, &indices.inner, indices_layout)?,
            ))
        }

        fn scatter(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
            full_shape: &[usize],
        ) -> Result<Self> {
            Ok(Self::from_cpu_storage(
                self.inner
                    .scatter(layout, dim, &indices.inner, indices_layout, full_shape)?,
            ))
        }

        fn dtype(&self) -> DType {
            self.inner.dtype()
        }

        fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
            self.inner.to_vec(layout)
        }

        fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()> {
            self.inner.copy_compact(src_layout, &mut dst.inner)
        }
    }
}

pub use imp::MpsStorage;
