//! Apple Metal (MPS) tensor storage with pre-compiled shaders and tiled
//! matmul kernels.

use std::borrow::Borrow;

use half::f16;

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
struct GatherScatterMeta {
    left_len: u32,
    src_dim: u32,
    dst_dim: u32,
    right_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IndexSelectMeta {
    left_len: u32,
    index_len: u32,
    src_dim: u32,
    right_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IndexAddMeta {
    left_len: u32,
    src_dim: u32,
    dst_dim: u32,
    right_len: u32,
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use crate::profiler;
    use metal::{
        Buffer, CommandBuffer, CommandQueue, CompileOptions, ComputePipelineState,
        CounterSampleBuffer, Device, MTLCounterSamplingPoint, MTLResourceOptions, MTLSize, NSRange,
    };
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::fmt;
    use std::sync::{Arc, OnceLock};

    const KERNELS: &str = include_str!("kernels.metal");
    // Metal caps this counter sample buffer path at 32 KiB. Timestamp samples
    // resolve to one u64 each, so 4096 samples is the practical maximum.
    const MAX_PROFILE_SAMPLES: u64 = 4_096;

    struct PendingProfileSample {
        event_id: usize,
        start_idx: u64,
        end_idx: u64,
    }

    struct CommandBufferProfile {
        counter_sample_buffer: CounterSampleBuffer,
        resolved_sample_buffer: Buffer,
        cpu_start: u64,
        gpu_start: u64,
        next_sample: u64,
        samples: Vec<PendingProfileSample>,
    }

    impl CommandBufferProfile {
        fn try_new(ctx: &MpsContext) -> Option<Self> {
            if !profiler::is_active() {
                return None;
            }
            if !ctx.device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
                return None;
            }

            let counter_sets = ctx.device.counter_sets();
            let timestamp_counter_set =
                counter_sets.iter().find(|set| set.name() == "timestamp")?;

            let descriptor = metal::CounterSampleBufferDescriptor::new();
            descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
            descriptor.set_sample_count(MAX_PROFILE_SAMPLES);
            descriptor.set_counter_set(&timestamp_counter_set);

            let counter_sample_buffer =
                ctx.device.new_counter_sample_buffer_with_descriptor(&descriptor).ok()?;
            let resolved_sample_buffer = ctx.device.new_buffer(
                (MAX_PROFILE_SAMPLES as usize * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let mut cpu_start = 0;
            let mut gpu_start = 0;
            ctx.device.sample_timestamps(&mut cpu_start, &mut gpu_start);

            Some(Self {
                counter_sample_buffer,
                resolved_sample_buffer,
                cpu_start,
                gpu_start,
                next_sample: 0,
                samples: Vec::new(),
            })
        }

        fn reserve_sample(&mut self) -> Option<(u64, u64)> {
            let event_id = profiler::current_scope_id()?;
            if self.next_sample + 2 > MAX_PROFILE_SAMPLES {
                return None;
            }

            let start_idx = self.next_sample;
            let end_idx = start_idx + 1;
            self.next_sample += 2;
            self.samples.push(PendingProfileSample { event_id, start_idx, end_idx });
            Some((start_idx, end_idx))
        }

        fn has_samples(&self) -> bool {
            !self.samples.is_empty()
        }

        fn resolve_pending_samples(&self, command_buffer: &CommandBuffer) {
            let blit = command_buffer.new_blit_command_encoder();
            blit.resolve_counters(
                &self.counter_sample_buffer,
                NSRange::new(0, self.next_sample),
                &self.resolved_sample_buffer,
                0,
            );
            blit.end_encoding();
        }

        fn record_elapsed_times(&self, ctx: &MpsContext) {
            let mut cpu_end = 0;
            let mut gpu_end = 0;
            ctx.device.sample_timestamps(&mut cpu_end, &mut gpu_end);
            let cpu_span = cpu_end.saturating_sub(self.cpu_start);
            let gpu_span = gpu_end.saturating_sub(self.gpu_start);
            if cpu_span == 0 || gpu_span == 0 {
                return;
            }

            let samples = unsafe {
                std::slice::from_raw_parts(
                    self.resolved_sample_buffer.contents().cast::<u64>(),
                    self.next_sample as usize,
                )
            };
            for sample in &self.samples {
                let begin = samples[sample.start_idx as usize];
                let end = samples[sample.end_idx as usize];
                if end <= begin {
                    continue;
                }

                let gpu_delta = end - begin;
                let elapsed_ns =
                    ((gpu_delta as f64 / gpu_span as f64) * cpu_span as f64).round() as u64;
                profiler::record_device_time(sample.event_id, elapsed_ns);
            }
        }
    }

    struct ActiveCommandBuffer {
        command_buffer: CommandBuffer,
        profile: Option<CommandBufferProfile>,
    }

    impl ActiveCommandBuffer {
        fn new(ctx: &MpsContext) -> Self {
            let command_buffer = ctx.queue.new_command_buffer().to_owned();
            let profile = CommandBufferProfile::try_new(ctx);
            Self { command_buffer, profile }
        }

        fn new_compute_encoder(&mut self) -> &metal::ComputeCommandEncoderRef {
            let Some(profile) = self.profile.as_mut() else {
                return self.command_buffer.new_compute_command_encoder();
            };
            let Some((start_idx, end_idx)) = profile.reserve_sample() else {
                return self.command_buffer.new_compute_command_encoder();
            };

            let descriptor = metal::ComputePassDescriptor::new();
            let attachment = descriptor.sample_buffer_attachments().object_at(0).unwrap();
            attachment.set_sample_buffer(&profile.counter_sample_buffer);
            attachment.set_start_of_encoder_sample_index(start_idx);
            attachment.set_end_of_encoder_sample_index(end_idx);
            self.command_buffer.compute_command_encoder_with_descriptor(descriptor)
        }

        fn commit_and_wait(self, ctx: &MpsContext) {
            if let Some(profile) = self.profile.as_ref().filter(|profile| profile.has_samples()) {
                profile.resolve_pending_samples(&self.command_buffer);
            }

            self.command_buffer.commit();
            self.command_buffer.wait_until_completed();

            if let Some(profile) = self.profile.as_ref().filter(|profile| profile.has_samples()) {
                profile.record_elapsed_times(ctx);
            }
        }
    }

    thread_local! {
        static ACTIVE_COMMAND_BUFFER: RefCell<Option<ActiveCommandBuffer>> = const { RefCell::new(None) };
    }

    const ALL_KERNELS: &[&str] = &[
        "neg_f16",
        "neg_f32",
        "exp_f16",
        "exp_f32",
        "log_f16",
        "log_f32",
        "sin_f16",
        "sin_f32",
        "cos_f16",
        "cos_f32",
        "tanh_f16",
        "tanh_f32",
        "relu_f16",
        "relu_f32",
        "relu_backward_f16",
        "relu_backward_f32",
        "scalar_add_f16",
        "scalar_add_f32",
        "scalar_mul_f16",
        "scalar_mul_f32",
        "scalar_div_f16",
        "scalar_div_f32",
        "scalar_powf_f16",
        "scalar_powf_f32",
        "copy_compact_f16",
        "copy_compact_f32",
        "copy_compact_i64",
        "add_f16",
        "add_f32",
        "sub_f16",
        "sub_f32",
        "mul_f16",
        "mul_f32",
        "div_f16",
        "div_f32",
        "pow_f16",
        "pow_f32",
        "eq_f16",
        "eq_f32",
        "reduce_sum_f16",
        "reduce_sum_f32",
        "reduce_max_f16",
        "reduce_max_f32",
        "matmul_f16",
        "matmul_f32",
        "gather_f16",
        "gather_f32",
        "scatter_add_f16",
        "scatter_add_f32",
        "index_select_f16",
        "index_select_f32",
        "index_add_f16",
        "index_add_f32",
        "reduce_sum_par_f16",
        "reduce_sum_par_f32",
        "reduce_max_par_f16",
        "reduce_max_par_f32",
        "matmul_big_f16",
        "matmul_big_f32",
        "matmul_xl_f16",
        "matmul_xl_f32",
        "log_sum_exp_f16",
        "log_sum_exp_f32",
    ];

    #[derive(Debug)]
    struct MpsContext {
        device: Device,
        queue: CommandQueue,
        pipelines: HashMap<&'static str, ComputePipelineState>,
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

            Self { device, queue, pipelines }
        }

        fn with_command_buffer<R>(&self, f: impl FnOnce(&mut ActiveCommandBuffer) -> R) -> R {
            ACTIVE_COMMAND_BUFFER.with(|slot| {
                let mut guard = slot.borrow_mut();
                if guard.is_none() {
                    *guard = Some(ActiveCommandBuffer::new(self));
                }
                let cb = guard.as_mut().unwrap();
                f(cb)
            })
        }

        fn clear_command_buffer(&self) -> Option<ActiveCommandBuffer> {
            ACTIVE_COMMAND_BUFFER.with(|slot| slot.borrow_mut().take())
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

        fn buffer_from_f16(&self, data: &[f16]) -> Buffer {
            let byte_len = std::mem::size_of_val(data) as u64;
            self.device.new_buffer_with_data(
                data.as_ptr().cast::<c_void>(),
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
        }

        fn buffer_from_i64(&self, data: &[i64]) -> Buffer {
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

        fn empty_f16_buffer(&self, len: usize) -> Buffer {
            self.device.new_buffer(
                (len * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        fn empty_i64_buffer(&self, len: usize) -> Buffer {
            self.device.new_buffer(
                (len * std::mem::size_of::<i64>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        fn flush_command_buffer(&self) {
            if let Some(command_buffer) = self.clear_command_buffer() {
                command_buffer.commit_and_wait(self);
            }
        }

        fn synchronize(&self) {
            let has_active = ACTIVE_COMMAND_BUFFER.with(|slot| slot.borrow().is_some());
            if has_active {
                self.flush_command_buffer();
                return;
            }

            let command_buffer = self.queue.new_command_buffer();
            command_buffer.commit();
            command_buffer.wait_until_completed();
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
            self.with_command_buffer(|active| {
                let encoder = active.new_compute_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                configure(&encoder);
                let width = total_threads.max(1) as u64;
                let tg = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
                encoder.dispatch_threads(MTLSize::new(width, 1, 1), MTLSize::new(tg.max(1), 1, 1));
                encoder.end_encoding();
            });
        }

        /// Dispatches one threadgroup per output row for parallel reductions.
        fn dispatch_reduce(
            &self,
            pipeline_name: &'static str,
            outer_size: usize,
            configure: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) {
            const REDUCE_THREADS: u64 = 256;
            let pipeline = self.pipeline(pipeline_name);
            self.with_command_buffer(|active| {
                let encoder = active.new_compute_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                configure(&encoder);
                let groups = MTLSize::new(outer_size as u64, 1, 1);
                let tg_size = MTLSize::new(REDUCE_THREADS, 1, 1);
                encoder.dispatch_thread_groups(groups, tg_size);
                encoder.end_encoding();
            });
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
            self.with_command_buffer(|active| {
                let encoder = active.new_compute_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                configure(&encoder);
                encoder.dispatch_threads(
                    MTLSize::new(width as u64, height as u64, depth as u64),
                    MTLSize::new(16, 16, 1),
                );
                encoder.end_encoding();
            });
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
        pub(crate) fn synchronize_profiler() {
            MpsContext::shared().flush_command_buffer();
        }

        pub fn empty(len: usize, dtype: DType) -> Self {
            let ctx = MpsContext::shared();
            let buffer = match dtype {
                DType::F16 => ctx.empty_f16_buffer(len),
                DType::F32 => ctx.empty_f32_buffer(len),
                DType::I64 => ctx.empty_i64_buffer(len),
            };
            Self { inner: MpsInner::Accelerated { ctx, buffer, len, dtype } }
        }

        pub fn zeros(len: usize, dtype: DType) -> Self {
            let storage = Self::empty(len, dtype);
            let (buffer, byte_len) = match &storage.inner {
                MpsInner::Accelerated { buffer, len, dtype, .. } => {
                    let elem_size = match dtype {
                        DType::F16 => std::mem::size_of::<f16>(),
                        DType::F32 => std::mem::size_of::<f32>(),
                        DType::I64 => std::mem::size_of::<i64>(),
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
                MpsInner::Accelerated { buffer, len, dtype: DType::F16, .. } => {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer.contents().cast::<f16>(), *len)
                    };
                    slice.fill(f16::from_f32(1.0));
                }
                MpsInner::Accelerated { buffer, len, dtype: DType::F32, .. } => {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), *len)
                    };
                    slice.fill(1.0);
                }
                MpsInner::Accelerated { buffer, len, dtype: DType::I64, .. } => {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer.contents().cast::<i64>(), *len)
                    };
                    slice.fill(1);
                }
                MpsInner::Cpu(_) => unreachable!(),
            }
            storage
        }

        pub fn from_cpu_storage(inner: CpuStorage) -> Self {
            match inner {
                CpuStorage::F16(data) => {
                    let ctx = MpsContext::shared();
                    let buffer = ctx.buffer_from_f16(&data);
                    Self {
                        inner: MpsInner::Accelerated {
                            ctx,
                            buffer,
                            len: data.len(),
                            dtype: DType::F16,
                        },
                    }
                }
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
                CpuStorage::I64(data) => {
                    let ctx = MpsContext::shared();
                    let buffer = ctx.buffer_from_i64(&data);
                    Self {
                        inner: MpsInner::Accelerated {
                            ctx,
                            buffer,
                            len: data.len(),
                            dtype: DType::I64,
                        },
                    }
                }
            }
        }

        /// Concatenates compact MPS storages into a single output buffer.
        pub fn cat(parts: &[(&MpsStorage, usize)]) -> MpsStorage {
            assert!(!parts.is_empty());
            let total_len: usize = parts.iter().map(|(_, len)| *len).sum();
            let dtype = match &parts[0].0.inner {
                MpsInner::Accelerated { dtype, .. } => *dtype,
                MpsInner::Cpu(_) => panic!("cat requires accelerated MPS storage"),
            };
            let ctx = MpsContext::shared();
            let elem_size = match dtype {
                DType::F16 => std::mem::size_of::<f16>(),
                DType::F32 => std::mem::size_of::<f32>(),
                DType::I64 => std::mem::size_of::<i64>(),
            };
            let out_buffer = match dtype {
                DType::F16 => ctx.empty_f16_buffer(total_len),
                DType::F32 => ctx.empty_f32_buffer(total_len),
                DType::I64 => ctx.empty_i64_buffer(total_len),
            };
            ctx.with_command_buffer(|active| {
                let blit = active.command_buffer.new_blit_command_encoder();
                let mut byte_offset = 0usize;
                for (storage, len) in parts {
                    let src_buffer = match &storage.inner {
                        MpsInner::Accelerated { buffer, .. } => buffer,
                        MpsInner::Cpu(_) => panic!("cat requires accelerated MPS storage"),
                    };
                    let byte_len = len * elem_size;
                    blit.copy_from_buffer(
                        src_buffer,
                        0,
                        &out_buffer,
                        byte_offset as u64,
                        byte_len as u64,
                    );
                    byte_offset += byte_len;
                }
                blit.end_encoding();
            });
            MpsStorage {
                inner: MpsInner::Accelerated { ctx, buffer: out_buffer, len: total_len, dtype },
            }
        }

        pub fn into_cpu(self) -> CpuStorage {
            match self.inner {
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::F16 } => {
                    ctx.synchronize();
                    let slice =
                        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f16>(), len) };
                    CpuStorage::F16(slice.to_vec())
                }
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::F32 } => {
                    ctx.synchronize();
                    let slice =
                        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len) };
                    CpuStorage::F32(slice.to_vec())
                }
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::I64 } => {
                    ctx.synchronize();
                    let slice =
                        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<i64>(), len) };
                    CpuStorage::I64(slice.to_vec())
                }
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

        fn unary_kernel_name<O: UnaryOp>(dtype: DType) -> Option<&'static str> {
            match (O::KERNEL, dtype) {
                ("Neg", DType::F16) => Some("neg_f16"),
                ("Neg", DType::F32) => Some("neg_f32"),
                ("exp", DType::F16) => Some("exp_f16"),
                ("exp", DType::F32) => Some("exp_f32"),
                ("log", DType::F16) => Some("log_f16"),
                ("log", DType::F32) => Some("log_f32"),
                ("sin", DType::F16) => Some("sin_f16"),
                ("sin", DType::F32) => Some("sin_f32"),
                ("cos", DType::F16) => Some("cos_f16"),
                ("cos", DType::F32) => Some("cos_f32"),
                ("tanh", DType::F16) => Some("tanh_f16"),
                ("tanh", DType::F32) => Some("tanh_f32"),
                ("relu", DType::F16) => Some("relu_f16"),
                ("relu", DType::F32) => Some("relu_f32"),
                ("relu_backward", DType::F16) => Some("relu_backward_f16"),
                ("relu_backward", DType::F32) => Some("relu_backward_f32"),
                _ => None,
            }
        }

        fn scalar_kernel_name<O: UnaryOp>(dtype: DType) -> Option<&'static str> {
            match (O::KERNEL, dtype) {
                ("scalar_add", DType::F16) => Some("scalar_add_f16"),
                ("scalar_add", DType::F32) => Some("scalar_add_f32"),
                ("scalar_mul", DType::F16) => Some("scalar_mul_f16"),
                ("scalar_mul", DType::F32) => Some("scalar_mul_f32"),
                ("scalar_div", DType::F16) => Some("scalar_div_f16"),
                ("scalar_div", DType::F32) => Some("scalar_div_f32"),
                _ => None,
            }
        }

        fn binary_kernel_name<O: BinaryOp>(dtype: DType) -> Option<&'static str> {
            match (O::KERNEL, dtype) {
                ("add", DType::F16) => Some("add_f16"),
                ("add", DType::F32) => Some("add_f32"),
                ("sub", DType::F16) => Some("sub_f16"),
                ("sub", DType::F32) => Some("sub_f32"),
                ("mul", DType::F16) => Some("mul_f16"),
                ("mul", DType::F32) => Some("mul_f32"),
                ("div", DType::F16) => Some("div_f16"),
                ("div", DType::F32) => Some("div_f32"),
                ("powf", DType::F16) => Some("pow_f16"),
                ("powf", DType::F32) => Some("pow_f32"),
                ("eq", DType::F16) => Some("eq_f16"),
                ("eq", DType::F32) => Some("eq_f32"),
                _ => None,
            }
        }

        /// Selects matmul kernel and tile size based on problem dimensions.
        fn matmul_config(batch: usize, m: usize, n: usize, suffix: &str) -> (&'static str, u64) {
            let output_size = batch * m * n;
            if output_size >= (1 << 20) || (m >= 64 && n >= 64) {
                return (if suffix == "f32" { "matmul_xl_f32" } else { "matmul_xl_f16" }, 64);
            }
            if output_size >= (1 << 15) {
                return (if suffix == "f32" { "matmul_big_f32" } else { "matmul_big_f16" }, 32);
            }
            (if suffix == "f32" { "matmul_f32" } else { "matmul_f16" }, 16)
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

    pub(crate) fn has_active_command_buffer() -> bool {
        ACTIVE_COMMAND_BUFFER.with(|slot| slot.borrow().is_some())
    }

    impl From<Vec<f16>> for MpsStorage {
        fn from(value: Vec<f16>) -> Self {
            Self::from_cpu_storage(CpuStorage::from(value))
        }
    }

    impl From<Vec<f32>> for MpsStorage {
        fn from(value: Vec<f32>) -> Self {
            Self::from_cpu_storage(CpuStorage::from(value))
        }
    }

    impl From<Vec<i64>> for MpsStorage {
        fn from(value: Vec<i64>) -> Self {
            Self::from_cpu_storage(CpuStorage::from(value))
        }
    }

    impl BackendStorage for MpsStorage {
        fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
            if let Some((ctx, input, _)) = self.accelerated(DType::F16) {
                let out = ctx.empty_f16_buffer(l.size());
                let params = ScalarMeta {
                    input: Self::strided_meta(l),
                    scalar: e as f32,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                };
                ctx.dispatch_1d("scalar_powf_f16", l.size(), |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(&out), 0);
                    MpsContext::set_params(encoder, 2, &params);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: l.size(),
                        dtype: DType::F16,
                    },
                });
            }

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
            if let Some((ctx, input, _)) = self.accelerated(DType::F16) {
                if let Some(kernel) = Self::unary_kernel_name::<O>(DType::F16) {
                    let out = ctx.empty_f16_buffer(l.size());
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
                            dtype: DType::F16,
                        },
                    });
                }

                if let Some(kernel) = Self::scalar_kernel_name::<O>(DType::F16) {
                    let scalar = match O::KERNEL {
                        "scalar_add" => op.f16(f16::from_f32(0.0)).to_f32(),
                        "scalar_mul" => op.f16(f16::from_f32(1.0)).to_f32(),
                        "scalar_div" => 1.0 / op.f16(f16::from_f32(1.0)).to_f32(),
                        _ => unreachable!(),
                    };
                    let out = ctx.empty_f16_buffer(l.size());
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
                            dtype: DType::F16,
                        },
                    });
                }
            }

            if let Some((ctx, input, _)) = self.accelerated(DType::F32) {
                if let Some(kernel) = Self::unary_kernel_name::<O>(DType::F32) {
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

                if let Some(kernel) = Self::scalar_kernel_name::<O>(DType::F32) {
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
                (self.accelerated(DType::F16), other.accelerated(DType::F16))
                && let Some(kernel) = Self::binary_kernel_name::<O>(DType::F16)
            {
                let out = ctx.empty_f16_buffer(layout.size());
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
                        dtype: DType::F16,
                    },
                });
            }

            if let (Some((ctx, lhs, _)), Some((_, rhs, _))) =
                (self.accelerated(DType::F32), other.accelerated(DType::F32))
                && let Some(kernel) = Self::binary_kernel_name::<O>(DType::F32)
            {
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

            let inner = self.as_cpu_storage().binary_op::<O>(
                layout,
                &other.as_cpu_storage(),
                other_layout,
            )?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
            // Use parallel threadgroup reduce for large reduce dimensions,
            // serial for small ones where threadgroup overhead dominates.
            const PARALLEL_THRESHOLD: usize = 1024;

            if let (Some((ctx, input, _)), Some((_, output, out_len))) =
                (self.accelerated(DType::F16), dst.accelerated(DType::F16))
            {
                assert!(layout.is_compact());
                let reduce_size = layout.size() / out_len;
                let meta =
                    ReduceMeta { outer_size: out_len as u32, reduce_size: reduce_size as u32 };
                if reduce_size >= PARALLEL_THRESHOLD {
                    let kernel = match O::KERNEL {
                        "reduce_sum" => "reduce_sum_par_f16",
                        "reduce_max" => "reduce_max_par_f16",
                        _ => unreachable!(),
                    };
                    ctx.dispatch_reduce(kernel, out_len, |encoder| {
                        encoder.set_buffer(0, Some(input), 0);
                        encoder.set_buffer(1, Some(output), 0);
                        MpsContext::set_params(encoder, 2, &meta);
                    });
                } else {
                    let kernel = match O::KERNEL {
                        "reduce_sum" => "reduce_sum_f16",
                        "reduce_max" => "reduce_max_f16",
                        _ => unreachable!(),
                    };
                    ctx.dispatch_1d(kernel, out_len, |encoder| {
                        encoder.set_buffer(0, Some(input), 0);
                        encoder.set_buffer(1, Some(output), 0);
                        MpsContext::set_params(encoder, 2, &meta);
                    });
                }
                return Ok(());
            }

            if let (Some((ctx, input, _)), Some((_, output, out_len))) =
                (self.accelerated(DType::F32), dst.accelerated(DType::F32))
            {
                assert!(layout.is_compact());
                let reduce_size = layout.size() / out_len;
                let meta =
                    ReduceMeta { outer_size: out_len as u32, reduce_size: reduce_size as u32 };
                if reduce_size >= PARALLEL_THRESHOLD {
                    let kernel = match O::KERNEL {
                        "reduce_sum" => "reduce_sum_par_f32",
                        "reduce_max" => "reduce_max_par_f32",
                        _ => unreachable!(),
                    };
                    ctx.dispatch_reduce(kernel, out_len, |encoder| {
                        encoder.set_buffer(0, Some(input), 0);
                        encoder.set_buffer(1, Some(output), 0);
                        MpsContext::set_params(encoder, 2, &meta);
                    });
                } else {
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
                }
                return Ok(());
            }

            self.as_cpu_storage().reduce::<O>(layout, &mut dst.clone().into_cpu())?;
            *dst = Self::from_cpu_storage(dst.clone().into_cpu());
            Ok(())
        }

        fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
            if let (Some((ctx, lhs, _)), Some((_, rhs, _))) =
                (self.accelerated(DType::F16), other.accelerated(DType::F16))
            {
                assert!(layout.is_compact() && layout_other.is_compact());
                let ndim = layout.ndim();
                let m = layout.shape()[ndim - 2];
                let k = layout.shape()[ndim - 1];
                let n = layout_other.shape()[ndim - 1];
                let batch: usize = (0..ndim - 2).map(|i| layout.shape()[i]).product();
                let total = batch * m * n;
                let out = ctx.empty_f16_buffer(total);
                let meta =
                    MatmulMeta { m: m as u32, k: k as u32, n: n as u32, batch: batch as u32 };
                let (pipeline_name, tile) = Self::matmul_config(batch, m, n, "f16");
                let pipeline = ctx.pipeline(pipeline_name);
                ctx.with_command_buffer(|active| {
                    let encoder = active.new_compute_encoder();
                    encoder.set_compute_pipeline_state(&pipeline);
                    encoder.set_buffer(0, Some(lhs), 0);
                    encoder.set_buffer(1, Some(rhs), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(&encoder, 3, &meta);
                    let groups = MTLSize::new(
                        (n as u64 + tile - 1) / tile,
                        (m as u64 + tile - 1) / tile,
                        batch as u64,
                    );
                    let tg_size = MTLSize::new(16, 16, 1);
                    encoder.dispatch_thread_groups(groups, tg_size);
                    encoder.end_encoding();
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: total,
                        dtype: DType::F16,
                    },
                });
            }

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
                let (pipeline_name, tile) = Self::matmul_config(batch, m, n, "f32");
                let pipeline = ctx.pipeline(pipeline_name);
                ctx.with_command_buffer(|active| {
                    let encoder = active.new_compute_encoder();
                    encoder.set_compute_pipeline_state(&pipeline);
                    encoder.set_buffer(0, Some(lhs), 0);
                    encoder.set_buffer(1, Some(rhs), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(&encoder, 3, &meta);
                    let groups = MTLSize::new(
                        (n as u64 + tile - 1) / tile,
                        (m as u64 + tile - 1) / tile,
                        batch as u64,
                    );
                    let tg_size = MTLSize::new(16, 16, 1);
                    encoder.dispatch_thread_groups(groups, tg_size);
                    encoder.end_encoding();
                });
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
            assert!(layout.is_compact());
            assert!(indices_layout.is_compact());
            assert_eq!(layout.ndim(), indices_layout.ndim(), "gather requires matching ranks");
            assert!(dim < layout.ndim(), "gather dim out of bounds");

            let left_len: usize = indices_layout.shape().iter().take(dim).product();
            let dst_dim = indices_layout.shape()[dim];
            let right_len: usize = indices_layout.shape().iter().skip(dim + 1).product();
            let src_dim = layout.shape()[dim];

            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F16), indices.accelerated(DType::I64))
            {
                let out = ctx.empty_f16_buffer(left_len * dst_dim * right_len);
                let meta = GatherScatterMeta {
                    left_len: left_len as u32,
                    src_dim: src_dim as u32,
                    dst_dim: dst_dim as u32,
                    right_len: right_len as u32,
                };
                ctx.dispatch_2d("gather_f16", right_len, left_len * dst_dim, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: left_len * dst_dim * right_len,
                        dtype: DType::F16,
                    },
                });
            }

            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F32), indices.accelerated(DType::I64))
            {
                let out = ctx.empty_f32_buffer(left_len * dst_dim * right_len);
                let meta = GatherScatterMeta {
                    left_len: left_len as u32,
                    src_dim: src_dim as u32,
                    dst_dim: dst_dim as u32,
                    right_len: right_len as u32,
                };
                ctx.dispatch_2d("gather_f32", right_len, left_len * dst_dim, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: left_len * dst_dim * right_len,
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

        fn scatter_add(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
            dst_shape: &[usize],
        ) -> Result<Self> {
            assert!(layout.is_compact());
            assert!(indices_layout.is_compact());
            assert_eq!(
                layout.ndim(),
                indices_layout.ndim(),
                "scatter_add requires source and indices to have matching ranks"
            );
            assert_eq!(
                layout.ndim(),
                dst_shape.len(),
                "scatter_add requires source and destination to have matching ranks"
            );
            assert!(dim < dst_shape.len(), "scatter_add dim out of bounds");

            let left_len: usize = layout.shape().iter().take(dim).product();
            let src_dim = layout.shape()[dim];
            let dst_dim = dst_shape[dim];
            let right_len: usize = layout.shape().iter().skip(dim + 1).product();
            let total_len: usize = dst_shape.iter().product();

            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F16), indices.accelerated(DType::I64))
            {
                let out = ctx.empty_f16_buffer(total_len);
                unsafe {
                    std::ptr::write_bytes(
                        out.contents(),
                        0,
                        total_len * std::mem::size_of::<f16>(),
                    );
                }
                let meta = GatherScatterMeta {
                    left_len: left_len as u32,
                    src_dim: src_dim as u32,
                    dst_dim: dst_dim as u32,
                    right_len: right_len as u32,
                };
                ctx.dispatch_2d("scatter_add_f16", right_len, left_len * dst_dim, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });

                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: total_len,
                        dtype: DType::F16,
                    },
                });
            }

            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F32), indices.accelerated(DType::I64))
            {
                let out = ctx.empty_f32_buffer(total_len);
                unsafe {
                    std::ptr::write_bytes(
                        out.contents(),
                        0,
                        total_len * std::mem::size_of::<f32>(),
                    );
                }
                let meta = GatherScatterMeta {
                    left_len: left_len as u32,
                    src_dim: src_dim as u32,
                    dst_dim: dst_dim as u32,
                    right_len: right_len as u32,
                };
                ctx.dispatch_2d("scatter_add_f32", right_len, left_len * dst_dim, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });

                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: total_len,
                        dtype: DType::F32,
                    },
                });
            }

            let inner = self.as_cpu_storage().scatter_add(
                layout,
                dim,
                &indices.as_cpu_storage(),
                indices_layout,
                dst_shape,
            )?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn index_select(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
        ) -> Result<Self> {
            assert!(layout.is_compact());
            assert!(indices_layout.is_compact());
            assert_eq!(indices_layout.ndim(), 1, "index_select requires 1D indices");
            assert!(dim < layout.ndim(), "index_select dim out of bounds");

            let index_len = indices_layout.shape()[0];
            let left_len: usize = layout.shape().iter().take(dim).product();
            let src_dim = layout.shape()[dim];
            let right_len: usize = layout.shape().iter().skip(dim + 1).product();

            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F16), indices.accelerated(DType::I64))
            {
                let out = ctx.empty_f16_buffer(left_len * index_len * right_len);
                let meta = IndexSelectMeta {
                    left_len: left_len as u32,
                    index_len: index_len as u32,
                    src_dim: src_dim as u32,
                    right_len: right_len as u32,
                };
                ctx.dispatch_2d("index_select_f16", right_len, left_len * index_len, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: left_len * index_len * right_len,
                        dtype: DType::F16,
                    },
                });
            }

            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F32), indices.accelerated(DType::I64))
            {
                let out = ctx.empty_f32_buffer(left_len * index_len * right_len);
                let meta = IndexSelectMeta {
                    left_len: left_len as u32,
                    index_len: index_len as u32,
                    src_dim: src_dim as u32,
                    right_len: right_len as u32,
                };
                ctx.dispatch_2d("index_select_f32", right_len, left_len * index_len, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: left_len * index_len * right_len,
                        dtype: DType::F32,
                    },
                });
            }

            let inner = self.as_cpu_storage().index_select(
                layout,
                dim,
                &indices.as_cpu_storage(),
                indices_layout,
            )?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn index_add(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
            dst_shape: &[usize],
        ) -> Result<Self> {
            assert!(layout.is_compact());
            assert!(indices_layout.is_compact());
            assert_eq!(indices_layout.ndim(), 1, "index_add requires 1D indices");
            assert!(dim < dst_shape.len(), "index_add dim out of bounds");

            let src_dim = layout.shape()[dim];
            let dst_dim = dst_shape[dim];
            let left_len: usize = dst_shape.iter().take(dim).product();
            let right_len: usize = dst_shape.iter().skip(dim + 1).product();
            let total_len: usize = dst_shape.iter().product();

            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F16), indices.accelerated(DType::I64))
            {
                let out = ctx.empty_f16_buffer(total_len);
                unsafe {
                    std::ptr::write_bytes(
                        out.contents(),
                        0,
                        total_len * std::mem::size_of::<f16>(),
                    );
                }
                let meta = IndexAddMeta {
                    left_len: left_len as u32,
                    src_dim: src_dim as u32,
                    dst_dim: dst_dim as u32,
                    right_len: right_len as u32,
                };
                ctx.dispatch_2d("index_add_f16", right_len, left_len * dst_dim, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: total_len,
                        dtype: DType::F16,
                    },
                });
            }

            if let (Some((ctx, input, _)), Some((_, index_buffer, _))) =
                (self.accelerated(DType::F32), indices.accelerated(DType::I64))
            {
                let out = ctx.empty_f32_buffer(total_len);
                unsafe {
                    std::ptr::write_bytes(
                        out.contents(),
                        0,
                        total_len * std::mem::size_of::<f32>(),
                    );
                }
                let meta = IndexAddMeta {
                    left_len: left_len as u32,
                    src_dim: src_dim as u32,
                    dst_dim: dst_dim as u32,
                    right_len: right_len as u32,
                };
                ctx.dispatch_2d("index_add_f32", right_len, left_len * dst_dim, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(index_buffer), 0);
                    encoder.set_buffer(2, Some(&out), 0);
                    MpsContext::set_params(encoder, 3, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: total_len,
                        dtype: DType::F32,
                    },
                });
            }

            let inner = self.as_cpu_storage().index_add(
                layout,
                dim,
                &indices.as_cpu_storage(),
                indices_layout,
                dst_shape,
            )?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn log_sum_exp(
            &self,
            layout: &Layout,
            outer_size: usize,
            reduce_size: usize,
        ) -> Result<Self> {
            if let Some((ctx, input, _)) = self.accelerated(DType::F16) {
                assert!(layout.is_compact());
                let out = ctx.empty_f16_buffer(outer_size);
                let meta =
                    ReduceMeta { outer_size: outer_size as u32, reduce_size: reduce_size as u32 };
                ctx.dispatch_reduce("log_sum_exp_f16", outer_size, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(&out), 0);
                    MpsContext::set_params(encoder, 2, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: outer_size,
                        dtype: DType::F16,
                    },
                });
            }

            if let Some((ctx, input, _)) = self.accelerated(DType::F32) {
                assert!(layout.is_compact());
                let out = ctx.empty_f32_buffer(outer_size);
                let meta =
                    ReduceMeta { outer_size: outer_size as u32, reduce_size: reduce_size as u32 };
                ctx.dispatch_reduce("log_sum_exp_f32", outer_size, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(&out), 0);
                    MpsContext::set_params(encoder, 2, &meta);
                });
                return Ok(Self {
                    inner: MpsInner::Accelerated {
                        ctx: ctx.clone(),
                        buffer: out,
                        len: outer_size,
                        dtype: DType::F32,
                    },
                });
            }

            let inner = self.as_cpu_storage().log_sum_exp(layout, outer_size, reduce_size)?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn log_softmax_fwd(
            &self,
            layout: &Layout,
            outer_size: usize,
            inner_size: usize,
        ) -> Result<Self> {
            let inner = self.as_cpu_storage().log_softmax_fwd(layout, outer_size, inner_size)?;
            Ok(Self::from_cpu_storage(inner))
        }

        fn log_softmax_bwd(
            &self,
            grad_layout: &Layout,
            lsm: &Self,
            lsm_layout: &Layout,
            outer_size: usize,
            inner_size: usize,
        ) -> Result<Self> {
            let inner = self.as_cpu_storage().log_softmax_bwd(
                grad_layout,
                &lsm.as_cpu_storage(),
                lsm_layout,
                outer_size,
                inner_size,
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
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::F16 } => {
                    ctx.synchronize();
                    let data = read_f16(ctx, buffer, *len, layout);
                    D::to_vec(&CpuStorage::F16(data))
                }
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::F32 } => {
                    ctx.synchronize();
                    let data = read_f32(ctx, buffer, *len, layout);
                    D::to_vec(&CpuStorage::F32(data))
                }
                MpsInner::Accelerated { ctx, buffer, len, dtype: DType::I64 } => {
                    ctx.synchronize();
                    let data = read_i64(ctx, buffer, *len, layout);
                    D::to_vec(&CpuStorage::I64(data))
                }
                MpsInner::Cpu(storage) => storage.to_vec(layout),
            }
        }

        fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()> {
            if let (Some((ctx, input, _)), Some((_, output, out_len))) =
                (self.accelerated(DType::F16), dst.accelerated(DType::F16))
            {
                assert_eq!(src_layout.size(), out_len);
                let meta = Self::strided_meta(src_layout);
                ctx.dispatch_1d("copy_compact_f16", out_len, |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(output), 0);
                    MpsContext::set_params(encoder, 2, &meta);
                });
                return Ok(());
            }

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
                (self.accelerated(DType::I64), dst.accelerated(DType::I64))
            {
                assert_eq!(src_layout.size(), out_len);
                let meta = Self::strided_meta(src_layout);
                ctx.dispatch_1d("copy_compact_i64", out_len, |encoder| {
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

    fn read_f16(ctx: &Arc<MpsContext>, buffer: &Buffer, len: usize, layout: &Layout) -> Vec<f16> {
        if layout.is_compact() && layout.size() == len {
            return unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f16>(), len) }
                .to_vec();
        }

        let tmp = MpsStorage {
            inner: MpsInner::Accelerated {
                ctx: ctx.clone(),
                buffer: buffer.to_owned(),
                len,
                dtype: DType::F16,
            },
        };
        let mut dst = MpsStorage {
            inner: MpsInner::Accelerated {
                ctx: ctx.clone(),
                buffer: ctx.empty_f16_buffer(layout.size()),
                len: layout.size(),
                dtype: DType::F16,
            },
        };
        tmp.copy_compact(layout, &mut dst).unwrap();
        ctx.synchronize();

        match dst.inner {
            MpsInner::Accelerated { buffer, len, .. } => {
                unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f16>(), len) }.to_vec()
            }
            MpsInner::Cpu(_) => unreachable!(),
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

    fn read_i64(ctx: &Arc<MpsContext>, buffer: &Buffer, len: usize, layout: &Layout) -> Vec<i64> {
        if layout.is_compact() && layout.size() == len {
            return unsafe { std::slice::from_raw_parts(buffer.contents().cast::<i64>(), len) }
                .to_vec();
        }

        let tmp = MpsStorage {
            inner: MpsInner::Accelerated {
                ctx: ctx.clone(),
                buffer: buffer.to_owned(),
                len,
                dtype: DType::I64,
            },
        };
        let mut dst = MpsStorage {
            inner: MpsInner::Accelerated {
                ctx: ctx.clone(),
                buffer: ctx.empty_i64_buffer(layout.size()),
                len: layout.size(),
                dtype: DType::I64,
            },
        };
        tmp.copy_compact(layout, &mut dst).unwrap();
        ctx.synchronize();

        match dst.inner {
            MpsInner::Accelerated { buffer, len, .. } => {
                unsafe { std::slice::from_raw_parts(buffer.contents().cast::<i64>(), len) }.to_vec()
            }
            MpsInner::Cpu(_) => unreachable!(),
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn synchronize() {
    if !imp::has_active_command_buffer() {
        return;
    }
    imp::MpsStorage::synchronize_profiler();
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

        pub(crate) fn synchronize_profiler() {}
    }

    impl From<Vec<f16>> for MpsStorage {
        fn from(_: Vec<f16>) -> Self {
            Self::unavailable()
        }
    }

    impl From<Vec<f32>> for MpsStorage {
        fn from(_: Vec<f32>) -> Self {
            Self::unavailable()
        }
    }

    impl From<Vec<i64>> for MpsStorage {
        fn from(_: Vec<i64>) -> Self {
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
        fn scatter_add(
            &self,
            _: &Layout,
            _: usize,
            _: &Self,
            _: &Layout,
            _: &[usize],
        ) -> Result<Self> {
            Self::unavailable()
        }
        fn index_select(&self, _: &Layout, _: usize, _: &Self, _: &Layout) -> Result<Self> {
            Self::unavailable()
        }
        fn index_add(
            &self,
            _: &Layout,
            _: usize,
            _: &Self,
            _: &Layout,
            _: &[usize],
        ) -> Result<Self> {
            Self::unavailable()
        }
        fn log_sum_exp(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
            Self::unavailable()
        }
        fn log_softmax_fwd(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
            Self::unavailable()
        }
        fn log_softmax_bwd(&self, _: &Layout, _: &Self, _: &Layout, _: usize, _: usize) -> Result<Self> {
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

#[cfg(not(target_os = "macos"))]
pub(crate) fn synchronize() {}

pub use imp::MpsStorage;
