use std::borrow::Borrow;

use crate::{
    dtype::{DType, WithDType},
    error::{Error, Result},
    layout::Layout,
    storage::{BackendStorage, BinaryOp, CpuStorage, ReduceOp, UnaryOp},
};

const MAX_DIMS: usize = 8;

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod imp {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Arc, OnceLock};

    use half::f16;

    use crate::profiler;

    use cudarc::{
        cublas::{CudaBlas, result as cublas_result, sys as cublas_sys},
        driver::{
            CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
            DeviceRepr, LaunchConfig, PushKernelArg, sys::CUevent_flags,
        },
        nvrtc,
    };

    const KERNELS: &str = r#"
    #include <cuda_fp16.h>
    #define MAX_DIMS 8
    #define REDUCE_THREADS 256

    typedef struct {
        unsigned int ndim;
        unsigned int offset;
        unsigned int size;
        unsigned int pad;
        unsigned int shape[MAX_DIMS];
        int strides[MAX_DIMS];
    } StridedMeta;

    typedef long long index_t;

    typedef struct {
        float scalar;
    } ScalarMeta;

    __device__ __forceinline__ unsigned int compact_to_strided(unsigned int idx, const StridedMeta* meta) {
        unsigned int src = meta->offset;
        unsigned int rem = idx;
        for (int dim = (int)meta->ndim - 1; dim >= 0; --dim) {
            unsigned int cur = rem % meta->shape[dim];
            rem /= meta->shape[dim];
            src += (unsigned int)((int)cur * meta->strides[dim]);
        }
        return src;
    }

    template <typename T>
    __device__ __forceinline__ T zero_value();
    template <>
    __device__ __forceinline__ float zero_value<float>() { return 0.0f; }
    template <>
    __device__ __forceinline__ half zero_value<half>() { return __float2half(0.0f); }

    template <typename T>
    __device__ __forceinline__ T one_value();
    template <>
    __device__ __forceinline__ float one_value<float>() { return 1.0f; }
    template <>
    __device__ __forceinline__ half one_value<half>() { return __float2half(1.0f); }

    template <typename T>
    __device__ __forceinline__ float to_float(T v);
    template <>
    __device__ __forceinline__ float to_float<float>(float v) { return v; }
    template <>
    __device__ __forceinline__ float to_float<half>(half v) { return __half2float(v); }

    template <typename T>
    __device__ __forceinline__ T from_float(float v);
    template <>
    __device__ __forceinline__ float from_float<float>(float v) { return v; }
    template <>
    __device__ __forceinline__ half from_float<half>(float v) { return __float2half(v); }

    template <typename T>
    __device__ __forceinline__ void atomic_add_t(T* dst, T v);
    template <>
    __device__ __forceinline__ void atomic_add_t<float>(float* dst, float v) { atomicAdd(dst, v); }
    template <>
    __device__ __forceinline__ void atomic_add_t<half>(half* dst, half v) { atomicAdd(dst, v); }

    template <typename T>
    __global__ void copy_compact_kernel(const T* src, T* dst, StridedMeta meta) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < meta.size) {
            dst[idx] = src[compact_to_strided(idx, &meta)];
        }
    }

    #define DEFINE_UNARY_F32(name, expr) \
    extern "C" __global__ void name##_f32(const float* src, float* dst, unsigned int size) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { float x = src[idx]; dst[idx] = (expr); } \
    }

    #define DEFINE_UNARY_F16(name, expr) \
    extern "C" __global__ void name##_f16(const half* src, half* dst, unsigned int size) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { float x = __half2float(src[idx]); dst[idx] = __float2half((expr)); } \
    }

    DEFINE_UNARY_F32(neg, -x)
    DEFINE_UNARY_F16(neg, -x)
    DEFINE_UNARY_F32(exp, expf(x))
    DEFINE_UNARY_F16(exp, expf(x))
    DEFINE_UNARY_F32(log, logf(x))
    DEFINE_UNARY_F16(log, logf(x))
    DEFINE_UNARY_F32(sin, sinf(x))
    DEFINE_UNARY_F16(sin, sinf(x))
    DEFINE_UNARY_F32(cos, cosf(x))
    DEFINE_UNARY_F16(cos, cosf(x))
    DEFINE_UNARY_F32(tanh, tanhf(x))
    DEFINE_UNARY_F16(tanh, tanhf(x))
    DEFINE_UNARY_F32(relu, fmaxf(x, 0.0f))
    DEFINE_UNARY_F16(relu, fmaxf(x, 0.0f))
    DEFINE_UNARY_F32(relu_backward, x > 0.0f ? 1.0f : 0.0f)
    DEFINE_UNARY_F16(relu_backward, x > 0.0f ? 1.0f : 0.0f)

    #define DEFINE_SCALAR_F32(name, expr) \
    extern "C" __global__ void name##_f32(const float* src, float* dst, unsigned int size, ScalarMeta meta) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { float x = src[idx]; float s = meta.scalar; dst[idx] = (expr); } \
    }

    #define DEFINE_SCALAR_F16(name, expr) \
    extern "C" __global__ void name##_f16(const half* src, half* dst, unsigned int size, ScalarMeta meta) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { float x = __half2float(src[idx]); float s = meta.scalar; dst[idx] = __float2half((expr)); } \
    }

    DEFINE_SCALAR_F32(scalar_add, x + s)
    DEFINE_SCALAR_F16(scalar_add, x + s)
    DEFINE_SCALAR_F32(scalar_mul, x * s)
    DEFINE_SCALAR_F16(scalar_mul, x * s)
    DEFINE_SCALAR_F32(scalar_div, x / s)
    DEFINE_SCALAR_F16(scalar_div, x / s)
    DEFINE_SCALAR_F32(scalar_powf, powf(x, s))
    DEFINE_SCALAR_F16(scalar_powf, powf(x, s))

    #define DEFINE_BINARY_F32(name, expr) \
    extern "C" __global__ void name##_f32(const float* lhs, const float* rhs, float* dst, unsigned int size) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { float x = lhs[idx]; float y = rhs[idx]; dst[idx] = (expr); } \
    }

    #define DEFINE_BINARY_F16(name, expr) \
    extern "C" __global__ void name##_f16(const half* lhs, const half* rhs, half* dst, unsigned int size) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { float x = __half2float(lhs[idx]); float y = __half2float(rhs[idx]); dst[idx] = __float2half((expr)); } \
    }

    DEFINE_BINARY_F32(add, x + y)
    DEFINE_BINARY_F16(add, x + y)
    DEFINE_BINARY_F32(sub, x - y)
    DEFINE_BINARY_F16(sub, x - y)
    DEFINE_BINARY_F32(mul, x * y)
    DEFINE_BINARY_F16(mul, x * y)
    DEFINE_BINARY_F32(div, x / y)
    DEFINE_BINARY_F16(div, x / y)
    DEFINE_BINARY_F32(powf, powf(x, y))
    DEFINE_BINARY_F16(powf, powf(x, y))
    DEFINE_BINARY_F32(eq, x == y ? 1.0f : 0.0f)
    DEFINE_BINARY_F16(eq, x == y ? 1.0f : 0.0f)

    template <typename T>
    __global__ void reduce_sum_kernel(const T* src, T* dst, unsigned int outer_size, unsigned int reduce_size) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        __shared__ float smem[REDUCE_THREADS];
        float acc = 0.0f;
        for (unsigned int col = threadIdx.x; col < reduce_size; col += blockDim.x) {
            acc += to_float(src[row * reduce_size + col]);
        }
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) dst[row] = from_float<T>(smem[0]);
    }

    template <typename T>
    __global__ void reduce_max_kernel(const T* src, T* dst, unsigned int outer_size, unsigned int reduce_size) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        __shared__ float smem[REDUCE_THREADS];
        float acc = -1.0f / 0.0f;
        for (unsigned int col = threadIdx.x; col < reduce_size; col += blockDim.x) {
            acc = fmaxf(acc, to_float(src[row * reduce_size + col]));
        }
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x == 0) dst[row] = from_float<T>(smem[0]);
    }

    template <typename T>
    __global__ void log_sum_exp_kernel(const T* src, T* dst, unsigned int outer_size, unsigned int reduce_size) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        __shared__ float smem[REDUCE_THREADS];
        float row_max = -1.0f / 0.0f;
        for (unsigned int col = threadIdx.x; col < reduce_size; col += blockDim.x) {
            row_max = fmaxf(row_max, to_float(src[row * reduce_size + col]));
        }
        smem[threadIdx.x] = row_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
            __syncthreads();
        }
        row_max = smem[0];

        float acc = 0.0f;
        for (unsigned int col = threadIdx.x; col < reduce_size; col += blockDim.x) {
            acc += expf(to_float(src[row * reduce_size + col]) - row_max);
        }
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) dst[row] = from_float<T>(logf(smem[0]) + row_max);
    }

    extern "C" __global__ void reduce_sum_f32(const float* src, float* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_sum_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void reduce_sum_f16(const half* src, half* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_sum_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void reduce_max_f32(const float* src, float* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_max_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void reduce_max_f16(const half* src, half* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_max_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void log_sum_exp_f32(const float* src, float* dst, unsigned int outer_size, unsigned int reduce_size) { log_sum_exp_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void log_sum_exp_f16(const half* src, half* dst, unsigned int outer_size, unsigned int reduce_size) { log_sum_exp_kernel(src, dst, outer_size, reduce_size); }

    template <typename T>
    __global__ void gather_kernel(const T* src, const index_t* indices, T* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) {
        unsigned int right = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row = blockIdx.y;
        if (right >= right_len || row >= left_len * dst_dim) return;
        unsigned int left = row / dst_dim;
        unsigned int out_index = row * right_len + right;
        long long idx = indices[out_index];
        dst[out_index] = src[(left * src_dim + (unsigned int)idx) * right_len + right];
    }

    template <typename T>
    __global__ void scatter_add_kernel(const T* src, const index_t* indices, T* dst, unsigned int left_len, unsigned int dst_dim, unsigned int index_len, unsigned int right_len) {
        unsigned int right = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row = blockIdx.y;
        if (right >= right_len || row >= left_len * index_len) return;
        unsigned int left = row / index_len;
        unsigned int index_pos = row % index_len;
        unsigned int src_index = row * right_len + right;
        long long idx = indices[src_index];
        unsigned int dst_index = (left * dst_dim + (unsigned int)idx) * right_len + right;
        atomic_add_t(dst + dst_index, src[src_index]);
    }

    template <typename T>
    __global__ void index_select_kernel(const T* src, const index_t* indices, T* dst, unsigned int left_len, unsigned int index_len, unsigned int src_dim, unsigned int right_len) {
        unsigned int right = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row = blockIdx.y;
        if (right >= right_len || row >= left_len * index_len) return;
        unsigned int left = row / index_len;
        unsigned int out_col = row % index_len;
        long long idx = indices[out_col];
        dst[row * right_len + right] = src[(left * src_dim + (unsigned int)idx) * right_len + right];
    }

    template <typename T>
    __global__ void index_add_kernel(const T* src, const index_t* indices, T* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) {
        unsigned int right = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row = blockIdx.y;
        if (right >= right_len || row >= left_len * src_dim) return;
        unsigned int left = row / src_dim;
        unsigned int src_col = row % src_dim;
        long long idx = indices[src_col];
        unsigned int dst_index = (left * dst_dim + (unsigned int)idx) * right_len + right;
        atomic_add_t(dst + dst_index, src[row * right_len + right]);
    }

    extern "C" __global__ void gather_f32(const float* src, const index_t* indices, float* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) { gather_kernel(src, indices, dst, left_len, src_dim, dst_dim, right_len); }
    extern "C" __global__ void gather_f16(const half* src, const index_t* indices, half* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) { gather_kernel(src, indices, dst, left_len, src_dim, dst_dim, right_len); }
    extern "C" __global__ void scatter_add_f32(const float* src, const index_t* indices, float* dst, unsigned int left_len, unsigned int dst_dim, unsigned int index_len, unsigned int right_len) { scatter_add_kernel(src, indices, dst, left_len, dst_dim, index_len, right_len); }
    extern "C" __global__ void scatter_add_f16(const half* src, const index_t* indices, half* dst, unsigned int left_len, unsigned int dst_dim, unsigned int index_len, unsigned int right_len) { scatter_add_kernel(src, indices, dst, left_len, dst_dim, index_len, right_len); }
    extern "C" __global__ void index_select_f32(const float* src, const index_t* indices, float* dst, unsigned int left_len, unsigned int index_len, unsigned int src_dim, unsigned int right_len) { index_select_kernel(src, indices, dst, left_len, index_len, src_dim, right_len); }
    extern "C" __global__ void index_select_f16(const half* src, const index_t* indices, half* dst, unsigned int left_len, unsigned int index_len, unsigned int src_dim, unsigned int right_len) { index_select_kernel(src, indices, dst, left_len, index_len, src_dim, right_len); }
    extern "C" __global__ void index_add_f32(const float* src, const index_t* indices, float* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) { index_add_kernel(src, indices, dst, left_len, src_dim, dst_dim, right_len); }
    extern "C" __global__ void index_add_f16(const half* src, const index_t* indices, half* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) { index_add_kernel(src, indices, dst, left_len, src_dim, dst_dim, right_len); }

    extern "C" __global__ void copy_compact_f32(const float* src, float* dst, StridedMeta meta) { copy_compact_kernel(src, dst, meta); }
    extern "C" __global__ void copy_compact_f16(const half* src, half* dst, StridedMeta meta) { copy_compact_kernel(src, dst, meta); }
    extern "C" __global__ void copy_compact_i64(const index_t* src, index_t* dst, StridedMeta meta) { copy_compact_kernel(src, dst, meta); }
    "#;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct StridedMeta {
        ndim: u32,
        offset: u32,
        size: u32,
        pad: u32,
        shape: [u32; MAX_DIMS],
        strides: [i32; MAX_DIMS],
    }

    unsafe impl DeviceRepr for StridedMeta {}

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ScalarMeta {
        scalar: f32,
    }

    unsafe impl DeviceRepr for ScalarMeta {}

    #[derive(Clone, Debug)]
    pub enum CudaInner {
        F16(CudaSlice<f16>),
        F32(CudaSlice<f32>),
        I64(CudaSlice<i64>),
    }

    #[derive(Clone, Debug)]
    pub struct CudaStorage {
        inner: CudaInner,
        runtime: Arc<CudaRuntime>,
    }

    #[derive(Debug)]
    struct CudaRuntime {
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        blas: Arc<CudaBlas>,
    }

    static RUNTIME: OnceLock<std::result::Result<Arc<CudaRuntime>, String>> = OnceLock::new();

    fn cuda_include_paths() -> Vec<String> {
        let env_roots = ["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR", "CUDNN_LIB"]
            .into_iter()
            .filter_map(|name| std::env::var(name).ok())
            .map(PathBuf::from);
        let standard_roots = [
            "/usr",
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
            "/opt/cuda/targets/x86_64-linux",
        ]
        .into_iter()
        .map(PathBuf::from);

        env_roots
            .chain(standard_roots)
            .flat_map(|root| {
                [
                    root.join("include"),
                    root.join("targets").join("x86_64-linux").join("include"),
                ]
            })
            .filter(|path| path.join("cuda.h").is_file())
            .map(|path| path.display().to_string())
            .collect()
    }

    fn runtime() -> Result<Arc<CudaRuntime>> {
        match RUNTIME
            .get_or_init(|| CudaRuntime::new().map(Arc::new).map_err(|err| err.to_string()))
        {
            Ok(runtime) => Ok(runtime.clone()),
            Err(err) => Err(Error::Cuda(err.clone())),
        }
    }

    impl CudaRuntime {
        fn new() -> Result<Self> {
            let context = CudaContext::new(0)
                .map_err(|err| Error::Cuda(format!("failed to init cuda context: {err}")))?;
            let stream = context.default_stream();
            let blas = CudaBlas::new(stream.clone())
                .map_err(|err| Error::Cuda(format!("failed to init cuBLAS: {err}")))?;
            let ptx = nvrtc::safe::compile_ptx_with_opts(
                KERNELS,
                nvrtc::CompileOptions {
                    use_fast_math: Some(true),
                    include_paths: cuda_include_paths(),
                    ..Default::default()
                },
            )
            .map_err(|err| Error::Cuda(format!("failed to compile cuda kernels: {err}")))?;
            let module = context
                .load_module(ptx)
                .map_err(|err| Error::Cuda(format!("failed to load cuda module: {err}")))?;
            Ok(Self { context, stream, module, blas: Arc::new(blas) })
        }

        fn load_function(&self, name: &str) -> Result<CudaFunction> {
            self.module
                .load_function(name)
                .map_err(|err| Error::Cuda(format!("failed to load kernel {name}: {err}")))
        }
    }

    fn maybe_profile_launch<T>(
        runtime: &CudaRuntime,
        launch: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !profiler::is_active() {
            return launch();
        }
        let Some(event_id) = profiler::current_scope_id() else {
            return launch();
        };

        let start = runtime
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|err| Error::Cuda(format!("failed to create cuda start event: {err}")))?;
        let end = runtime
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|err| Error::Cuda(format!("failed to create cuda end event: {err}")))?;

        start
            .record(&runtime.stream)
            .map_err(|err| Error::Cuda(format!("failed to record cuda start event: {err}")))?;
        let result = launch()?;
        end.record(&runtime.stream)
            .map_err(|err| Error::Cuda(format!("failed to record cuda end event: {err}")))?;
        let elapsed_ms = start
            .elapsed_ms(&end)
            .map_err(|err| Error::Cuda(format!("failed to read cuda event elapsed time: {err}")))?;
        profiler::record_device_time(event_id, (elapsed_ms * 1_000_000.0).round() as u64);
        Ok(result)
    }

    macro_rules! launch_1d {
        ($runtime:expr, $kernel:expr, $elements:expr, $($arg:expr),+ $(,)?) => {{
            let func = $runtime.load_function($kernel)?;
            let mut builder = $runtime.stream.launch_builder(&func);
            $(builder.arg($arg);)+
            maybe_profile_launch($runtime, || {
                unsafe { builder.launch(LaunchConfig::for_num_elems($elements as u32)) }
                    .map_err(|err| Error::Cuda(format!("kernel launch failed for {}: {err}", $kernel)))?;
                Ok(())
            })?;
        }};
    }

    macro_rules! launch_2d {
        ($runtime:expr, $kernel:expr, $width:expr, $height:expr, $($arg:expr),+ $(,)?) => {{
            let func = $runtime.load_function($kernel)?;
            let mut builder = $runtime.stream.launch_builder(&func);
            $(builder.arg($arg);)+
            let cfg = LaunchConfig {
                grid_dim: ($width.div_ceil(256) as u32, $height as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            maybe_profile_launch($runtime, || {
                unsafe { builder.launch(cfg) }
                    .map_err(|err| Error::Cuda(format!("kernel launch failed for {}: {err}", $kernel)))?;
                Ok(())
            })?;
        }};
    }

    macro_rules! launch_reduce {
        ($runtime:expr, $kernel:expr, $outer_size:expr, $($arg:expr),+ $(,)?) => {{
            let func = $runtime.load_function($kernel)?;
            let mut builder = $runtime.stream.launch_builder(&func);
            $(builder.arg($arg);)+
            let cfg = LaunchConfig {
                grid_dim: ($outer_size as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            maybe_profile_launch($runtime, || {
                unsafe { builder.launch(cfg) }
                    .map_err(|err| Error::Cuda(format!("kernel launch failed for {}: {err}", $kernel)))?;
                Ok(())
            })?;
        }};
    }

    fn strided_meta(layout: &Layout) -> StridedMeta {
        assert!(layout.ndim() <= MAX_DIMS, "cuda backend supports at most {MAX_DIMS} dims");
        let mut shape = [1u32; MAX_DIMS];
        let mut strides = [0i32; MAX_DIMS];
        for (dst, src) in shape.iter_mut().zip(layout.shape().iter()) {
            *dst = *src as u32;
        }
        for (dst, src) in strides.iter_mut().zip(layout.strides.0.iter()) {
            *dst = *src as i32;
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

    /// Allocates an uninitialised device buffer on the given stream.
    ///
    /// # Safety
    ///
    /// The caller must ensure every element is written before it is read.
    unsafe fn alloc_uninit<T: DeviceRepr>(
        runtime: &CudaRuntime,
        len: usize,
    ) -> Result<CudaSlice<T>> {
        // SAFETY: the caller is responsible for writing every element before reading.
        unsafe { runtime.stream.alloc::<T>(len) }
            .map_err(|err| Error::Cuda(format!("cuda alloc failed: {err}")))
    }

    impl CudaStorage {
        pub fn is_available() -> bool {
            runtime().is_ok()
        }

        /// Creates a storage with uninitialised device memory of `size` elements.
        ///
        /// # Safety
        ///
        /// The caller must write every element before reading.
        fn uninit(size: usize, dtype: DType) -> Result<Self> {
            let runtime = runtime()?;
            let inner = match dtype {
                DType::F16 => CudaInner::F16(unsafe { alloc_uninit::<f16>(&runtime, size) }?),
                DType::F32 => CudaInner::F32(unsafe { alloc_uninit::<f32>(&runtime, size) }?),
                DType::I64 => CudaInner::I64(unsafe { alloc_uninit::<i64>(&runtime, size) }?),
            };
            Ok(Self { inner, runtime })
        }

        pub fn zeros(size: usize, dtype: DType) -> Self {
            let runtime = runtime().expect("cuda backend unavailable");
            let inner = match dtype {
                DType::F16 => CudaInner::F16(
                    runtime.stream.alloc_zeros::<f16>(size).expect("cuda alloc failed"),
                ),
                DType::F32 => CudaInner::F32(
                    runtime.stream.alloc_zeros::<f32>(size).expect("cuda alloc failed"),
                ),
                DType::I64 => CudaInner::I64(
                    runtime.stream.alloc_zeros::<i64>(size).expect("cuda alloc failed"),
                ),
            };
            Self { inner, runtime }
        }

        pub fn ones(size: usize, dtype: DType) -> Self {
            match dtype {
                DType::F16 => {
                    Self::from_cpu_storage(CpuStorage::F16(vec![f16::from_f32(1.0); size]))
                }
                DType::F32 => Self::from_cpu_storage(CpuStorage::F32(vec![1.0; size])),
                DType::I64 => Self::from_cpu_storage(CpuStorage::I64(vec![1; size])),
            }
        }

        pub fn from_cpu_storage(inner: CpuStorage) -> Self {
            let runtime = runtime().expect("cuda backend unavailable");
            let inner = match inner {
                CpuStorage::F16(data) => {
                    CudaInner::F16(runtime.stream.clone_htod(&data).expect("cuda copy failed"))
                }
                CpuStorage::F32(data) => {
                    CudaInner::F32(runtime.stream.clone_htod(&data).expect("cuda copy failed"))
                }
                CpuStorage::I64(data) => {
                    CudaInner::I64(runtime.stream.clone_htod(&data).expect("cuda copy failed"))
                }
            };
            Self { inner, runtime }
        }

        pub fn cat(parts: &[(&CudaStorage, usize)]) -> Result<Self> {
            if parts.is_empty() {
                return Err(Error::LayoutMismatch("cat: empty parts".into()));
            }
            let dtype = parts[0].0.dtype();
            let total_len: usize = parts.iter().map(|(_, len)| *len).sum();
            // memcpy_dtod fills every element, so no zeroing needed.
            let mut out = Self::uninit(total_len, dtype)?;
            let mut offset = 0usize;
            match (&parts[0].0.inner, &mut out.inner) {
                (CudaInner::F16(_), CudaInner::F16(dst)) => {
                    for (part, len) in parts {
                        let CudaInner::F16(src) = &part.inner else {
                            return Err(Error::DTypeMismatch("cat: mixed dtypes".into()));
                        };
                        part.runtime
                            .stream
                            .memcpy_dtod(
                                &src.slice(..*len),
                                &mut dst.slice_mut(offset..offset + *len),
                            )
                            .map_err(|err| Error::Cuda(format!("cat memcpy failed: {err}")))?;
                        offset += *len;
                    }
                }
                (CudaInner::F32(_), CudaInner::F32(dst)) => {
                    for (part, len) in parts {
                        let CudaInner::F32(src) = &part.inner else {
                            return Err(Error::DTypeMismatch("cat: mixed dtypes".into()));
                        };
                        part.runtime
                            .stream
                            .memcpy_dtod(
                                &src.slice(..*len),
                                &mut dst.slice_mut(offset..offset + *len),
                            )
                            .map_err(|err| Error::Cuda(format!("cat memcpy failed: {err}")))?;
                        offset += *len;
                    }
                }
                (CudaInner::I64(_), CudaInner::I64(dst)) => {
                    for (part, len) in parts {
                        let CudaInner::I64(src) = &part.inner else {
                            return Err(Error::DTypeMismatch("cat: mixed dtypes".into()));
                        };
                        part.runtime
                            .stream
                            .memcpy_dtod(
                                &src.slice(..*len),
                                &mut dst.slice_mut(offset..offset + *len),
                            )
                            .map_err(|err| Error::Cuda(format!("cat memcpy failed: {err}")))?;
                        offset += *len;
                    }
                }
                _ => return Err(Error::DTypeMismatch("cat: mixed dtypes".into())),
            }
            Ok(out)
        }

        fn len(&self) -> usize {
            match &self.inner {
                CudaInner::F16(slice) => slice.len(),
                CudaInner::F32(slice) => slice.len(),
                CudaInner::I64(slice) => slice.len(),
            }
        }

        fn compact(&self, layout: &Layout) -> Result<Self> {
            if layout.is_compact() && layout.offset == 0 && layout.size() == self.len() {
                return Ok(self.clone());
            }
            // copy_compact writes every element, so no zeroing needed.
            let mut out = Self::uninit(layout.size(), self.dtype())?;
            self.copy_compact(layout, &mut out)?;
            Ok(out)
        }

        fn launch_unary_f16(&self, kernel: &str, src: &CudaSlice<f16>) -> Result<Self> {
            let out = unsafe { alloc_uninit::<f16>(&self.runtime, src.len()) }?;
            let len = src.len() as u32;
            launch_1d!(&self.runtime, kernel, src.len(), src, &out, &len);
            Ok(Self { inner: CudaInner::F16(out), runtime: self.runtime.clone() })
        }

        fn launch_unary_f32(&self, kernel: &str, src: &CudaSlice<f32>) -> Result<Self> {
            let out = unsafe { alloc_uninit::<f32>(&self.runtime, src.len()) }?;
            let len = src.len() as u32;
            launch_1d!(&self.runtime, kernel, src.len(), src, &out, &len);
            Ok(Self { inner: CudaInner::F32(out), runtime: self.runtime.clone() })
        }

        fn launch_scalar_f16(
            &self,
            kernel: &str,
            src: &CudaSlice<f16>,
            scalar: f32,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<f16>(&self.runtime, src.len()) }?;
            let len = src.len() as u32;
            let meta = ScalarMeta { scalar };
            launch_1d!(&self.runtime, kernel, src.len(), src, &out, &len, &meta);
            Ok(Self { inner: CudaInner::F16(out), runtime: self.runtime.clone() })
        }

        fn launch_scalar_f32(
            &self,
            kernel: &str,
            src: &CudaSlice<f32>,
            scalar: f32,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<f32>(&self.runtime, src.len()) }?;
            let len = src.len() as u32;
            let meta = ScalarMeta { scalar };
            launch_1d!(&self.runtime, kernel, src.len(), src, &out, &len, &meta);
            Ok(Self { inner: CudaInner::F32(out), runtime: self.runtime.clone() })
        }

        fn launch_binary_f16(
            &self,
            kernel: &str,
            lhs: &CudaSlice<f16>,
            rhs: &CudaSlice<f16>,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<f16>(&self.runtime, lhs.len()) }?;
            let len = lhs.len() as u32;
            launch_1d!(&self.runtime, kernel, lhs.len(), lhs, rhs, &out, &len);
            Ok(Self { inner: CudaInner::F16(out), runtime: self.runtime.clone() })
        }

        fn launch_binary_f32(
            &self,
            kernel: &str,
            lhs: &CudaSlice<f32>,
            rhs: &CudaSlice<f32>,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<f32>(&self.runtime, lhs.len()) }?;
            let len = lhs.len() as u32;
            launch_1d!(&self.runtime, kernel, lhs.len(), lhs, rhs, &out, &len);
            Ok(Self { inner: CudaInner::F32(out), runtime: self.runtime.clone() })
        }

        fn reduce_impl(&self, kernel: &str, outer_size: usize, reduce_size: usize) -> Result<Self> {
            match &self.inner {
                CudaInner::F16(src) => {
                    let out = unsafe { alloc_uninit::<f16>(&self.runtime, outer_size) }?;
                    let outer = outer_size as u32;
                    let reduce = reduce_size as u32;
                    launch_reduce!(&self.runtime, kernel, outer_size, src, &out, &outer, &reduce);
                    Ok(Self { inner: CudaInner::F16(out), runtime: self.runtime.clone() })
                }
                CudaInner::F32(src) => {
                    let out = unsafe { alloc_uninit::<f32>(&self.runtime, outer_size) }?;
                    let outer = outer_size as u32;
                    let reduce = reduce_size as u32;
                    launch_reduce!(&self.runtime, kernel, outer_size, src, &out, &outer, &reduce);
                    Ok(Self { inner: CudaInner::F32(out), runtime: self.runtime.clone() })
                }
                CudaInner::I64(_) => {
                    Err(Error::NotImplemented("cuda reductions for i64 are not implemented"))
                }
            }
        }
    }

    impl BackendStorage for CudaStorage {
        fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
            let compact = self.compact(l)?;
            match &compact.inner {
                CudaInner::F16(src) => compact.launch_scalar_f16("scalar_powf_f16", src, e as f32),
                CudaInner::F32(src) => compact.launch_scalar_f32("scalar_powf_f32", src, e as f32),
                CudaInner::I64(_) => {
                    Err(Error::NotImplemented("cuda scalar powf for i64 is not implemented"))
                }
            }
        }

        fn unary_op<O: UnaryOp>(&self, _op: O, l: &Layout) -> Result<Self> {
            let compact = self.compact(l)?;
            match (&compact.inner, O::KERNEL) {
                (CudaInner::F16(src), "Neg") => compact.launch_unary_f16("neg_f16", src),
                (CudaInner::F32(src), "Neg") => compact.launch_unary_f32("neg_f32", src),
                (CudaInner::F16(src), "exp") => compact.launch_unary_f16("exp_f16", src),
                (CudaInner::F32(src), "exp") => compact.launch_unary_f32("exp_f32", src),
                (CudaInner::F16(src), "log") => compact.launch_unary_f16("log_f16", src),
                (CudaInner::F32(src), "log") => compact.launch_unary_f32("log_f32", src),
                (CudaInner::F16(src), "sin") => compact.launch_unary_f16("sin_f16", src),
                (CudaInner::F32(src), "sin") => compact.launch_unary_f32("sin_f32", src),
                (CudaInner::F16(src), "cos") => compact.launch_unary_f16("cos_f16", src),
                (CudaInner::F32(src), "cos") => compact.launch_unary_f32("cos_f32", src),
                (CudaInner::F16(src), "tanh") => compact.launch_unary_f16("tanh_f16", src),
                (CudaInner::F32(src), "tanh") => compact.launch_unary_f32("tanh_f32", src),
                (CudaInner::F16(src), "relu") => compact.launch_unary_f16("relu_f16", src),
                (CudaInner::F32(src), "relu") => compact.launch_unary_f32("relu_f32", src),
                (CudaInner::F16(src), "scalar_add") => {
                    compact.launch_scalar_f16("scalar_add_f16", src, _op.f32(0.0))
                }
                (CudaInner::F32(src), "scalar_add") => {
                    compact.launch_scalar_f32("scalar_add_f32", src, _op.f32(0.0))
                }
                (CudaInner::F16(src), "scalar_mul") => {
                    compact.launch_scalar_f16("scalar_mul_f16", src, _op.f32(1.0))
                }
                (CudaInner::F32(src), "scalar_mul") => {
                    compact.launch_scalar_f32("scalar_mul_f32", src, _op.f32(1.0))
                }
                (CudaInner::F16(src), "scalar_div") => {
                    compact.launch_scalar_f16("scalar_div_f16", src, 1.0 / _op.f32(1.0))
                }
                (CudaInner::F32(src), "scalar_div") => {
                    compact.launch_scalar_f32("scalar_div_f32", src, 1.0 / _op.f32(1.0))
                }
                (CudaInner::F16(src), "relu_backward") => {
                    compact.launch_unary_f16("relu_backward_f16", src)
                }
                (CudaInner::F32(src), "relu_backward") => {
                    compact.launch_unary_f32("relu_backward_f32", src)
                }
                (CudaInner::I64(_), _) => {
                    Err(Error::NotImplemented("cuda unary ops for i64 are not implemented"))
                }
                _ => Err(Error::NotImplemented("cuda unary op is not implemented")),
            }
        }

        fn binary_op<O: BinaryOp>(
            &self,
            layout: &Layout,
            other: &Self,
            layout_other: &Layout,
        ) -> Result<Self> {
            let lhs = self.compact(layout)?;
            let rhs = other.compact(layout_other)?;
            match (&lhs.inner, &rhs.inner, O::KERNEL) {
                (CudaInner::F16(a), CudaInner::F16(b), "add") => {
                    lhs.launch_binary_f16("add_f16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "add") => {
                    lhs.launch_binary_f32("add_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "sub") => {
                    lhs.launch_binary_f16("sub_f16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "sub") => {
                    lhs.launch_binary_f32("sub_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "mul") => {
                    lhs.launch_binary_f16("mul_f16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "mul") => {
                    lhs.launch_binary_f32("mul_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "div") => {
                    lhs.launch_binary_f16("div_f16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "div") => {
                    lhs.launch_binary_f32("div_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "powf") => {
                    lhs.launch_binary_f16("powf_f16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "powf") => {
                    lhs.launch_binary_f32("powf_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "eq") => {
                    lhs.launch_binary_f16("eq_f16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "eq") => {
                    lhs.launch_binary_f32("eq_f32", a, b)
                }
                _ => Err(Error::NotImplemented("cuda binary op is not implemented for this dtype")),
            }
        }

        fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
            let compact = self.compact(layout)?;
            let reduced = match O::KERNEL {
                "reduce_sum" => compact.reduce_impl(
                    match compact.dtype() {
                        DType::F16 => "reduce_sum_f16",
                        DType::F32 => "reduce_sum_f32",
                        DType::I64 => {
                            return Err(Error::NotImplemented(
                                "cuda reduce_sum for i64 is not implemented",
                            ));
                        }
                    },
                    dst.len(),
                    compact.len() / dst.len(),
                )?,
                "reduce_max" => compact.reduce_impl(
                    match compact.dtype() {
                        DType::F16 => "reduce_max_f16",
                        DType::F32 => "reduce_max_f32",
                        DType::I64 => {
                            return Err(Error::NotImplemented(
                                "cuda reduce_max for i64 is not implemented",
                            ));
                        }
                    },
                    dst.len(),
                    compact.len() / dst.len(),
                )?,
                _ => return Err(Error::NotImplemented("cuda reduction is not implemented")),
            };
            *dst = reduced;
            Ok(())
        }

        fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
            use cublas_sys::{
                cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t,
            };

            let lhs = self.compact(layout)?;
            let rhs = other.compact(layout_other)?;
            let ndim = layout.ndim();
            let m = layout.shape()[ndim - 2];
            let k = layout.shape()[ndim - 1];
            let n = layout_other.shape()[ndim - 1];
            let batch = if ndim > 2 { layout.shape().iter().take(ndim - 2).product() } else { 1 };

            match (&lhs.inner, &rhs.inner) {
                (CudaInner::F32(a), CudaInner::F32(b)) => {
                    // cuBLAS with beta=0 overwrites every output element; no zeroing needed.
                    let mut out = unsafe { alloc_uninit::<f32>(&lhs.runtime, batch * m * n) }?;
                    let alpha = 1.0f32;
                    let beta = 0.0f32;
                    let alpha_ptr = &alpha as *const f32 as *const _;
                    let beta_ptr = &beta as *const f32 as *const _;
                    let stream = out.stream().clone();
                    let b_view = b.slice(..);
                    let a_view = a.slice(..);
                    let (b_ptr, gb) = b_view.device_ptr(&stream);
                    let (a_ptr, ga) = a_view.device_ptr(&stream);
                    let (c_ptr, gc) = out.device_ptr_mut(&stream);
                    maybe_profile_launch(&lhs.runtime, || {
                        unsafe {
                            cublas_result::gemm_strided_batched_ex(
                                *lhs.runtime.blas.handle(),
                                cublasOperation_t::CUBLAS_OP_N,
                                cublasOperation_t::CUBLAS_OP_N,
                                n as i32,
                                m as i32,
                                k as i32,
                                alpha_ptr,
                                b_ptr as *const _,
                                cudaDataType_t::CUDA_R_32F,
                                n as i32,
                                (k * n) as i64,
                                a_ptr as *const _,
                                cudaDataType_t::CUDA_R_32F,
                                k as i32,
                                (m * k) as i64,
                                beta_ptr,
                                c_ptr as *mut _,
                                cudaDataType_t::CUDA_R_32F,
                                n as i32,
                                (m * n) as i64,
                                batch as i32,
                                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                            )
                        }
                        .map_err(|err| Error::Cuda(format!("cuBLAS matmul failed: {err}")))?;
                        Ok(())
                    })?;
                    drop(gc);
                    drop(ga);
                    drop(gb);
                    Ok(Self { inner: CudaInner::F32(out), runtime: lhs.runtime.clone() })
                }
                (CudaInner::F16(a), CudaInner::F16(b)) => {
                    let mut out = unsafe { alloc_uninit::<f16>(&lhs.runtime, batch * m * n) }?;
                    let alpha = f16::from_f32(1.0);
                    let beta = f16::from_f32(0.0);
                    let alpha_f32 = 1.0f32;
                    let beta_f32 = 0.0f32;
                    let alpha_ptr = &alpha_f32 as *const f32 as *const _;
                    let beta_ptr = &beta_f32 as *const f32 as *const _;
                    let _alpha_half = alpha;
                    let _beta_half = beta;
                    let stream = out.stream().clone();
                    let b_view = b.slice(..);
                    let a_view = a.slice(..);
                    let (b_ptr, gb) = b_view.device_ptr(&stream);
                    let (a_ptr, ga) = a_view.device_ptr(&stream);
                    let (c_ptr, gc) = out.device_ptr_mut(&stream);
                    maybe_profile_launch(&lhs.runtime, || {
                        unsafe {
                            cublas_result::gemm_strided_batched_ex(
                                *lhs.runtime.blas.handle(),
                                cublasOperation_t::CUBLAS_OP_N,
                                cublasOperation_t::CUBLAS_OP_N,
                                n as i32,
                                m as i32,
                                k as i32,
                                alpha_ptr,
                                b_ptr as *const _,
                                cudaDataType_t::CUDA_R_16F,
                                n as i32,
                                (k * n) as i64,
                                a_ptr as *const _,
                                cudaDataType_t::CUDA_R_16F,
                                k as i32,
                                (m * k) as i64,
                                beta_ptr,
                                c_ptr as *mut _,
                                cudaDataType_t::CUDA_R_16F,
                                n as i32,
                                (m * n) as i64,
                                batch as i32,
                                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                            )
                        }
                        .map_err(|err| Error::Cuda(format!("cuBLAS matmul failed: {err}")))?;
                        Ok(())
                    })?;
                    drop(gc);
                    drop(ga);
                    drop(gb);
                    Ok(Self { inner: CudaInner::F16(out), runtime: lhs.runtime.clone() })
                }
                _ => Err(Error::DTypeMismatch("matmul dtype mismatch".into())),
            }
        }

        fn gather(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
        ) -> Result<Self> {
            let src = self.compact(layout)?;
            let indices = indices.compact(indices_layout)?;
            let left_len: usize = layout.shape().iter().take(dim).product();
            let src_dim = layout.shape()[dim];
            let dst_dim = indices_layout.shape()[dim];
            let right_len: usize = layout.shape().iter().skip(dim + 1).product();
            match (&src.inner, &indices.inner) {
                (CudaInner::F16(src), CudaInner::I64(indices)) => {
                    let out = unsafe { alloc_uninit::<f16>(&self.runtime, indices_layout.size()) }?;
                    let left = left_len as u32;
                    let src_dim_u32 = src_dim as u32;
                    let dst_dim_u32 = dst_dim as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "gather_f16",
                        right_len,
                        left_len * dst_dim,
                        src,
                        indices,
                        &out,
                        &left,
                        &src_dim_u32,
                        &dst_dim_u32,
                        &right
                    );
                    Ok(Self { inner: CudaInner::F16(out), runtime: self.runtime.clone() })
                }
                (CudaInner::F32(src), CudaInner::I64(indices)) => {
                    let out = unsafe { alloc_uninit::<f32>(&self.runtime, indices_layout.size()) }?;
                    let left = left_len as u32;
                    let src_dim_u32 = src_dim as u32;
                    let dst_dim_u32 = dst_dim as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "gather_f32",
                        right_len,
                        left_len * dst_dim,
                        src,
                        indices,
                        &out,
                        &left,
                        &src_dim_u32,
                        &dst_dim_u32,
                        &right
                    );
                    Ok(Self { inner: CudaInner::F32(out), runtime: self.runtime.clone() })
                }
                _ => Err(Error::DTypeMismatch(
                    "gather requires floating source and i64 indices".into(),
                )),
            }
        }

        fn scatter_add(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
            dst_shape: &[usize],
        ) -> Result<Self> {
            let src = self.compact(layout)?;
            let indices = indices.compact(indices_layout)?;
            let left_len: usize = dst_shape[..dim].iter().product();
            let dst_dim = dst_shape[dim];
            let index_len = indices_layout.shape()[dim];
            let right_len: usize = dst_shape[dim + 1..].iter().product();
            match (&src.inner, &indices.inner) {
                (CudaInner::F16(src), CudaInner::I64(indices)) => {
                    let out = src
                        .stream()
                        .alloc_zeros::<f16>(dst_shape.iter().product())
                        .map_err(|err| Error::Cuda(format!("cuda alloc failed: {err}")))?;
                    let left = left_len as u32;
                    let dst_dim_u32 = dst_dim as u32;
                    let index_len_u32 = index_len as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "scatter_add_f16",
                        right_len,
                        left_len * index_len,
                        src,
                        indices,
                        &out,
                        &left,
                        &dst_dim_u32,
                        &index_len_u32,
                        &right
                    );
                    Ok(Self { inner: CudaInner::F16(out), runtime: self.runtime.clone() })
                }
                (CudaInner::F32(src), CudaInner::I64(indices)) => {
                    let out = src
                        .stream()
                        .alloc_zeros::<f32>(dst_shape.iter().product())
                        .map_err(|err| Error::Cuda(format!("cuda alloc failed: {err}")))?;
                    let left = left_len as u32;
                    let dst_dim_u32 = dst_dim as u32;
                    let index_len_u32 = index_len as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "scatter_add_f32",
                        right_len,
                        left_len * index_len,
                        src,
                        indices,
                        &out,
                        &left,
                        &dst_dim_u32,
                        &index_len_u32,
                        &right
                    );
                    Ok(Self { inner: CudaInner::F32(out), runtime: self.runtime.clone() })
                }
                _ => Err(Error::DTypeMismatch(
                    "scatter_add requires floating source and i64 indices".into(),
                )),
            }
        }

        fn index_select(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
        ) -> Result<Self> {
            let src = self.compact(layout)?;
            let indices = indices.compact(indices_layout)?;
            let left_len: usize = layout.shape().iter().take(dim).product();
            let index_len = indices_layout.shape()[0];
            let src_dim = layout.shape()[dim];
            let right_len: usize = layout.shape().iter().skip(dim + 1).product();
            match (&src.inner, &indices.inner) {
                (CudaInner::F16(src), CudaInner::I64(indices)) => {
                    let out = unsafe { alloc_uninit::<f16>(&self.runtime, left_len * index_len * right_len) }?;
                    let left = left_len as u32;
                    let index_len_u32 = index_len as u32;
                    let src_dim_u32 = src_dim as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "index_select_f16",
                        right_len,
                        left_len * index_len,
                        src,
                        indices,
                        &out,
                        &left,
                        &index_len_u32,
                        &src_dim_u32,
                        &right
                    );
                    Ok(Self { inner: CudaInner::F16(out), runtime: self.runtime.clone() })
                }
                (CudaInner::F32(src), CudaInner::I64(indices)) => {
                    let out = unsafe { alloc_uninit::<f32>(&self.runtime, left_len * index_len * right_len) }?;
                    let left = left_len as u32;
                    let index_len_u32 = index_len as u32;
                    let src_dim_u32 = src_dim as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "index_select_f32",
                        right_len,
                        left_len * index_len,
                        src,
                        indices,
                        &out,
                        &left,
                        &index_len_u32,
                        &src_dim_u32,
                        &right
                    );
                    Ok(Self { inner: CudaInner::F32(out), runtime: self.runtime.clone() })
                }
                _ => Err(Error::DTypeMismatch(
                    "index_select requires floating source and i64 indices".into(),
                )),
            }
        }

        fn index_add(
            &self,
            layout: &Layout,
            dim: usize,
            indices: &Self,
            indices_layout: &Layout,
            dst_shape: &[usize],
        ) -> Result<Self> {
            let src = self.compact(layout)?;
            let indices = indices.compact(indices_layout)?;
            let left_len: usize = dst_shape[..dim].iter().product();
            let src_dim = layout.shape()[dim];
            let dst_dim = dst_shape[dim];
            let right_len: usize = dst_shape[dim + 1..].iter().product();
            match (&src.inner, &indices.inner) {
                (CudaInner::F16(src), CudaInner::I64(indices)) => {
                    let out = src
                        .stream()
                        .alloc_zeros::<f16>(dst_shape.iter().product())
                        .map_err(|err| Error::Cuda(format!("cuda alloc failed: {err}")))?;
                    let left = left_len as u32;
                    let src_dim_u32 = src_dim as u32;
                    let dst_dim_u32 = dst_dim as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "index_add_f16",
                        right_len,
                        left_len * src_dim,
                        src,
                        indices,
                        &out,
                        &left,
                        &src_dim_u32,
                        &dst_dim_u32,
                        &right
                    );
                    Ok(Self { inner: CudaInner::F16(out), runtime: self.runtime.clone() })
                }
                (CudaInner::F32(src), CudaInner::I64(indices)) => {
                    let out = src
                        .stream()
                        .alloc_zeros::<f32>(dst_shape.iter().product())
                        .map_err(|err| Error::Cuda(format!("cuda alloc failed: {err}")))?;
                    let left = left_len as u32;
                    let src_dim_u32 = src_dim as u32;
                    let dst_dim_u32 = dst_dim as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "index_add_f32",
                        right_len,
                        left_len * src_dim,
                        src,
                        indices,
                        &out,
                        &left,
                        &src_dim_u32,
                        &dst_dim_u32,
                        &right
                    );
                    Ok(Self { inner: CudaInner::F32(out), runtime: self.runtime.clone() })
                }
                _ => Err(Error::DTypeMismatch(
                    "index_add requires floating source and i64 indices".into(),
                )),
            }
        }

        fn log_sum_exp(
            &self,
            layout: &Layout,
            outer_size: usize,
            reduce_size: usize,
        ) -> Result<Self> {
            self.compact(layout)?.reduce_impl(
                match self.dtype() {
                    DType::F16 => "log_sum_exp_f16",
                    DType::F32 => "log_sum_exp_f32",
                    DType::I64 => {
                        return Err(Error::NotImplemented(
                            "cuda log_sum_exp for i64 is not implemented",
                        ));
                    }
                },
                outer_size,
                reduce_size,
            )
        }

        fn dtype(&self) -> DType {
            match &self.inner {
                CudaInner::F16(_) => DType::F16,
                CudaInner::F32(_) => DType::F32,
                CudaInner::I64(_) => DType::I64,
            }
        }

        fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
            let layout = layout.borrow();
            let compact = self.compact(layout).expect("cuda compact failed");
            match &compact.inner {
                CudaInner::F16(slice) => D::to_vec(&CpuStorage::F16(
                    compact.runtime.stream.clone_dtoh(slice).expect("cuda dtoh failed"),
                )),
                CudaInner::F32(slice) => D::to_vec(&CpuStorage::F32(
                    compact.runtime.stream.clone_dtoh(slice).expect("cuda dtoh failed"),
                )),
                CudaInner::I64(slice) => D::to_vec(&CpuStorage::I64(
                    compact.runtime.stream.clone_dtoh(slice).expect("cuda dtoh failed"),
                )),
            }
        }

        fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()> {
            let meta = strided_meta(src_layout);
            match (&self.inner, &mut dst.inner) {
                (CudaInner::F16(src), CudaInner::F16(dst)) => {
                    launch_1d!(
                        &self.runtime,
                        "copy_compact_f16",
                        src_layout.size(),
                        src,
                        dst,
                        &meta
                    );
                    Ok(())
                }
                (CudaInner::F32(src), CudaInner::F32(dst)) => {
                    launch_1d!(
                        &self.runtime,
                        "copy_compact_f32",
                        src_layout.size(),
                        src,
                        dst,
                        &meta
                    );
                    Ok(())
                }
                (CudaInner::I64(src), CudaInner::I64(dst)) => {
                    launch_1d!(
                        &self.runtime,
                        "copy_compact_i64",
                        src_layout.size(),
                        src,
                        dst,
                        &meta
                    );
                    Ok(())
                }
                _ => Err(Error::DTypeMismatch("copy_compact: dtype mismatch".into())),
            }
        }
    }

    pub fn synchronize() {
        if let Ok(runtime) = runtime() {
            let _ = runtime.context.synchronize();
        }
    }

    pub fn availability() -> Result<()> {
        runtime().map(|_| ())
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
mod imp {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct CudaStorage;

    impl CudaStorage {
        pub fn is_available() -> bool {
            false
        }

        pub fn zeros(_size: usize, _dtype: DType) -> Self {
            panic!("CUDA backend is only available on Linux with the `cuda` feature enabled")
        }

        pub fn ones(_size: usize, _dtype: DType) -> Self {
            panic!("CUDA backend is only available on Linux with the `cuda` feature enabled")
        }

        pub fn from_cpu_storage(_inner: CpuStorage) -> Self {
            panic!("CUDA backend is only available on Linux with the `cuda` feature enabled")
        }

        pub fn cat(_parts: &[(&CudaStorage, usize)]) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
    }

    impl BackendStorage for CudaStorage {
        fn ewise_powf(&self, _: f64, _: &Layout) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn unary_op<O: UnaryOp>(&self, _: O, _: &Layout) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn binary_op<O: BinaryOp>(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn reduce<O: ReduceOp>(&self, _: &Layout, _: &mut Self) -> Result<()> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn matmul(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn gather(&self, _: &Layout, _: usize, _: &Self, _: &Layout) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn scatter_add(
            &self,
            _: &Layout,
            _: usize,
            _: &Self,
            _: &Layout,
            _: &[usize],
        ) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn index_select(&self, _: &Layout, _: usize, _: &Self, _: &Layout) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn index_add(
            &self,
            _: &Layout,
            _: usize,
            _: &Self,
            _: &Layout,
            _: &[usize],
        ) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn log_sum_exp(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn dtype(&self) -> DType {
            panic!("cuda backend is unavailable")
        }
        fn to_vec<D: WithDType>(&self, _: impl Borrow<Layout>) -> Vec<D> {
            panic!("cuda backend is unavailable")
        }
        fn copy_compact(&self, _: &Layout, _: &mut Self) -> Result<()> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
    }

    pub fn synchronize() {}

    pub fn availability() -> Result<()> {
        Err(Error::NotImplemented(
            "cuda backend is only available on Linux with the `cuda` feature enabled",
        ))
    }
}

pub use imp::*;
