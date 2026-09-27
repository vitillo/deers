//! CUDA tensor storage with JIT-compiled PTX kernels and cuBLAS matmul.

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

    use half::{bf16, f16};

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
    #include <cuda_bf16.h>
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
    template <>
    __device__ __forceinline__ __nv_bfloat16 zero_value<__nv_bfloat16>() { return __float2bfloat16(0.0f); }

    template <typename T>
    __device__ __forceinline__ T one_value();
    template <>
    __device__ __forceinline__ float one_value<float>() { return 1.0f; }
    template <>
    __device__ __forceinline__ half one_value<half>() { return __float2half(1.0f); }
    template <>
    __device__ __forceinline__ __nv_bfloat16 one_value<__nv_bfloat16>() { return __float2bfloat16(1.0f); }

    template <typename T>
    __device__ __forceinline__ float to_float(T v);
    template <>
    __device__ __forceinline__ float to_float<float>(float v) { return v; }
    template <>
    __device__ __forceinline__ float to_float<half>(half v) { return __half2float(v); }
    template <>
    __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

    template <typename T>
    __device__ __forceinline__ T from_float(float v);
    template <>
    __device__ __forceinline__ float from_float<float>(float v) { return v; }
    template <>
    __device__ __forceinline__ half from_float<half>(float v) { return __float2half(v); }
    template <>
    __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

    template <typename T>
    __device__ __forceinline__ void atomic_add_t(T* dst, T v);
    template <>
    __device__ __forceinline__ void atomic_add_t<float>(float* dst, float v) { atomicAdd(dst, v); }
    template <>
    __device__ __forceinline__ void atomic_add_t<half>(half* dst, half v) { atomicAdd(dst, v); }
    template <>
    __device__ __forceinline__ void atomic_add_t<__nv_bfloat16>(__nv_bfloat16* dst, __nv_bfloat16 v) { atomicAdd(dst, v); }

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

    #define DEFINE_UNARY_BF16(name, expr) \
    extern "C" __global__ void name##_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int size) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { float x = __bfloat162float(src[idx]); dst[idx] = __float2bfloat16((expr)); } \
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
    DEFINE_UNARY_BF16(neg, -x)
    DEFINE_UNARY_BF16(exp, expf(x))
    DEFINE_UNARY_BF16(log, logf(x))
    DEFINE_UNARY_BF16(sin, sinf(x))
    DEFINE_UNARY_BF16(cos, cosf(x))
    DEFINE_UNARY_BF16(tanh, tanhf(x))
    DEFINE_UNARY_BF16(relu, fmaxf(x, 0.0f))
    DEFINE_UNARY_BF16(relu_backward, x > 0.0f ? 1.0f : 0.0f)

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

    #define DEFINE_SCALAR_BF16(name, expr) \
    extern "C" __global__ void name##_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int size, ScalarMeta meta) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { float x = __bfloat162float(src[idx]); float s = meta.scalar; dst[idx] = __float2bfloat16((expr)); } \
    }

    DEFINE_SCALAR_F32(scalar_add, x + s)
    DEFINE_SCALAR_F16(scalar_add, x + s)
    DEFINE_SCALAR_F32(scalar_mul, x * s)
    DEFINE_SCALAR_F16(scalar_mul, x * s)
    DEFINE_SCALAR_F32(scalar_div, x / s)
    DEFINE_SCALAR_F16(scalar_div, x / s)
    DEFINE_SCALAR_F32(scalar_powf, powf(x, s))
    DEFINE_SCALAR_F16(scalar_powf, powf(x, s))
    DEFINE_SCALAR_BF16(scalar_add, x + s)
    DEFINE_SCALAR_BF16(scalar_mul, x * s)
    DEFINE_SCALAR_BF16(scalar_div, x / s)
    DEFINE_SCALAR_BF16(scalar_powf, powf(x, s))

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

    #define DEFINE_BINARY_BF16(name, expr) \
    extern "C" __global__ void name##_bf16(const __nv_bfloat16* lhs, const __nv_bfloat16* rhs, __nv_bfloat16* dst, unsigned int size) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { float x = __bfloat162float(lhs[idx]); float y = __bfloat162float(rhs[idx]); dst[idx] = __float2bfloat16((expr)); } \
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
    DEFINE_BINARY_BF16(add, x + y)
    DEFINE_BINARY_BF16(sub, x - y)
    DEFINE_BINARY_BF16(mul, x * y)
    DEFINE_BINARY_BF16(div, x / y)
    DEFINE_BINARY_BF16(powf, powf(x, y))
    DEFINE_BINARY_BF16(eq, x == y ? 1.0f : 0.0f)

    // Dtype casts: one native kernel per (source, target) pair. Narrowing rounds
    // once to nearest-even through the CUDA half/bfloat16 intrinsics; widening is
    // exact; float-to-integer casts truncate toward zero with saturating bounds so
    // out-of-range and NaN inputs match the host `as` semantics instead of
    // trapping on undefined device conversions.
    __device__ __forceinline__ long long f32_to_ll_sat(float x) {
        if (isnan(x)) return 0;
        if (x >= 9223372036854775808.0f) return 9223372036854775807LL;
        if (x <= -9223372036854775808.0f) return -9223372036854775807LL - 1;
        return (long long)x;
    }

    #define DEFINE_CAST(name, src_t, dst_t, expr) \
    extern "C" __global__ void name(const src_t* src, dst_t* dst, unsigned int size) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { auto x = src[idx]; dst[idx] = (expr); } \
    }

    DEFINE_CAST(cast_f16_f32, half, float, __half2float(x))
    DEFINE_CAST(cast_f16_bf16, half, __nv_bfloat16, __float2bfloat16(__half2float(x)))
    DEFINE_CAST(cast_f16_i64, half, long long, f32_to_ll_sat(__half2float(x)))
    DEFINE_CAST(cast_bf16_f16, __nv_bfloat16, half, __float2half(__bfloat162float(x)))
    DEFINE_CAST(cast_bf16_f32, __nv_bfloat16, float, __bfloat162float(x))
    DEFINE_CAST(cast_bf16_i64, __nv_bfloat16, long long, f32_to_ll_sat(__bfloat162float(x)))
    DEFINE_CAST(cast_f32_f16, float, half, __float2half(x))
    DEFINE_CAST(cast_f32_bf16, float, __nv_bfloat16, __float2bfloat16(x))
    DEFINE_CAST(cast_f32_i64, float, long long, f32_to_ll_sat(x))
    DEFINE_CAST(cast_i64_f32, long long, float, (float)x)
    DEFINE_CAST(cast_i64_f16, long long, half, __float2half((float)x))
    DEFINE_CAST(cast_i64_bf16, long long, __nv_bfloat16, __float2bfloat16((float)x))

    #define DEFINE_CMP_SCALAR_F32(name, op) \
    extern "C" __global__ void name(const float* src, float* dst, unsigned int size, ScalarMeta meta) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { dst[idx] = (src[idx] op meta.scalar ? 1.0f : 0.0f); } \
    }

    #define DEFINE_CMP_SCALAR_F16(name, op) \
    extern "C" __global__ void name(const half* src, half* dst, unsigned int size, ScalarMeta meta) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { dst[idx] = (__half2float(src[idx]) op __half2float(__float2half(meta.scalar)) ? __float2half(1.0f) : __float2half(0.0f)); } \
    }

    #define DEFINE_CMP_SCALAR_BF16(name, op) \
    extern "C" __global__ void name(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int size, ScalarMeta meta) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { dst[idx] = (__bfloat162float(src[idx]) op __bfloat162float(__float2bfloat16(meta.scalar)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)); } \
    }

    DEFINE_CMP_SCALAR_F32(ne_scalar_f32, !=)
    DEFINE_CMP_SCALAR_F16(ne_scalar_f16, !=)
    DEFINE_CMP_SCALAR_BF16(ne_scalar_bf16, !=)

    typedef struct {
        long long scalar;
    } ScalarMetaI64;

    #define DEFINE_CMP_SCALAR_I64(name, op) \
    extern "C" __global__ void name(const long long* src, long long* dst, unsigned int size, ScalarMetaI64 meta) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { dst[idx] = (src[idx] op meta.scalar ? 1LL : 0LL); } \
    }

    DEFINE_CMP_SCALAR_I64(ne_scalar_i64, !=)

    #define DEFINE_WHERE(name, T, is_nonzero) \
    extern "C" __global__ void name(const T* cond, const T* on_true, const T* on_false, T* dst, unsigned int size) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < size) { dst[idx] = ((is_nonzero) ? on_true[idx] : on_false[idx]); } \
    }

    DEFINE_WHERE(where_f32, float, cond[idx] != 0.0f)
    DEFINE_WHERE(where_f16, half, __half2float(cond[idx]) != 0.0f)
    DEFINE_WHERE(where_bf16, __nv_bfloat16, __bfloat162float(cond[idx]) != 0.0f)
    DEFINE_WHERE(where_i64, long long, cond[idx] != 0LL)

    // Warp-shuffle sum of 32 floats within a single warp — no __syncthreads needed.
    __device__ __forceinline__ float warp_reduce_sum(float v) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, mask);
        return v;
    }

    // Warp-shuffle max of 32 floats within a single warp — no __syncthreads needed.
    __device__ __forceinline__ float warp_reduce_max(float v) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
        return v;
    }

    // Each block handles one output row. Shared memory tree reduction drives the
    // count down to 32, then the final warp finishes with shuffle (faster, avoids
    // __syncthreads for those last 5 steps).
    template <typename T>
    __global__ void reduce_sum_kernel(const T* src, T* dst, unsigned int outer_size, unsigned int reduce_size) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        __shared__ float smem[REDUCE_THREADS];
        float acc = 0.0f;
        for (unsigned int col = threadIdx.x; col < reduce_size; col += blockDim.x)
            acc += to_float(src[row * reduce_size + col]);
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x < 32) {
            float val = warp_reduce_sum(smem[threadIdx.x]);
            if (threadIdx.x == 0) dst[row] = from_float<T>(val);
        }
    }

    template <typename T>
    __global__ void reduce_max_kernel(const T* src, T* dst, unsigned int outer_size, unsigned int reduce_size) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        __shared__ float smem[REDUCE_THREADS];
        float acc = -1.0f / 0.0f;
        for (unsigned int col = threadIdx.x; col < reduce_size; col += blockDim.x)
            acc = fmaxf(acc, to_float(src[row * reduce_size + col]));
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x < 32) {
            float val = warp_reduce_max(smem[threadIdx.x]);
            if (threadIdx.x == 0) dst[row] = from_float<T>(val);
        }
    }

    template <typename T>
    __global__ void log_sum_exp_kernel(const T* src, T* dst, unsigned int outer_size, unsigned int reduce_size) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        __shared__ float smem[REDUCE_THREADS];

        // Pass 1: find row max.
        float row_max = -1.0f / 0.0f;
        for (unsigned int col = threadIdx.x; col < reduce_size; col += blockDim.x)
            row_max = fmaxf(row_max, to_float(src[row * reduce_size + col]));
        smem[threadIdx.x] = row_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x < 32) smem[threadIdx.x] = warp_reduce_max(smem[threadIdx.x]);
        __syncthreads();
        row_max = smem[0];

        // Pass 2: sum exp(x - max).
        float acc = 0.0f;
        for (unsigned int col = threadIdx.x; col < reduce_size; col += blockDim.x)
            acc += expf(to_float(src[row * reduce_size + col]) - row_max);
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x < 32) {
            float val = warp_reduce_sum(smem[threadIdx.x]);
            if (threadIdx.x == 0) dst[row] = from_float<T>(logf(val) + row_max);
        }
    }

    extern "C" __global__ void reduce_sum_f32(const float* src, float* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_sum_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void reduce_sum_f16(const half* src, half* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_sum_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void reduce_sum_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_sum_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void reduce_max_f32(const float* src, float* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_max_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void reduce_max_f16(const half* src, half* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_max_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void reduce_max_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int outer_size, unsigned int reduce_size) { reduce_max_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void log_sum_exp_f32(const float* src, float* dst, unsigned int outer_size, unsigned int reduce_size) { log_sum_exp_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void log_sum_exp_f16(const half* src, half* dst, unsigned int outer_size, unsigned int reduce_size) { log_sum_exp_kernel(src, dst, outer_size, reduce_size); }
    extern "C" __global__ void log_sum_exp_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int outer_size, unsigned int reduce_size) { log_sum_exp_kernel(src, dst, outer_size, reduce_size); }

    // Fused log-softmax forward: computes x[i] - log(sum_j exp(x[j] - max)) - max for each row.
    // Avoids materialising the broadcast LSE tensor and the separate subtraction kernel.
    template <typename T>
    __global__ void log_softmax_fwd_kernel(const T* src, T* dst, unsigned int outer_size, unsigned int inner_size) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        __shared__ float smem[REDUCE_THREADS];

        // Pass 1: row max.
        float row_max = -1.0f / 0.0f;
        for (unsigned int col = threadIdx.x; col < inner_size; col += blockDim.x)
            row_max = fmaxf(row_max, to_float(src[row * inner_size + col]));
        smem[threadIdx.x] = row_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x < 32) smem[threadIdx.x] = warp_reduce_max(smem[threadIdx.x]);
        __syncthreads();
        row_max = smem[0];

        // Pass 2: sum(exp(x - max)).
        float acc = 0.0f;
        for (unsigned int col = threadIdx.x; col < inner_size; col += blockDim.x)
            acc += expf(to_float(src[row * inner_size + col]) - row_max);
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x < 32) smem[threadIdx.x] = warp_reduce_sum(smem[threadIdx.x]);
        __syncthreads();
        float lse = logf(smem[0]) + row_max;

        // Pass 3: write x[i] - lse.
        for (unsigned int col = threadIdx.x; col < inner_size; col += blockDim.x)
            dst[row * inner_size + col] = from_float<T>(to_float(src[row * inner_size + col]) - lse);
    }

    // Fused log-softmax backward: grad_input[i] = grad[i] - exp(lsm_out[i]) * sum_j grad[j].
    // exp(lsm_out[i]) = softmax(x)[i], so this is the standard log-softmax gradient.
    template <typename T>
    __global__ void log_softmax_bwd_kernel(const T* grad, const T* lsm_out, T* grad_input, unsigned int outer_size, unsigned int inner_size) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        __shared__ float smem[REDUCE_THREADS];

        // Pass 1: sum(grad) per row.
        float sum_grad = 0.0f;
        for (unsigned int col = threadIdx.x; col < inner_size; col += blockDim.x)
            sum_grad += to_float(grad[row * inner_size + col]);
        smem[threadIdx.x] = sum_grad;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x < 32) smem[threadIdx.x] = warp_reduce_sum(smem[threadIdx.x]);
        __syncthreads();
        sum_grad = smem[0];

        // Pass 2: grad_input[i] = grad[i] - softmax(x)[i] * sum_grad.
        for (unsigned int col = threadIdx.x; col < inner_size; col += blockDim.x) {
            unsigned int i = row * inner_size + col;
            grad_input[i] = from_float<T>(to_float(grad[i]) - expf(to_float(lsm_out[i])) * sum_grad);
        }
    }

    extern "C" __global__ void log_softmax_fwd_f32(const float* src, float* dst, unsigned int outer_size, unsigned int inner_size) { log_softmax_fwd_kernel(src, dst, outer_size, inner_size); }
    extern "C" __global__ void log_softmax_fwd_f16(const half* src, half* dst, unsigned int outer_size, unsigned int inner_size) { log_softmax_fwd_kernel(src, dst, outer_size, inner_size); }
    extern "C" __global__ void log_softmax_fwd_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int outer_size, unsigned int inner_size) { log_softmax_fwd_kernel(src, dst, outer_size, inner_size); }
    extern "C" __global__ void log_softmax_bwd_f32(const float* grad, const float* lsm_out, float* grad_input, unsigned int outer_size, unsigned int inner_size) { log_softmax_bwd_kernel(grad, lsm_out, grad_input, outer_size, inner_size); }
    extern "C" __global__ void log_softmax_bwd_f16(const half* grad, const half* lsm_out, half* grad_input, unsigned int outer_size, unsigned int inner_size) { log_softmax_bwd_kernel(grad, lsm_out, grad_input, outer_size, inner_size); }
    extern "C" __global__ void log_softmax_bwd_bf16(const __nv_bfloat16* grad, const __nv_bfloat16* lsm_out, __nv_bfloat16* grad_input, unsigned int outer_size, unsigned int inner_size) { log_softmax_bwd_kernel(grad, lsm_out, grad_input, outer_size, inner_size); }

    // Fused RMSNorm forward: dst[row,col] = src[row,col] * rsqrt(mean(src[row]^2) + eps) * w[col].
    // One block per row, fp32 accumulation. When has_weight is 0 the scale is 1 and the
    // weight pointer is ignored (callers pass a valid dummy pointer so launch args stay uniform).
    template <typename T>
    __global__ void rms_norm_fwd_kernel(const T* src, const T* weight, T* dst, unsigned int outer_size, unsigned int inner_size, float eps, unsigned int has_weight) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        __shared__ float smem[REDUCE_THREADS];

        // Pass 1: sum of squares.
        float acc = 0.0f;
        for (unsigned int col = threadIdx.x; col < inner_size; col += blockDim.x) {
            float v = to_float(src[row * inner_size + col]);
            acc += v * v;
        }
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
            if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x < 32) smem[threadIdx.x] = warp_reduce_sum(smem[threadIdx.x]);
        __syncthreads();
        float inv = rsqrtf(smem[0] / (float)inner_size + eps);

        // Pass 2: normalize and scale.
        for (unsigned int col = threadIdx.x; col < inner_size; col += blockDim.x) {
            float v = to_float(src[row * inner_size + col]) * inv;
            if (has_weight) v *= to_float(weight[col]);
            dst[row * inner_size + col] = from_float<T>(v);
        }
    }

    extern "C" __global__ void rms_norm_fwd_f32(const float* src, const float* weight, float* dst, unsigned int outer_size, unsigned int inner_size, float eps, unsigned int has_weight) { rms_norm_fwd_kernel(src, weight, dst, outer_size, inner_size, eps, has_weight); }
    extern "C" __global__ void rms_norm_fwd_f16(const half* src, const half* weight, half* dst, unsigned int outer_size, unsigned int inner_size, float eps, unsigned int has_weight) { rms_norm_fwd_kernel(src, weight, dst, outer_size, inner_size, eps, has_weight); }
    extern "C" __global__ void rms_norm_fwd_bf16(const __nv_bfloat16* src, const __nv_bfloat16* weight, __nv_bfloat16* dst, unsigned int outer_size, unsigned int inner_size, float eps, unsigned int has_weight) { rms_norm_fwd_kernel(src, weight, dst, outer_size, inner_size, eps, has_weight); }

    // Fused RoPE forward over `[B, T, H, D]` rows: y1 = x1*cos - x2*sin,
    // y2 = x1*sin + x2*cos, with cos/sin rows selected by token position.
    // One block per (batch, token, head) row; all math in fp32.
    template <typename T>
    __global__ void rope_fwd_kernel(const T* x, const T* cos, const T* sin, T* dst, unsigned int outer_size, unsigned int head_dim, unsigned int n_heads, unsigned int t_len, unsigned int cos_t_len) {
        unsigned int row = blockIdx.x;
        if (row >= outer_size) return;
        unsigned int half = head_dim / 2;
        unsigned int t = (row / n_heads) % t_len;
        unsigned int ct = t % cos_t_len;
        for (unsigned int col = threadIdx.x; col < head_dim; col += blockDim.x) {
            if (col < half) {
                float x1 = to_float(x[row * head_dim + col]);
                float x2 = to_float(x[row * head_dim + col + half]);
                float c = to_float(cos[ct * half + col]);
                float s = to_float(sin[ct * half + col]);
                dst[row * head_dim + col] = from_float<T>(x1 * c - x2 * s);
            } else {
                unsigned int h = col - half;
                float x1 = to_float(x[row * head_dim + h]);
                float x2 = to_float(x[row * head_dim + col]);
                float c = to_float(cos[ct * half + h]);
                float s = to_float(sin[ct * half + h]);
                dst[row * head_dim + col] = from_float<T>(x1 * s + x2 * c);
            }
        }
    }

    extern "C" __global__ void rope_fwd_f32(const float* x, const float* cos, const float* sin, float* dst, unsigned int outer_size, unsigned int head_dim, unsigned int n_heads, unsigned int t_len, unsigned int cos_t_len) { rope_fwd_kernel(x, cos, sin, dst, outer_size, head_dim, n_heads, t_len, cos_t_len); }
    extern "C" __global__ void rope_fwd_f16(const half* x, const half* cos, const half* sin, half* dst, unsigned int outer_size, unsigned int head_dim, unsigned int n_heads, unsigned int t_len, unsigned int cos_t_len) { rope_fwd_kernel(x, cos, sin, dst, outer_size, head_dim, n_heads, t_len, cos_t_len); }
    extern "C" __global__ void rope_fwd_bf16(const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin, __nv_bfloat16* dst, unsigned int outer_size, unsigned int head_dim, unsigned int n_heads, unsigned int t_len, unsigned int cos_t_len) { rope_fwd_kernel(x, cos, sin, dst, outer_size, head_dim, n_heads, t_len, cos_t_len); }

    // Fused SiLU forward: dst[i] = src[i] / (1 + exp(-src[i])), fp32 math.
    template <typename T>
    __global__ void silu_fwd_kernel(const T* src, T* dst, unsigned int size) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;
        float v = to_float(src[i]);
        dst[i] = from_float<T>(v / (1.0f + expf(-v)));
    }

    extern "C" __global__ void silu_fwd_f32(const float* src, float* dst, unsigned int size) { silu_fwd_kernel(src, dst, size); }
    extern "C" __global__ void silu_fwd_f16(const half* src, half* dst, unsigned int size) { silu_fwd_kernel(src, dst, size); }
    extern "C" __global__ void silu_fwd_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int size) { silu_fwd_kernel(src, dst, size); }

    // Block copy for cache appends: `blocks` contiguous `block_len`-element runs
    // from compact `src`, where destination run `n` starts at
    // `dst_base + n * dst_stride`. Copies one `[B, H, T, D]` slice into the
    // rows `[off, off + T)` of a `[B, H, Cap, D]` buffer without touching the
    // cached prefix.
    template <typename T>
    __global__ void copy_blocks_kernel(const T* src, T* dst, unsigned int total, unsigned int block_len, unsigned int dst_base, unsigned int dst_stride) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= total) return;
        unsigned int b = i / block_len;
        unsigned int j = i % block_len;
        dst[dst_base + b * dst_stride + j] = src[i];
    }

    extern "C" __global__ void copy_blocks_f32(const float* src, float* dst, unsigned int total, unsigned int block_len, unsigned int dst_base, unsigned int dst_stride) { copy_blocks_kernel(src, dst, total, block_len, dst_base, dst_stride); }
    extern "C" __global__ void copy_blocks_f16(const half* src, half* dst, unsigned int total, unsigned int block_len, unsigned int dst_base, unsigned int dst_stride) { copy_blocks_kernel(src, dst, total, block_len, dst_base, dst_stride); }
    extern "C" __global__ void copy_blocks_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int total, unsigned int block_len, unsigned int dst_base, unsigned int dst_stride) { copy_blocks_kernel(src, dst, total, block_len, dst_base, dst_stride); }
    extern "C" __global__ void copy_blocks_i64(const long long* src, long long* dst, unsigned int total, unsigned int block_len, unsigned int dst_base, unsigned int dst_stride) { copy_blocks_kernel(src, dst, total, block_len, dst_base, dst_stride); }

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
    extern "C" __global__ void gather_bf16(const __nv_bfloat16* src, const index_t* indices, __nv_bfloat16* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) { gather_kernel(src, indices, dst, left_len, src_dim, dst_dim, right_len); }
    extern "C" __global__ void scatter_add_f32(const float* src, const index_t* indices, float* dst, unsigned int left_len, unsigned int dst_dim, unsigned int index_len, unsigned int right_len) { scatter_add_kernel(src, indices, dst, left_len, dst_dim, index_len, right_len); }
    extern "C" __global__ void scatter_add_f16(const half* src, const index_t* indices, half* dst, unsigned int left_len, unsigned int dst_dim, unsigned int index_len, unsigned int right_len) { scatter_add_kernel(src, indices, dst, left_len, dst_dim, index_len, right_len); }
    extern "C" __global__ void scatter_add_bf16(const __nv_bfloat16* src, const index_t* indices, __nv_bfloat16* dst, unsigned int left_len, unsigned int dst_dim, unsigned int index_len, unsigned int right_len) { scatter_add_kernel(src, indices, dst, left_len, dst_dim, index_len, right_len); }
    extern "C" __global__ void index_select_f32(const float* src, const index_t* indices, float* dst, unsigned int left_len, unsigned int index_len, unsigned int src_dim, unsigned int right_len) { index_select_kernel(src, indices, dst, left_len, index_len, src_dim, right_len); }
    extern "C" __global__ void index_select_f16(const half* src, const index_t* indices, half* dst, unsigned int left_len, unsigned int index_len, unsigned int src_dim, unsigned int right_len) { index_select_kernel(src, indices, dst, left_len, index_len, src_dim, right_len); }
    extern "C" __global__ void index_select_bf16(const __nv_bfloat16* src, const index_t* indices, __nv_bfloat16* dst, unsigned int left_len, unsigned int index_len, unsigned int src_dim, unsigned int right_len) { index_select_kernel(src, indices, dst, left_len, index_len, src_dim, right_len); }
    extern "C" __global__ void index_add_f32(const float* src, const index_t* indices, float* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) { index_add_kernel(src, indices, dst, left_len, src_dim, dst_dim, right_len); }
    extern "C" __global__ void index_add_f16(const half* src, const index_t* indices, half* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) { index_add_kernel(src, indices, dst, left_len, src_dim, dst_dim, right_len); }
    extern "C" __global__ void index_add_bf16(const __nv_bfloat16* src, const index_t* indices, __nv_bfloat16* dst, unsigned int left_len, unsigned int src_dim, unsigned int dst_dim, unsigned int right_len) { index_add_kernel(src, indices, dst, left_len, src_dim, dst_dim, right_len); }

    extern "C" __global__ void copy_compact_f32(const float* src, float* dst, StridedMeta meta) { copy_compact_kernel(src, dst, meta); }
    extern "C" __global__ void copy_compact_f16(const half* src, half* dst, StridedMeta meta) { copy_compact_kernel(src, dst, meta); }
    extern "C" __global__ void copy_compact_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, StridedMeta meta) { copy_compact_kernel(src, dst, meta); }
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

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ScalarMetaI64 {
        scalar: i64,
    }

    unsafe impl DeviceRepr for ScalarMetaI64 {}

    #[derive(Clone, Debug)]
    pub enum CudaInner {
        F16(CudaSlice<f16>),
        BF16(CudaSlice<bf16>),
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
                [root.join("include"), root.join("targets").join("x86_64-linux").join("include")]
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
                DType::BF16 => CudaInner::BF16(unsafe { alloc_uninit::<bf16>(&runtime, size) }?),
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
                DType::BF16 => CudaInner::BF16(
                    runtime.stream.alloc_zeros::<bf16>(size).expect("cuda alloc failed"),
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
                DType::BF16 => Self::from_cpu_storage(CpuStorage::BF16(vec![bf16::ONE; size])),
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
                CpuStorage::BF16(data) => {
                    CudaInner::BF16(runtime.stream.clone_htod(&data).expect("cuda copy failed"))
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

        /// Copies the `layout` region to a fresh device buffer with no host traffic.
        ///
        /// The source is compacted on the device first when strided, then moved
        /// with a single device-to-device copy. The result is compact.
        pub(crate) fn copy_to_device(&self, layout: &Layout) -> Result<Self> {
            let compact = self.compact(layout)?;
            let mut out = Self::uninit(compact.len(), compact.dtype())?;
            let map_err = |err| Error::Cuda(format!("cuda device copy failed: {err}"));
            match (&compact.inner, &mut out.inner) {
                (CudaInner::F16(src), CudaInner::F16(dst)) => {
                    compact.runtime.stream.memcpy_dtod(src, dst).map_err(map_err)?;
                }
                (CudaInner::BF16(src), CudaInner::BF16(dst)) => {
                    compact.runtime.stream.memcpy_dtod(src, dst).map_err(map_err)?;
                }
                (CudaInner::F32(src), CudaInner::F32(dst)) => {
                    compact.runtime.stream.memcpy_dtod(src, dst).map_err(map_err)?;
                }
                (CudaInner::I64(src), CudaInner::I64(dst)) => {
                    compact.runtime.stream.memcpy_dtod(src, dst).map_err(map_err)?;
                }
                _ => {
                    return Err(Error::DTypeMismatch("to_device: dtype mismatch".into()));
                }
            }
            Ok(out)
        }

        /// Downloads the `layout` region to host memory in a single copy.
        ///
        /// The source is compacted on the device first when strided, so the
        /// returned host buffer is filled directly with no staging allocation.
        pub(crate) fn copy_to_cpu(&self, layout: &Layout) -> Result<CpuStorage> {
            let compact = self.compact(layout)?;
            let map_err = |err| Error::Cuda(format!("cuda download failed: {err}"));
            match &compact.inner {
                CudaInner::F16(src) => {
                    let mut dst = vec![f16::from_f32(0.0); src.len()];
                    compact.runtime.stream.memcpy_dtoh(src, &mut dst[..]).map_err(map_err)?;
                    Ok(CpuStorage::F16(dst))
                }
                CudaInner::BF16(src) => {
                    let mut dst = vec![bf16::ZERO; src.len()];
                    compact.runtime.stream.memcpy_dtoh(src, &mut dst[..]).map_err(map_err)?;
                    Ok(CpuStorage::BF16(dst))
                }
                CudaInner::F32(src) => {
                    let mut dst = vec![0.0f32; src.len()];
                    compact.runtime.stream.memcpy_dtoh(src, &mut dst[..]).map_err(map_err)?;
                    Ok(CpuStorage::F32(dst))
                }
                CudaInner::I64(src) => {
                    let mut dst = vec![0i64; src.len()];
                    compact.runtime.stream.memcpy_dtoh(src, &mut dst[..]).map_err(map_err)?;
                    Ok(CpuStorage::I64(dst))
                }
            }
        }

        /// Uploads the `layout` region from host memory in a single copy.
        ///
        /// Compact sources upload straight from the caller's buffer; strided
        /// sources are compacted into one host staging buffer first.
        pub(crate) fn copy_from_cpu(src: &CpuStorage, layout: &Layout) -> Result<Self> {
            let runtime = runtime()?;
            let map_err = |err| Error::Cuda(format!("cuda upload failed: {err}"));
            let inner = match src {
                CpuStorage::F16(data) => {
                    let staged = CpuStorage::borrow_or_compact(data, src, layout);
                    CudaInner::F16(runtime.stream.clone_htod(&staged[..]).map_err(map_err)?)
                }
                CpuStorage::BF16(data) => {
                    let staged = CpuStorage::borrow_or_compact(data, src, layout);
                    CudaInner::BF16(runtime.stream.clone_htod(&staged[..]).map_err(map_err)?)
                }
                CpuStorage::F32(data) => {
                    let staged = CpuStorage::borrow_or_compact(data, src, layout);
                    CudaInner::F32(runtime.stream.clone_htod(&staged[..]).map_err(map_err)?)
                }
                CpuStorage::I64(data) => {
                    let staged = CpuStorage::borrow_or_compact(data, src, layout);
                    CudaInner::I64(runtime.stream.clone_htod(&staged[..]).map_err(map_err)?)
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
                (CudaInner::BF16(_), CudaInner::BF16(dst)) => {
                    for (part, len) in parts {
                        let CudaInner::BF16(src) = &part.inner else {
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
                CudaInner::BF16(slice) => slice.len(),
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

        fn launch_unary_bf16(&self, kernel: &str, src: &CudaSlice<bf16>) -> Result<Self> {
            let out = unsafe { alloc_uninit::<bf16>(&self.runtime, src.len()) }?;
            let len = src.len() as u32;
            launch_1d!(&self.runtime, kernel, src.len(), src, &out, &len);
            Ok(Self { inner: CudaInner::BF16(out), runtime: self.runtime.clone() })
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

        fn launch_scalar_bf16(
            &self,
            kernel: &str,
            src: &CudaSlice<bf16>,
            scalar: f32,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<bf16>(&self.runtime, src.len()) }?;
            let len = src.len() as u32;
            let meta = ScalarMeta { scalar };
            launch_1d!(&self.runtime, kernel, src.len(), src, &out, &len, &meta);
            Ok(Self { inner: CudaInner::BF16(out), runtime: self.runtime.clone() })
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

        fn launch_binary_bf16(
            &self,
            kernel: &str,
            lhs: &CudaSlice<bf16>,
            rhs: &CudaSlice<bf16>,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<bf16>(&self.runtime, lhs.len()) }?;
            let len = lhs.len() as u32;
            launch_1d!(&self.runtime, kernel, lhs.len(), lhs, rhs, &out, &len);
            Ok(Self { inner: CudaInner::BF16(out), runtime: self.runtime.clone() })
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

        /// Launches a `cast_<src>_<dst>` kernel over a compact source slice,
        /// wrapping the fresh device buffer in the matching [`CudaInner`] variant.
        fn launch_cast<S: DeviceRepr, D: DeviceRepr>(
            &self,
            kernel: &str,
            src: &CudaSlice<S>,
            wrap: impl FnOnce(CudaSlice<D>) -> CudaInner,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<D>(&self.runtime, src.len()) }?;
            let len = src.len() as u32;
            launch_1d!(&self.runtime, kernel, src.len(), src, &out, &len);
            Ok(Self { inner: wrap(out), runtime: self.runtime.clone() })
        }

        fn launch_cmp_i64(&self, kernel: &str, src: &CudaSlice<i64>, scalar: i64) -> Result<Self> {
            let out = unsafe { alloc_uninit::<i64>(&self.runtime, src.len()) }?;
            let len = src.len() as u32;
            let meta = ScalarMetaI64 { scalar };
            launch_1d!(&self.runtime, kernel, src.len(), src, &out, &len, &meta);
            Ok(Self { inner: CudaInner::I64(out), runtime: self.runtime.clone() })
        }

        fn launch_where_f16(
            &self,
            kernel: &str,
            cond: &CudaSlice<f16>,
            on_true: &CudaSlice<f16>,
            on_false: &CudaSlice<f16>,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<f16>(&self.runtime, cond.len()) }?;
            let len = cond.len() as u32;
            launch_1d!(&self.runtime, kernel, cond.len(), cond, on_true, on_false, &out, &len);
            Ok(Self { inner: CudaInner::F16(out), runtime: self.runtime.clone() })
        }

        fn launch_where_bf16(
            &self,
            kernel: &str,
            cond: &CudaSlice<bf16>,
            on_true: &CudaSlice<bf16>,
            on_false: &CudaSlice<bf16>,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<bf16>(&self.runtime, cond.len()) }?;
            let len = cond.len() as u32;
            launch_1d!(&self.runtime, kernel, cond.len(), cond, on_true, on_false, &out, &len);
            Ok(Self { inner: CudaInner::BF16(out), runtime: self.runtime.clone() })
        }

        fn launch_where_f32(
            &self,
            kernel: &str,
            cond: &CudaSlice<f32>,
            on_true: &CudaSlice<f32>,
            on_false: &CudaSlice<f32>,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<f32>(&self.runtime, cond.len()) }?;
            let len = cond.len() as u32;
            launch_1d!(&self.runtime, kernel, cond.len(), cond, on_true, on_false, &out, &len);
            Ok(Self { inner: CudaInner::F32(out), runtime: self.runtime.clone() })
        }

        fn launch_where_i64(
            &self,
            kernel: &str,
            cond: &CudaSlice<i64>,
            on_true: &CudaSlice<i64>,
            on_false: &CudaSlice<i64>,
        ) -> Result<Self> {
            let out = unsafe { alloc_uninit::<i64>(&self.runtime, cond.len()) }?;
            let len = cond.len() as u32;
            launch_1d!(&self.runtime, kernel, cond.len(), cond, on_true, on_false, &out, &len);
            Ok(Self { inner: CudaInner::I64(out), runtime: self.runtime.clone() })
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
                CudaInner::BF16(src) => {
                    let out = unsafe { alloc_uninit::<bf16>(&self.runtime, outer_size) }?;
                    let outer = outer_size as u32;
                    let reduce = reduce_size as u32;
                    launch_reduce!(&self.runtime, kernel, outer_size, src, &out, &outer, &reduce);
                    Ok(Self { inner: CudaInner::BF16(out), runtime: self.runtime.clone() })
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
                CudaInner::BF16(src) => {
                    compact.launch_scalar_bf16("scalar_powf_bf16", src, e as f32)
                }
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
                (CudaInner::BF16(src), "Neg") => compact.launch_unary_bf16("neg_bf16", src),
                (CudaInner::F32(src), "Neg") => compact.launch_unary_f32("neg_f32", src),
                (CudaInner::F16(src), "exp") => compact.launch_unary_f16("exp_f16", src),
                (CudaInner::BF16(src), "exp") => compact.launch_unary_bf16("exp_bf16", src),
                (CudaInner::F32(src), "exp") => compact.launch_unary_f32("exp_f32", src),
                (CudaInner::F16(src), "log") => compact.launch_unary_f16("log_f16", src),
                (CudaInner::BF16(src), "log") => compact.launch_unary_bf16("log_bf16", src),
                (CudaInner::F32(src), "log") => compact.launch_unary_f32("log_f32", src),
                (CudaInner::F16(src), "sin") => compact.launch_unary_f16("sin_f16", src),
                (CudaInner::BF16(src), "sin") => compact.launch_unary_bf16("sin_bf16", src),
                (CudaInner::F32(src), "sin") => compact.launch_unary_f32("sin_f32", src),
                (CudaInner::F16(src), "cos") => compact.launch_unary_f16("cos_f16", src),
                (CudaInner::BF16(src), "cos") => compact.launch_unary_bf16("cos_bf16", src),
                (CudaInner::F32(src), "cos") => compact.launch_unary_f32("cos_f32", src),
                (CudaInner::F16(src), "tanh") => compact.launch_unary_f16("tanh_f16", src),
                (CudaInner::BF16(src), "tanh") => compact.launch_unary_bf16("tanh_bf16", src),
                (CudaInner::F32(src), "tanh") => compact.launch_unary_f32("tanh_f32", src),
                (CudaInner::F16(src), "relu") => compact.launch_unary_f16("relu_f16", src),
                (CudaInner::BF16(src), "relu") => compact.launch_unary_bf16("relu_bf16", src),
                (CudaInner::F32(src), "relu") => compact.launch_unary_f32("relu_f32", src),
                (CudaInner::F16(src), "scalar_add") => {
                    compact.launch_scalar_f16("scalar_add_f16", src, _op.f32(0.0))
                }
                (CudaInner::BF16(src), "scalar_add") => {
                    compact.launch_scalar_bf16("scalar_add_bf16", src, _op.f32(0.0))
                }
                (CudaInner::F32(src), "scalar_add") => {
                    compact.launch_scalar_f32("scalar_add_f32", src, _op.f32(0.0))
                }
                (CudaInner::F16(src), "scalar_mul") => {
                    compact.launch_scalar_f16("scalar_mul_f16", src, _op.f32(1.0))
                }
                (CudaInner::BF16(src), "scalar_mul") => {
                    compact.launch_scalar_bf16("scalar_mul_bf16", src, _op.f32(1.0))
                }
                (CudaInner::F32(src), "scalar_mul") => {
                    compact.launch_scalar_f32("scalar_mul_f32", src, _op.f32(1.0))
                }
                (CudaInner::F16(src), "scalar_div") => {
                    compact.launch_scalar_f16("scalar_div_f16", src, 1.0 / _op.f32(1.0))
                }
                (CudaInner::BF16(src), "scalar_div") => {
                    compact.launch_scalar_bf16("scalar_div_bf16", src, 1.0 / _op.f32(1.0))
                }
                (CudaInner::F32(src), "scalar_div") => {
                    compact.launch_scalar_f32("scalar_div_f32", src, 1.0 / _op.f32(1.0))
                }
                (CudaInner::F16(src), "relu_backward") => {
                    compact.launch_unary_f16("relu_backward_f16", src)
                }
                (CudaInner::BF16(src), "relu_backward") => {
                    compact.launch_unary_bf16("relu_backward_bf16", src)
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
                (CudaInner::BF16(a), CudaInner::BF16(b), "add") => {
                    lhs.launch_binary_bf16("add_bf16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "add") => {
                    lhs.launch_binary_f32("add_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "sub") => {
                    lhs.launch_binary_f16("sub_f16", a, b)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "sub") => {
                    lhs.launch_binary_bf16("sub_bf16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "sub") => {
                    lhs.launch_binary_f32("sub_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "mul") => {
                    lhs.launch_binary_f16("mul_f16", a, b)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "mul") => {
                    lhs.launch_binary_bf16("mul_bf16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "mul") => {
                    lhs.launch_binary_f32("mul_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "div") => {
                    lhs.launch_binary_f16("div_f16", a, b)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "div") => {
                    lhs.launch_binary_bf16("div_bf16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "div") => {
                    lhs.launch_binary_f32("div_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "powf") => {
                    lhs.launch_binary_f16("powf_f16", a, b)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "powf") => {
                    lhs.launch_binary_bf16("powf_bf16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "powf") => {
                    lhs.launch_binary_f32("powf_f32", a, b)
                }
                (CudaInner::F16(a), CudaInner::F16(b), "eq") => {
                    lhs.launch_binary_f16("eq_f16", a, b)
                }
                (CudaInner::BF16(a), CudaInner::BF16(b), "eq") => {
                    lhs.launch_binary_bf16("eq_bf16", a, b)
                }
                (CudaInner::F32(a), CudaInner::F32(b), "eq") => {
                    lhs.launch_binary_f32("eq_f32", a, b)
                }
                _ => Err(Error::NotImplemented("cuda binary op is not implemented for this dtype")),
            }
        }

        fn ne_scalar(&self, layout: &Layout, scalar: f64) -> Result<Self> {
            let compact = self.compact(layout)?;
            match &compact.inner {
                CudaInner::F16(src) => {
                    compact.launch_scalar_f16("ne_scalar_f16", src, scalar as f32)
                }
                CudaInner::BF16(src) => {
                    compact.launch_scalar_bf16("ne_scalar_bf16", src, scalar as f32)
                }
                CudaInner::F32(src) => {
                    compact.launch_scalar_f32("ne_scalar_f32", src, scalar as f32)
                }
                CudaInner::I64(_) => {
                    Err(Error::DTypeMismatch("ne_scalar requires float dtype".into()))
                }
            }
        }

        fn ne_scalar_i64(&self, layout: &Layout, scalar: i64) -> Result<Self> {
            let compact = self.compact(layout)?;
            match &compact.inner {
                CudaInner::I64(src) => compact.launch_cmp_i64("ne_scalar_i64", src, scalar),
                _ => Err(Error::DTypeMismatch("ne_scalar_i64 requires i64 dtype".into())),
            }
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
            let on_true = on_true.compact(true_layout)?;
            let on_false = on_false.compact(false_layout)?;
            match (&cond.inner, &on_true.inner, &on_false.inner) {
                (CudaInner::F16(c), CudaInner::F16(t), CudaInner::F16(f)) => {
                    cond.launch_where_f16("where_f16", c, t, f)
                }
                (CudaInner::BF16(c), CudaInner::BF16(t), CudaInner::BF16(f)) => {
                    cond.launch_where_bf16("where_bf16", c, t, f)
                }
                (CudaInner::F32(c), CudaInner::F32(t), CudaInner::F32(f)) => {
                    cond.launch_where_f32("where_f32", c, t, f)
                }
                (CudaInner::I64(c), CudaInner::I64(t), CudaInner::I64(f)) => {
                    cond.launch_where_i64("where_i64", c, t, f)
                }
                _ => Err(Error::NotImplemented("cuda select is not implemented for this dtype")),
            }
        }

        fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
            let compact = self.compact(layout)?;
            let reduced = match O::KERNEL {
                "reduce_sum" => compact.reduce_impl(
                    match compact.dtype() {
                        DType::F16 => "reduce_sum_f16",
                        DType::BF16 => "reduce_sum_bf16",
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
                        DType::BF16 => "reduce_max_bf16",
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

            /// Returns `(op, leading_dim, batch_stride)` for a matrix layout if it can be
            /// passed to cuBLAS without compacting — i.e., it is row-major contiguous or
            /// has only the last two dims transposed with contiguous batch dims.
            /// Uses the same convention as candle: the last two strides determine the op.
            fn try_gemm_params(
                layout: &Layout,
                rows: usize,
                cols: usize,
            ) -> Option<(cublasOperation_t, i32, i64)> {
                let ndim = layout.ndim();
                let strides = layout.strides();
                let shape = layout.shape();
                let m1 = strides[ndim - 1] as usize; // last stride
                let m2 = strides[ndim - 2] as usize; // second-to-last stride
                // Batch dims must be contiguous.
                let mut expected = rows * cols;
                for i in (0..ndim.saturating_sub(2)).rev() {
                    if strides[i] as usize != expected {
                        return None;
                    }
                    expected *= shape[i];
                }
                if (m1 == 1 || cols == 1) && (m2 == cols || rows == 1) {
                    // Row-major contiguous: CUBLAS_OP_N, leading_dim = cols
                    Some((cublasOperation_t::CUBLAS_OP_N, cols as i32, (rows * cols) as i64))
                } else if (m1 == rows || cols == 1) && (m2 == 1 || rows == 1) {
                    // Transposed contiguous: CUBLAS_OP_T, leading_dim = rows
                    Some((cublasOperation_t::CUBLAS_OP_T, rows as i32, (rows * cols) as i64))
                } else {
                    None
                }
            }

            let ndim = layout.ndim();
            let m = layout.shape()[ndim - 2];
            let k = layout.shape()[ndim - 1];
            let n = layout_other.shape()[ndim - 1];
            let batch = if ndim > 2 { layout.shape().iter().take(ndim - 2).product() } else { 1 };

            // Try to use each operand directly (CUBLAS_OP_T for transposed layouts) to avoid
            // copying. Fall back to compact for layouts with non-standard strides.
            let lhs_compact: Option<CudaStorage> = if try_gemm_params(layout, m, k).is_none() {
                Some(self.compact(layout)?)
            } else {
                None
            };
            let rhs_compact: Option<CudaStorage> = if try_gemm_params(layout_other, k, n).is_none()
            {
                Some(other.compact(layout_other)?)
            } else {
                None
            };
            let lhs_storage: &CudaStorage = lhs_compact.as_ref().unwrap_or(self);
            let rhs_storage: &CudaStorage = rhs_compact.as_ref().unwrap_or(other);
            // If we compacted, the result is always normal row-major (CUBLAS_OP_N, offset=0).
            let (transb, ldb, lhs_bs, lhs_offset) = if lhs_compact.is_some() {
                (cublasOperation_t::CUBLAS_OP_N, k as i32, (m * k) as i64, 0usize)
            } else {
                let (op, ld, bs) = try_gemm_params(layout, m, k).unwrap();
                (op, ld, bs, layout.offset)
            };
            let (transa, lda, rhs_bs, rhs_offset) = if rhs_compact.is_some() {
                (cublasOperation_t::CUBLAS_OP_N, n as i32, (k * n) as i64, 0usize)
            } else {
                let (op, ld, bs) = try_gemm_params(layout_other, k, n).unwrap();
                (op, ld, bs, layout_other.offset)
            };

            match (&lhs_storage.inner, &rhs_storage.inner) {
                (CudaInner::F32(a), CudaInner::F32(b)) => {
                    // cuBLAS with beta=0 overwrites every output element; no zeroing needed.
                    let mut out =
                        unsafe { alloc_uninit::<f32>(&lhs_storage.runtime, batch * m * n) }?;
                    let alpha = 1.0f32;
                    let beta = 0.0f32;
                    let alpha_ptr = &alpha as *const f32 as *const _;
                    let beta_ptr = &beta as *const f32 as *const _;
                    let stream = out.stream().clone();
                    let b_view = b.slice(rhs_offset..);
                    let a_view = a.slice(lhs_offset..);
                    let (b_ptr, gb) = b_view.device_ptr(&stream);
                    let (a_ptr, ga) = a_view.device_ptr(&stream);
                    let (c_ptr, gc) = out.device_ptr_mut(&stream);
                    maybe_profile_launch(&lhs_storage.runtime, || {
                        unsafe {
                            cublas_result::gemm_strided_batched_ex(
                                *lhs_storage.runtime.blas.handle(),
                                transa,
                                transb,
                                n as i32,
                                m as i32,
                                k as i32,
                                alpha_ptr,
                                b_ptr as *const _,
                                cudaDataType_t::CUDA_R_32F,
                                lda,
                                rhs_bs,
                                a_ptr as *const _,
                                cudaDataType_t::CUDA_R_32F,
                                ldb,
                                lhs_bs,
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
                    Ok(Self { inner: CudaInner::F32(out), runtime: lhs_storage.runtime.clone() })
                }
                (CudaInner::F16(a), CudaInner::F16(b)) => {
                    let mut out =
                        unsafe { alloc_uninit::<f16>(&lhs_storage.runtime, batch * m * n) }?;
                    let alpha_f32 = 1.0f32;
                    let beta_f32 = 0.0f32;
                    let alpha_ptr = &alpha_f32 as *const f32 as *const _;
                    let beta_ptr = &beta_f32 as *const f32 as *const _;
                    let stream = out.stream().clone();
                    let b_view = b.slice(rhs_offset..);
                    let a_view = a.slice(lhs_offset..);
                    let (b_ptr, gb) = b_view.device_ptr(&stream);
                    let (a_ptr, ga) = a_view.device_ptr(&stream);
                    let (c_ptr, gc) = out.device_ptr_mut(&stream);
                    maybe_profile_launch(&lhs_storage.runtime, || {
                        unsafe {
                            cublas_result::gemm_strided_batched_ex(
                                *lhs_storage.runtime.blas.handle(),
                                transa,
                                transb,
                                n as i32,
                                m as i32,
                                k as i32,
                                alpha_ptr,
                                b_ptr as *const _,
                                cudaDataType_t::CUDA_R_16F,
                                lda,
                                rhs_bs,
                                a_ptr as *const _,
                                cudaDataType_t::CUDA_R_16F,
                                ldb,
                                lhs_bs,
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
                    Ok(Self { inner: CudaInner::F16(out), runtime: lhs_storage.runtime.clone() })
                }
                (CudaInner::BF16(a), CudaInner::BF16(b)) => {
                    let mut out =
                        unsafe { alloc_uninit::<bf16>(&lhs_storage.runtime, batch * m * n) }?;
                    let alpha_f32 = 1.0f32;
                    let beta_f32 = 0.0f32;
                    let alpha_ptr = &alpha_f32 as *const f32 as *const _;
                    let beta_ptr = &beta_f32 as *const f32 as *const _;
                    let stream = out.stream().clone();
                    let b_view = b.slice(rhs_offset..);
                    let a_view = a.slice(lhs_offset..);
                    let (b_ptr, gb) = b_view.device_ptr(&stream);
                    let (a_ptr, ga) = a_view.device_ptr(&stream);
                    let (c_ptr, gc) = out.device_ptr_mut(&stream);
                    maybe_profile_launch(&lhs_storage.runtime, || {
                        unsafe {
                            cublas_result::gemm_strided_batched_ex(
                                *lhs_storage.runtime.blas.handle(),
                                transa,
                                transb,
                                n as i32,
                                m as i32,
                                k as i32,
                                alpha_ptr,
                                b_ptr as *const _,
                                cudaDataType_t::CUDA_R_16BF,
                                lda,
                                rhs_bs,
                                a_ptr as *const _,
                                cudaDataType_t::CUDA_R_16BF,
                                ldb,
                                lhs_bs,
                                beta_ptr,
                                c_ptr as *mut _,
                                cudaDataType_t::CUDA_R_16BF,
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
                    Ok(Self { inner: CudaInner::BF16(out), runtime: lhs_storage.runtime.clone() })
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
                (CudaInner::BF16(src), CudaInner::I64(indices)) => {
                    let out =
                        unsafe { alloc_uninit::<bf16>(&self.runtime, indices_layout.size()) }?;
                    let left = left_len as u32;
                    let src_dim_u32 = src_dim as u32;
                    let dst_dim_u32 = dst_dim as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "gather_bf16",
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
                    Ok(Self { inner: CudaInner::BF16(out), runtime: self.runtime.clone() })
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
                (CudaInner::BF16(src), CudaInner::I64(indices)) => {
                    let out = src
                        .stream()
                        .alloc_zeros::<bf16>(dst_shape.iter().product())
                        .map_err(|err| Error::Cuda(format!("cuda alloc failed: {err}")))?;
                    let left = left_len as u32;
                    let dst_dim_u32 = dst_dim as u32;
                    let index_len_u32 = index_len as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "scatter_add_bf16",
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
                    Ok(Self { inner: CudaInner::BF16(out), runtime: self.runtime.clone() })
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
                    let out = unsafe {
                        alloc_uninit::<f16>(&self.runtime, left_len * index_len * right_len)
                    }?;
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
                (CudaInner::BF16(src), CudaInner::I64(indices)) => {
                    let out = unsafe {
                        alloc_uninit::<bf16>(&self.runtime, left_len * index_len * right_len)
                    }?;
                    let left = left_len as u32;
                    let index_len_u32 = index_len as u32;
                    let src_dim_u32 = src_dim as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "index_select_bf16",
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
                    Ok(Self { inner: CudaInner::BF16(out), runtime: self.runtime.clone() })
                }
                (CudaInner::F32(src), CudaInner::I64(indices)) => {
                    let out = unsafe {
                        alloc_uninit::<f32>(&self.runtime, left_len * index_len * right_len)
                    }?;
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
                (CudaInner::BF16(src), CudaInner::I64(indices)) => {
                    let out = src
                        .stream()
                        .alloc_zeros::<bf16>(dst_shape.iter().product())
                        .map_err(|err| Error::Cuda(format!("cuda alloc failed: {err}")))?;
                    let left = left_len as u32;
                    let src_dim_u32 = src_dim as u32;
                    let dst_dim_u32 = dst_dim as u32;
                    let right = right_len as u32;
                    launch_2d!(
                        &self.runtime,
                        "index_add_bf16",
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
                    Ok(Self { inner: CudaInner::BF16(out), runtime: self.runtime.clone() })
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
                    DType::BF16 => "log_sum_exp_bf16",
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

        /// Fused log-softmax forward: `dst[i] = src[i] - log(sum_j exp(src[j]))` per row.
        ///
        /// `outer_size * inner_size` must equal `layout.size()`. The input must be compact.
        fn log_softmax_fwd(
            &self,
            layout: &Layout,
            outer_size: usize,
            inner_size: usize,
        ) -> Result<Self> {
            let src = self.compact(layout)?;
            match &src.inner {
                CudaInner::F16(s) => {
                    let out =
                        unsafe { alloc_uninit::<f16>(&src.runtime, outer_size * inner_size) }?;
                    let outer = outer_size as u32;
                    let inner = inner_size as u32;
                    launch_reduce!(
                        &src.runtime,
                        "log_softmax_fwd_f16",
                        outer_size,
                        s,
                        &out,
                        &outer,
                        &inner
                    );
                    Ok(Self { inner: CudaInner::F16(out), runtime: src.runtime.clone() })
                }
                CudaInner::BF16(s) => {
                    let out =
                        unsafe { alloc_uninit::<bf16>(&src.runtime, outer_size * inner_size) }?;
                    let outer = outer_size as u32;
                    let inner = inner_size as u32;
                    launch_reduce!(
                        &src.runtime,
                        "log_softmax_fwd_bf16",
                        outer_size,
                        s,
                        &out,
                        &outer,
                        &inner
                    );
                    Ok(Self { inner: CudaInner::BF16(out), runtime: src.runtime.clone() })
                }
                CudaInner::F32(s) => {
                    let out =
                        unsafe { alloc_uninit::<f32>(&src.runtime, outer_size * inner_size) }?;
                    let outer = outer_size as u32;
                    let inner = inner_size as u32;
                    launch_reduce!(
                        &src.runtime,
                        "log_softmax_fwd_f32",
                        outer_size,
                        s,
                        &out,
                        &outer,
                        &inner
                    );
                    Ok(Self { inner: CudaInner::F32(out), runtime: src.runtime.clone() })
                }
                CudaInner::I64(_) => {
                    Err(Error::NotImplemented("cuda log_softmax_fwd for i64 is not implemented"))
                }
            }
        }

        /// Fused log-softmax backward: `grad_input[i] = grad[i] - exp(lsm[i]) * sum_j grad[j]` per row.
        ///
        /// `lsm` is the saved log-softmax output from the forward pass.
        fn log_softmax_bwd(
            &self,
            grad_layout: &Layout,
            lsm: &Self,
            lsm_layout: &Layout,
            outer_size: usize,
            inner_size: usize,
        ) -> Result<Self> {
            let grad = self.compact(grad_layout)?;
            let lsm = lsm.compact(lsm_layout)?;
            match (&grad.inner, &lsm.inner) {
                (CudaInner::F16(g), CudaInner::F16(l)) => {
                    let out =
                        unsafe { alloc_uninit::<f16>(&grad.runtime, outer_size * inner_size) }?;
                    let outer = outer_size as u32;
                    let inner = inner_size as u32;
                    launch_reduce!(
                        &grad.runtime,
                        "log_softmax_bwd_f16",
                        outer_size,
                        g,
                        l,
                        &out,
                        &outer,
                        &inner
                    );
                    Ok(Self { inner: CudaInner::F16(out), runtime: grad.runtime.clone() })
                }
                (CudaInner::BF16(g), CudaInner::BF16(l)) => {
                    let out =
                        unsafe { alloc_uninit::<bf16>(&grad.runtime, outer_size * inner_size) }?;
                    let outer = outer_size as u32;
                    let inner = inner_size as u32;
                    launch_reduce!(
                        &grad.runtime,
                        "log_softmax_bwd_bf16",
                        outer_size,
                        g,
                        l,
                        &out,
                        &outer,
                        &inner
                    );
                    Ok(Self { inner: CudaInner::BF16(out), runtime: grad.runtime.clone() })
                }
                (CudaInner::F32(g), CudaInner::F32(l)) => {
                    let out =
                        unsafe { alloc_uninit::<f32>(&grad.runtime, outer_size * inner_size) }?;
                    let outer = outer_size as u32;
                    let inner = inner_size as u32;
                    launch_reduce!(
                        &grad.runtime,
                        "log_softmax_bwd_f32",
                        outer_size,
                        g,
                        l,
                        &out,
                        &outer,
                        &inner
                    );
                    Ok(Self { inner: CudaInner::F32(out), runtime: grad.runtime.clone() })
                }
                _ => Err(Error::DTypeMismatch(
                    "log_softmax_bwd: dtype mismatch between grad and lsm".into(),
                )),
            }
        }

        /// Copies `blocks` contiguous `block_len`-element runs from compact `src`
        /// into `self`, where destination run `n` starts at
        /// `dst_base + n * dst_stride`.
        ///
        /// `src_layout.size()` must equal `blocks * block_len`. A strided source
        /// is compacted first, so callers pass any layout.
        fn copy_blocks_into(
            &mut self,
            src: &Self,
            src_layout: &Layout,
            blocks: usize,
            block_len: usize,
            dst_base: usize,
            dst_stride: usize,
        ) -> Result<()> {
            let compact = src.compact(src_layout)?;
            assert_eq!(src_layout.size(), blocks * block_len);
            let total = src_layout.size();
            let args = [
                total as u32,
                block_len as u32,
                dst_base as u32,
                dst_stride as u32,
            ];
            match (&compact.inner, &mut self.inner) {
                (CudaInner::F16(s), CudaInner::F16(d)) => {
                    launch_1d!(
                        &compact.runtime, "copy_blocks_f16", total,
                        s, d, &args[0], &args[1], &args[2], &args[3]
                    );
                    Ok(())
                }
                (CudaInner::BF16(s), CudaInner::BF16(d)) => {
                    launch_1d!(
                        &compact.runtime, "copy_blocks_bf16", total,
                        s, d, &args[0], &args[1], &args[2], &args[3]
                    );
                    Ok(())
                }
                (CudaInner::F32(s), CudaInner::F32(d)) => {
                    launch_1d!(
                        &compact.runtime, "copy_blocks_f32", total,
                        s, d, &args[0], &args[1], &args[2], &args[3]
                    );
                    Ok(())
                }
                (CudaInner::I64(s), CudaInner::I64(d)) => {
                    launch_1d!(
                        &compact.runtime, "copy_blocks_i64", total,
                        s, d, &args[0], &args[1], &args[2], &args[3]
                    );
                    Ok(())
                }
                _ => Err(Error::DTypeMismatch(
                    "copy_blocks_into: dtype mismatch between source and destination".into(),
                )),
            }
        }

        /// Fused SiLU forward: `dst[i] = src[i] / (1 + exp(-src[i]))`.
        ///
        /// `layout.size()` elements are read in compact order and written to a
        /// fresh compact buffer.
        fn silu_fwd(&self, layout: &Layout) -> Result<Self> {
            let src = self.compact(layout)?;
            let len = layout.size();
            let len_u32 = len as u32;
            match &src.inner {
                CudaInner::F16(s) => {
                    let out = unsafe { alloc_uninit::<f16>(&src.runtime, len) }?;
                    launch_1d!(&src.runtime, "silu_fwd_f16", len, s, &out, &len_u32);
                    Ok(Self { inner: CudaInner::F16(out), runtime: src.runtime.clone() })
                }
                CudaInner::BF16(s) => {
                    let out = unsafe { alloc_uninit::<bf16>(&src.runtime, len) }?;
                    launch_1d!(&src.runtime, "silu_fwd_bf16", len, s, &out, &len_u32);
                    Ok(Self { inner: CudaInner::BF16(out), runtime: src.runtime.clone() })
                }
                CudaInner::F32(s) => {
                    let out = unsafe { alloc_uninit::<f32>(&src.runtime, len) }?;
                    launch_1d!(&src.runtime, "silu_fwd_f32", len, s, &out, &len_u32);
                    Ok(Self { inner: CudaInner::F32(out), runtime: src.runtime.clone() })
                }
                CudaInner::I64(_) => {
                    Err(Error::NotImplemented("cuda silu_fwd for i64 is not implemented"))
                }
            }
        }
        /// `y2 = x1*sin + x2*cos`, with the cos/sin row selected per token.
        ///
        /// `outer_size * head_dim` must equal `layout.size()`. Every input is
        /// compacted first; cos/sin hold `cos_t_len` rows of `head_dim / 2`.
        #[allow(clippy::too_many_arguments)]
        fn rope_fwd(
            &self,
            layout: &Layout,
            cos: &Self,
            cos_layout: &Layout,
            sin: &Self,
            sin_layout: &Layout,
            outer_size: usize,
            head_dim: usize,
            n_heads: usize,
            t_len: usize,
            cos_t_len: usize,
        ) -> Result<Self> {
            let x = self.compact(layout)?;
            let cos_c = cos.compact(cos_layout)?;
            let sin_c = sin.compact(sin_layout)?;
            let dims = [
                outer_size as u32,
                head_dim as u32,
                n_heads as u32,
                t_len as u32,
                cos_t_len as u32,
            ];
            match (&x.inner, &cos_c.inner, &sin_c.inner) {
                (CudaInner::F16(xs), CudaInner::F16(cc), CudaInner::F16(ss)) => {
                    let out =
                        unsafe { alloc_uninit::<f16>(&x.runtime, outer_size * head_dim) }?;
                    launch_reduce!(
                        &x.runtime, "rope_fwd_f16", outer_size,
                        xs, cc, ss, &out,
                        &dims[0], &dims[1], &dims[2], &dims[3], &dims[4]
                    );
                    Ok(Self { inner: CudaInner::F16(out), runtime: x.runtime.clone() })
                }
                (CudaInner::BF16(xs), CudaInner::BF16(cc), CudaInner::BF16(ss)) => {
                    let out =
                        unsafe { alloc_uninit::<bf16>(&x.runtime, outer_size * head_dim) }?;
                    launch_reduce!(
                        &x.runtime, "rope_fwd_bf16", outer_size,
                        xs, cc, ss, &out,
                        &dims[0], &dims[1], &dims[2], &dims[3], &dims[4]
                    );
                    Ok(Self { inner: CudaInner::BF16(out), runtime: x.runtime.clone() })
                }
                (CudaInner::F32(xs), CudaInner::F32(cc), CudaInner::F32(ss)) => {
                    let out =
                        unsafe { alloc_uninit::<f32>(&x.runtime, outer_size * head_dim) }?;
                    launch_reduce!(
                        &x.runtime, "rope_fwd_f32", outer_size,
                        xs, cc, ss, &out,
                        &dims[0], &dims[1], &dims[2], &dims[3], &dims[4]
                    );
                    Ok(Self { inner: CudaInner::F32(out), runtime: x.runtime.clone() })
                }
                _ => Err(Error::DTypeMismatch(
                    "rope_fwd: dtype mismatch between input, cos, and sin".into(),
                )),
            }
        }
        ///
        /// `outer_size * inner_size` must equal `layout.size()`. The input is
        /// compacted first; `weight` (if any) must hold `inner_size` elements of
        /// the same dtype and is compacted too, so the kernel reads `w[col]`.
        fn rms_norm_fwd(
            &self,
            layout: &Layout,
            weight: Option<(&Self, &Layout)>,
            outer_size: usize,
            inner_size: usize,
            eps: f32,
        ) -> Result<Self> {
            let src = self.compact(layout)?;
            let weight_c = weight
                .map(|(w, w_layout)| w.compact(w_layout))
                .transpose()?;
            match (&src.inner, weight_c.as_ref().map(|w| &w.inner)) {
                (CudaInner::F16(s), None) => {
                    let out =
                        unsafe { alloc_uninit::<f16>(&src.runtime, outer_size * inner_size) }?;
                    let (outer, inner, eps, has_weight) =
                        (outer_size as u32, inner_size as u32, eps, 0u32);
                    // No scale: the weight slot reuses the input pointer as a valid
                    // dummy because the kernel never reads it when has_weight is 0.
                    launch_reduce!(
                        &src.runtime,
                        "rms_norm_fwd_f16",
                        outer_size,
                        s,
                        s,
                        &out,
                        &outer,
                        &inner,
                        &eps,
                        &has_weight
                    );
                    Ok(Self { inner: CudaInner::F16(out), runtime: src.runtime.clone() })
                }
                (CudaInner::F16(s), Some(CudaInner::F16(w))) => {
                    let out =
                        unsafe { alloc_uninit::<f16>(&src.runtime, outer_size * inner_size) }?;
                    let (outer, inner, eps, has_weight) =
                        (outer_size as u32, inner_size as u32, eps, 1u32);
                    launch_reduce!(
                        &src.runtime,
                        "rms_norm_fwd_f16",
                        outer_size,
                        s,
                        w,
                        &out,
                        &outer,
                        &inner,
                        &eps,
                        &has_weight
                    );
                    Ok(Self { inner: CudaInner::F16(out), runtime: src.runtime.clone() })
                }
                (CudaInner::BF16(s), None) => {
                    let out =
                        unsafe { alloc_uninit::<bf16>(&src.runtime, outer_size * inner_size) }?;
                    let (outer, inner, eps, has_weight) =
                        (outer_size as u32, inner_size as u32, eps, 0u32);
                    launch_reduce!(
                        &src.runtime,
                        "rms_norm_fwd_bf16",
                        outer_size,
                        s,
                        s,
                        &out,
                        &outer,
                        &inner,
                        &eps,
                        &has_weight
                    );
                    Ok(Self { inner: CudaInner::BF16(out), runtime: src.runtime.clone() })
                }
                (CudaInner::BF16(s), Some(CudaInner::BF16(w))) => {
                    let out =
                        unsafe { alloc_uninit::<bf16>(&src.runtime, outer_size * inner_size) }?;
                    let (outer, inner, eps, has_weight) =
                        (outer_size as u32, inner_size as u32, eps, 1u32);
                    launch_reduce!(
                        &src.runtime,
                        "rms_norm_fwd_bf16",
                        outer_size,
                        s,
                        w,
                        &out,
                        &outer,
                        &inner,
                        &eps,
                        &has_weight
                    );
                    Ok(Self { inner: CudaInner::BF16(out), runtime: src.runtime.clone() })
                }
                (CudaInner::F32(s), None) => {
                    let out =
                        unsafe { alloc_uninit::<f32>(&src.runtime, outer_size * inner_size) }?;
                    let (outer, inner, eps, has_weight) =
                        (outer_size as u32, inner_size as u32, eps, 0u32);
                    launch_reduce!(
                        &src.runtime,
                        "rms_norm_fwd_f32",
                        outer_size,
                        s,
                        s,
                        &out,
                        &outer,
                        &inner,
                        &eps,
                        &has_weight
                    );
                    Ok(Self { inner: CudaInner::F32(out), runtime: src.runtime.clone() })
                }
                (CudaInner::F32(s), Some(CudaInner::F32(w))) => {
                    let out =
                        unsafe { alloc_uninit::<f32>(&src.runtime, outer_size * inner_size) }?;
                    let (outer, inner, eps, has_weight) =
                        (outer_size as u32, inner_size as u32, eps, 1u32);
                    launch_reduce!(
                        &src.runtime,
                        "rms_norm_fwd_f32",
                        outer_size,
                        s,
                        w,
                        &out,
                        &outer,
                        &inner,
                        &eps,
                        &has_weight
                    );
                    Ok(Self { inner: CudaInner::F32(out), runtime: src.runtime.clone() })
                }
                _ => Err(Error::DTypeMismatch(
                    "rms_norm_fwd: dtype mismatch between input and weight".into(),
                )),
            }
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
            // Compacting first keeps the cast kernels to a single compact read/write
            // each, and the copy itself stays on-device via the copy_compact kernel.
            let compact = self.compact(layout)?;
            if compact.dtype() == dtype {
                return Ok(compact);
            }
            match (&compact.inner, dtype) {
                (CudaInner::F16(src), DType::F32) => {
                    compact.launch_cast("cast_f16_f32", src, CudaInner::F32)
                }
                (CudaInner::F16(src), DType::BF16) => {
                    compact.launch_cast("cast_f16_bf16", src, CudaInner::BF16)
                }
                (CudaInner::F16(src), DType::I64) => {
                    compact.launch_cast("cast_f16_i64", src, CudaInner::I64)
                }
                (CudaInner::BF16(src), DType::F16) => {
                    compact.launch_cast("cast_bf16_f16", src, CudaInner::F16)
                }
                (CudaInner::BF16(src), DType::F32) => {
                    compact.launch_cast("cast_bf16_f32", src, CudaInner::F32)
                }
                (CudaInner::BF16(src), DType::I64) => {
                    compact.launch_cast("cast_bf16_i64", src, CudaInner::I64)
                }
                (CudaInner::F32(src), DType::F16) => {
                    compact.launch_cast("cast_f32_f16", src, CudaInner::F16)
                }
                (CudaInner::F32(src), DType::BF16) => {
                    compact.launch_cast("cast_f32_bf16", src, CudaInner::BF16)
                }
                (CudaInner::F32(src), DType::I64) => {
                    compact.launch_cast("cast_f32_i64", src, CudaInner::I64)
                }
                (CudaInner::I64(src), DType::F16) => {
                    compact.launch_cast("cast_i64_f16", src, CudaInner::F16)
                }
                (CudaInner::I64(src), DType::BF16) => {
                    compact.launch_cast("cast_i64_bf16", src, CudaInner::BF16)
                }
                (CudaInner::I64(src), DType::F32) => {
                    compact.launch_cast("cast_i64_f32", src, CudaInner::F32)
                }
                _ => unreachable!("same-dtype casts return early"),
            }
        }

        fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
            let layout = layout.borrow();
            let compact = self.compact(layout).expect("cuda compact failed");
            match &compact.inner {
                CudaInner::F16(slice) => D::to_vec(&CpuStorage::F16(
                    compact.runtime.stream.clone_dtoh(slice).expect("cuda dtoh failed"),
                )),
                CudaInner::BF16(slice) => D::to_vec(&CpuStorage::BF16(
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
                (CudaInner::BF16(src), CudaInner::BF16(dst)) => {
                    launch_1d!(
                        &self.runtime,
                        "copy_compact_bf16",
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

        pub(crate) fn copy_to_device(&self, _: &Layout) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }

        pub(crate) fn copy_to_cpu(&self, _: &Layout) -> Result<CpuStorage> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }

        pub(crate) fn copy_from_cpu(_src: &CpuStorage, _: &Layout) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
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
        fn ne_scalar(&self, _: &Layout, _: f64) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn ne_scalar_i64(&self, _: &Layout, _: i64) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn select(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
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
        fn log_softmax_fwd(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn log_softmax_bwd(
            &self,
            _: &Layout,
            _: &Self,
            _: &Layout,
            _: usize,
            _: usize,
        ) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn rms_norm_fwd(
            &self,
            _: &Layout,
            _: Option<(&Self, &Layout)>,
            _: usize,
            _: usize,
            _: f32,
        ) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        #[allow(clippy::too_many_arguments)]
        fn rope_fwd(
            &self,
            _: &Layout,
            _: &Self,
            _: &Layout,
            _: &Self,
            _: &Layout,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
        ) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn silu_fwd(&self, _: &Layout) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn copy_blocks_into(
            &mut self,
            _: &Self,
            _: &Layout,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
        ) -> Result<()> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
        }
        fn dtype(&self) -> DType {
            panic!("cuda backend is unavailable")
        }
        fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
            Err(Error::NotImplemented("cuda backend is unavailable"))
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
