#include <metal_stdlib>
using namespace metal;

constant uint MAX_DIMS = 8;

// --- Metadata structs (must match Rust repr(C) counterparts) ---

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
    uint k;
    uint n;
    uint batch;
};

struct GatherScatterMeta {
    uint left_len;
    uint src_dim;
    uint dst_dim;
    uint right_len;
};

struct IndexSelectMeta {
    uint left_len;
    uint index_len;
    uint src_dim;
    uint right_len;
};

struct IndexAddMeta {
    uint left_len;
    uint src_dim;
    uint dst_dim;
    uint right_len;
};

// --- Strided indexing ---

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

// --- Unary ops ---

kernel void neg_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = -input[linear_to_offset(id, meta)];
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

kernel void exp_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
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

kernel void log_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = log(input[linear_to_offset(id, meta)]);
}

kernel void sin_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = sin(input[linear_to_offset(id, meta)]);
}

kernel void sin_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = sin(input[linear_to_offset(id, meta)]);
}

kernel void cos_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = cos(input[linear_to_offset(id, meta)]);
}

kernel void cos_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = cos(input[linear_to_offset(id, meta)]);
}

kernel void tanh_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = tanh(input[linear_to_offset(id, meta)]);
}

kernel void tanh_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = tanh(input[linear_to_offset(id, meta)]);
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

kernel void relu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    half v = input[linear_to_offset(id, meta)];
    output[id] = max(v, half(0.0h));
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

kernel void relu_backward_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = input[linear_to_offset(id, meta)] > half(0.0h) ? half(1.0h) : half(0.0h);
}

// --- Scalar ops ---

kernel void scalar_add_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ScalarMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.input.size) return;
    output[id] = input[linear_to_offset(id, meta.input)] + half(meta.scalar);
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

kernel void scalar_mul_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ScalarMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.input.size) return;
    output[id] = input[linear_to_offset(id, meta.input)] * half(meta.scalar);
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

kernel void scalar_div_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ScalarMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.input.size) return;
    output[id] = input[linear_to_offset(id, meta.input)] / half(meta.scalar);
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

kernel void scalar_powf_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ScalarMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.input.size) return;
    output[id] = half(pow(float(input[linear_to_offset(id, meta.input)]), meta.scalar));
}

// --- Copy/compact ---

kernel void copy_compact_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = input[linear_to_offset(id, meta)];
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

kernel void copy_compact_i64(
    device const long* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant StridedMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.size) return;
    output[id] = input[linear_to_offset(id, meta)];
}

// --- Binary ops ---

kernel void add_f16(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant BinaryMeta& meta [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.lhs.size) return;
    output[id] = lhs[linear_to_offset(id, meta.lhs)] + rhs[linear_to_offset(id, meta.rhs)];
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

kernel void sub_f16(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output [[buffer(2)]],
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

kernel void mul_f16(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output [[buffer(2)]],
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

kernel void div_f16(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output [[buffer(2)]],
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

kernel void pow_f16(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant BinaryMeta& meta [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.lhs.size) return;
    output[id] = half(pow(float(lhs[linear_to_offset(id, meta.lhs)]), float(rhs[linear_to_offset(id, meta.rhs)])));
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

kernel void eq_f16(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant BinaryMeta& meta [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.lhs.size) return;
    output[id] = lhs[linear_to_offset(id, meta.lhs)] == rhs[linear_to_offset(id, meta.rhs)] ? half(1.0h) : half(0.0h);
}

// --- Reductions (serial, one thread per output) ---

kernel void reduce_sum_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ReduceMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.outer_size) return;
    uint start = id * meta.reduce_size;
    float acc = 0.0f;
    for (uint i = 0; i < meta.reduce_size; i++) {
        acc += float(input[start + i]);
    }
    output[id] = half(acc);
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

kernel void reduce_max_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ReduceMeta& meta [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= meta.outer_size) return;
    uint start = id * meta.reduce_size;
    float acc = -INFINITY;
    for (uint i = 0; i < meta.reduce_size; i++) {
        acc = max(acc, float(input[start + i]));
    }
    output[id] = half(acc);
}

// --- Reductions (parallel threadgroup, for large reduce dimensions) ---

constant uint REDUCE_THREADS = 256;

kernel void reduce_sum_par_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ReduceMeta& meta [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tgid >= meta.outer_size) return;
    uint start = tgid * meta.reduce_size;
    float acc = 0.0f;
    for (uint i = tid; i < meta.reduce_size; i += REDUCE_THREADS) {
        acc += float(input[start + i]);
    }
    threadgroup float shared[REDUCE_THREADS];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[tgid] = half(shared[0]);
}

kernel void reduce_sum_par_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceMeta& meta [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tgid >= meta.outer_size) return;
    uint start = tgid * meta.reduce_size;
    float acc = 0.0f;
    for (uint i = tid; i < meta.reduce_size; i += REDUCE_THREADS) {
        acc += input[start + i];
    }
    threadgroup float shared[REDUCE_THREADS];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[tgid] = shared[0];
}

kernel void reduce_max_par_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceMeta& meta [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tgid >= meta.outer_size) return;
    uint start = tgid * meta.reduce_size;
    float acc = -INFINITY;
    for (uint i = tid; i < meta.reduce_size; i += REDUCE_THREADS) {
        acc = max(acc, input[start + i]);
    }
    threadgroup float shared[REDUCE_THREADS];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[tgid] = shared[0];
}

kernel void reduce_max_par_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ReduceMeta& meta [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tgid >= meta.outer_size) return;
    uint start = tgid * meta.reduce_size;
    float acc = -INFINITY;
    for (uint i = tid; i < meta.reduce_size; i += REDUCE_THREADS) {
        acc = max(acc, float(input[start + i]));
    }
    threadgroup float shared[REDUCE_THREADS];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[tgid] = half(shared[0]);
}

// --- Tiled matrix multiply ---

constant uint TILE = 16;

kernel void matmul_f16(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MatmulMeta& meta [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    if (batch_idx >= meta.batch) return;

    device const half* a = lhs + batch_idx * meta.m * meta.k;
    device const half* b = rhs + batch_idx * meta.k * meta.n;
    device half* c = output + batch_idx * meta.m * meta.n;

    threadgroup half tileA[TILE][TILE];
    threadgroup half tileB[TILE][TILE];

    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;
    float acc = 0.0f;

    uint num_tiles = (meta.k + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {
        uint ak = t * TILE + tid.x;
        uint bk = t * TILE + tid.y;
        tileA[tid.y][tid.x] = (row < meta.m && ak < meta.k) ? a[row * meta.k + ak] : half(0.0h);
        tileB[tid.y][tid.x] = (bk < meta.k && col < meta.n) ? b[bk * meta.n + col] : half(0.0h);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE; kk++) {
            acc += float(tileA[tid.y][kk]) * float(tileB[kk][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < meta.m && col < meta.n) {
        c[row * meta.n + col] = half(acc);
    }
}

kernel void matmul_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulMeta& meta [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    if (batch_idx >= meta.batch) return;

    // Offset pointers to the current batch
    device const float* a = lhs + batch_idx * meta.m * meta.k;
    device const float* b = rhs + batch_idx * meta.k * meta.n;
    device float* c = output + batch_idx * meta.m * meta.n;

    threadgroup float tileA[TILE][TILE];
    threadgroup float tileB[TILE][TILE];

    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;
    float acc = 0.0f;

    uint num_tiles = (meta.k + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {
        uint ak = t * TILE + tid.x;
        uint bk = t * TILE + tid.y;
        tileA[tid.y][tid.x] = (row < meta.m && ak < meta.k) ? a[row * meta.k + ak] : 0.0f;
        tileB[tid.y][tid.x] = (bk < meta.k && col < meta.n) ? b[bk * meta.n + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE; kk++) {
            acc += tileA[tid.y][kk] * tileB[kk][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < meta.m && col < meta.n) {
        c[row * meta.n + col] = acc;
    }
}

// --- Large-tile matrix multiply (32×32 output tile, 16×16 threads, 2×2 per thread) ---

constant uint TILE_BIG = 32;

kernel void matmul_big_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulMeta& meta [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    if (batch_idx >= meta.batch) return;

    device const float* a = lhs + batch_idx * meta.m * meta.k;
    device const float* b = rhs + batch_idx * meta.k * meta.n;
    device float* c = output + batch_idx * meta.m * meta.n;

    // Each thread computes a 2×2 block of output
    uint row0 = tgid.y * TILE_BIG + tid.y * 2;
    uint col0 = tgid.x * TILE_BIG + tid.x * 2;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    threadgroup float tileA[TILE_BIG][TILE_BIG];
    threadgroup float tileB[TILE_BIG][TILE_BIG];

    uint num_tiles = (meta.k + TILE_BIG - 1) / TILE_BIG;
    for (uint t = 0; t < num_tiles; t++) {
        // Cooperative load: each of 16×16 threads loads a 2×2 block into shared memory
        uint base_k = t * TILE_BIG;
        for (uint di = 0; di < 2; di++) {
            for (uint dj = 0; dj < 2; dj++) {
                uint lr = tid.y * 2 + di;
                uint lc = tid.x * 2 + dj;
                uint ar = tgid.y * TILE_BIG + lr;
                uint ak = base_k + lc;
                tileA[lr][lc] = (ar < meta.m && ak < meta.k) ? a[ar * meta.k + ak] : 0.0f;

                uint br = base_k + lr;
                uint bc = tgid.x * TILE_BIG + lc;
                tileB[lr][lc] = (br < meta.k && bc < meta.n) ? b[br * meta.n + bc] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE_BIG; kk++) {
            float a0 = tileA[tid.y * 2][kk];
            float a1 = tileA[tid.y * 2 + 1][kk];
            float b0 = tileB[kk][tid.x * 2];
            float b1 = tileB[kk][tid.x * 2 + 1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < meta.m && col0 < meta.n) c[row0 * meta.n + col0] = acc00;
    if (row0 < meta.m && col0 + 1 < meta.n) c[row0 * meta.n + col0 + 1] = acc01;
    if (row0 + 1 < meta.m && col0 < meta.n) c[(row0 + 1) * meta.n + col0] = acc10;
    if (row0 + 1 < meta.m && col0 + 1 < meta.n) c[(row0 + 1) * meta.n + col0 + 1] = acc11;
}

kernel void matmul_big_f16(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MatmulMeta& meta [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    if (batch_idx >= meta.batch) return;

    device const half* a = lhs + batch_idx * meta.m * meta.k;
    device const half* b = rhs + batch_idx * meta.k * meta.n;
    device half* c = output + batch_idx * meta.m * meta.n;

    uint row0 = tgid.y * TILE_BIG + tid.y * 2;
    uint col0 = tgid.x * TILE_BIG + tid.x * 2;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    threadgroup half tileA[TILE_BIG][TILE_BIG];
    threadgroup half tileB[TILE_BIG][TILE_BIG];

    uint num_tiles = (meta.k + TILE_BIG - 1) / TILE_BIG;
    for (uint t = 0; t < num_tiles; t++) {
        uint base_k = t * TILE_BIG;
        for (uint di = 0; di < 2; di++) {
            for (uint dj = 0; dj < 2; dj++) {
                uint lr = tid.y * 2 + di;
                uint lc = tid.x * 2 + dj;
                uint ar = tgid.y * TILE_BIG + lr;
                uint ak = base_k + lc;
                tileA[lr][lc] = (ar < meta.m && ak < meta.k) ? a[ar * meta.k + ak] : half(0.0h);

                uint br = base_k + lr;
                uint bc = tgid.x * TILE_BIG + lc;
                tileB[lr][lc] = (br < meta.k && bc < meta.n) ? b[br * meta.n + bc] : half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE_BIG; kk++) {
            float a0 = float(tileA[tid.y * 2][kk]);
            float a1 = float(tileA[tid.y * 2 + 1][kk]);
            float b0 = float(tileB[kk][tid.x * 2]);
            float b1 = float(tileB[kk][tid.x * 2 + 1]);
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < meta.m && col0 < meta.n) c[row0 * meta.n + col0] = half(acc00);
    if (row0 < meta.m && col0 + 1 < meta.n) c[row0 * meta.n + col0 + 1] = half(acc01);
    if (row0 + 1 < meta.m && col0 < meta.n) c[(row0 + 1) * meta.n + col0] = half(acc10);
    if (row0 + 1 < meta.m && col0 + 1 < meta.n) c[(row0 + 1) * meta.n + col0 + 1] = half(acc11);
}

// --- Gather/scatter ---

kernel void gather_f16(
    device const half* input [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant GatherScatterMeta& meta [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint right = gid.x;
    if (row >= meta.left_len * meta.dst_dim || right >= meta.right_len) return;

    uint left = row / meta.dst_dim;
    uint dst_i = row % meta.dst_dim;
    uint ids_offset = (left * meta.dst_dim + dst_i) * meta.right_len + right;
    uint src_i = uint(indices[ids_offset]);
    output[ids_offset] = input[(left * meta.src_dim + src_i) * meta.right_len + right];
}

kernel void gather_f32(
    device const float* input [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant GatherScatterMeta& meta [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint right = gid.x;
    if (row >= meta.left_len * meta.dst_dim || right >= meta.right_len) return;

    uint left = row / meta.dst_dim;
    uint dst_i = row % meta.dst_dim;
    uint ids_offset = (left * meta.dst_dim + dst_i) * meta.right_len + right;
    uint src_i = uint(indices[ids_offset]);
    output[ids_offset] = input[(left * meta.src_dim + src_i) * meta.right_len + right];
}

kernel void scatter_add_f16(
    device const half* input [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant GatherScatterMeta& meta [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint right = gid.x;
    if (row >= meta.left_len * meta.dst_dim || right >= meta.right_len) return;

    uint left = row / meta.dst_dim;
    uint dst_i = row % meta.dst_dim;
    half acc = half(0.0h);
    for (uint src_i = 0; src_i < meta.src_dim; src_i++) {
        uint src_offset = (left * meta.src_dim + src_i) * meta.right_len + right;
        if (uint(indices[src_offset]) == dst_i) {
            acc += input[src_offset];
        }
    }
    output[(left * meta.dst_dim + dst_i) * meta.right_len + right] = acc;
}

kernel void scatter_add_f32(
    device const float* input [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant GatherScatterMeta& meta [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint right = gid.x;
    if (row >= meta.left_len * meta.dst_dim || right >= meta.right_len) return;

    uint left = row / meta.dst_dim;
    uint dst_i = row % meta.dst_dim;
    float acc = 0.0f;
    for (uint src_i = 0; src_i < meta.src_dim; src_i++) {
        uint src_offset = (left * meta.src_dim + src_i) * meta.right_len + right;
        if (uint(indices[src_offset]) == dst_i) {
            acc += input[src_offset];
        }
    }
    output[(left * meta.dst_dim + dst_i) * meta.right_len + right] = acc;
}

kernel void index_select_f16(
    device const half* input [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant IndexSelectMeta& meta [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint right = gid.x;
    if (row >= meta.left_len * meta.index_len || right >= meta.right_len) return;
    uint left = row / meta.index_len;
    uint index_i = row % meta.index_len;
    uint src_i = uint(indices[index_i]);
    output[(left * meta.index_len + index_i) * meta.right_len + right] =
        input[(left * meta.src_dim + src_i) * meta.right_len + right];
}

kernel void index_select_f32(
    device const float* input [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant IndexSelectMeta& meta [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint right = gid.x;
    if (row >= meta.left_len * meta.index_len || right >= meta.right_len) return;
    uint left = row / meta.index_len;
    uint index_i = row % meta.index_len;
    uint src_i = uint(indices[index_i]);
    output[(left * meta.index_len + index_i) * meta.right_len + right] =
        input[(left * meta.src_dim + src_i) * meta.right_len + right];
}

kernel void index_add_f16(
    device const half* input [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant IndexAddMeta& meta [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint right = gid.x;
    if (row >= meta.left_len * meta.dst_dim || right >= meta.right_len) return;

    uint left = row / meta.dst_dim;
    uint dst_i = row % meta.dst_dim;
    half acc = half(0.0h);
    for (uint src_i = 0; src_i < meta.src_dim; src_i++) {
        if (uint(indices[src_i]) == dst_i) {
            acc += input[(left * meta.src_dim + src_i) * meta.right_len + right];
        }
    }
    output[(left * meta.dst_dim + dst_i) * meta.right_len + right] = acc;
}

kernel void index_add_f32(
    device const float* input [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant IndexAddMeta& meta [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint right = gid.x;
    if (row >= meta.left_len * meta.dst_dim || right >= meta.right_len) return;

    uint left = row / meta.dst_dim;
    uint dst_i = row % meta.dst_dim;
    float acc = 0.0f;
    for (uint src_i = 0; src_i < meta.src_dim; src_i++) {
        if (uint(indices[src_i]) == dst_i) {
            acc += input[(left * meta.src_dim + src_i) * meta.right_len + right];
        }
    }
    output[(left * meta.dst_dim + dst_i) * meta.right_len + right] = acc;
}
