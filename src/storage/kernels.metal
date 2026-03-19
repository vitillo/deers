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
    uint n;
    uint p;
};

struct GatherMeta {
    uint rows;
    uint cols;
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

// --- Scalar ops ---

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

// --- Copy/compact ---

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

// --- Binary ops ---

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

// --- Reductions ---

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

// --- Tiled matrix multiply ---

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

// --- Gather/scatter ---

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
