# Qwen3-0.6B on CPU: deers vs candle, measured

Same weights (`Qwen/Qwen3-0.6B` `model.safetensors`, BF16 on disk, run as F32
on both sides), same token ids, same greedy decode, release builds, warmed up.
Harness: `examples/qwen3_head_to_head.rs` (head-to-head plus deers op
profiles) and `examples/qwen3_gemm_micro.rs` (GEMM parity probe).

Machine: Intel Core Ultra X7 358H, 16 threads, 31 GiB RAM, Linux, CPU only.
Loader mirrors `tests/qwen3_candle_parity.rs` so both sides see identical
values; every prompt length passed a last-position top-1 agreement guard
before timing. Prefill is median of 3 reps, decode is 32 greedy tokens with a
full `to_vec` read per step on both sides (median per-token time of 2 reps).

## Head-to-head

| prompt | deers prefill | candle prefill | ratio | deers tok/s | candle tok/s | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.215 s | 0.153 s | 1.40x | 13.6 | 12.0 | 0.88x |
| 128 | 0.524 s | 0.354 s | 1.48x | 12.7 | 12.1 | 0.95x |
| 512 | 2.738 s | 1.818 s | 1.51x | 9.7 | 9.2 | 0.95x |

Per-token decode cost grows with context on both sides (deers 73.5 ms at 32
to 103.1 ms at 512; candle 83.3 ms to 108.7 ms), which is the expected linear
read of the KV cache plus the per-step full-vocab projection both sides pay.

## The gap is framework overhead, not math

One-shape GEMM probe, F32, median of 20 (`examples/qwen3_gemm_micro.rs`):

| shape | deers | candle | ratio |
| --- | --- | --- | --- |
| [128,1024] @ [1024,3072] (prefill MLP-up) | 1.179 ms | 1.181 ms | 1.00x |
| [1,1024] @ [1024,1024] (decode proj) | 0.093 ms | 0.095 ms | 0.98x |

Both sides lower CPU matmul onto the same `gemm` crate, and it shows: matmul
throughput is identical. The 1.4-1.5x prefill gap lives entirely in what the
framework does around the GEMMs.

## Where deers prefill time goes (128 tokens, profiled)

CPU total 475 ms across the profiled prefill, 1.21 GB allocated:

| slice | measured | share |
| --- | --- | --- |
| matmul (all 197 calls incl. vocab head) | ~270 ms | ~57% |
| `compact` copies after permute/reshape | 58.7 ms | ~12% |
| decomposed softmax (log_sum_exp + sub + exp + scale) | ~45 ms | ~10% |
| elementwise mul/add/sub/div (RoPE, QK-norm, residual) | ~50 ms | ~10% |

Decode at context ~130 runs ~79 ms/token on the wall clock: the per-step
`[1,1024] @ [1024,151936]` vocab projection costs 18.3 ms (23%) and is
inherent to greedy sampling, cache `cat` copies cost ~4 ms (5%) and grow
linearly with context, the rest is small decode GEMMs.

## Suspect verdicts, each with its measurement

- Repeated compaction copies after permutes: present, 58.7 ms of 475 ms
  (12.3%) in the 128-token prefill. `Reshape::new` compacts unconditionally
  (`src/ops.rs`), so every `rearrange` ending in a reshape copies when its
  input is a permute view. CPU matmul already accepts strides, so the copies
  on the matmul path buy nothing.
- Clone-heavy paths: `Tensor::clone` itself is an Arc bump and is innocent.
  The real clone-shaped cost is `KvCache::append`, which `cat`s the whole
  cache every token, every layer (`src/models/gpt.rs`): ~4 ms/token at
  context 130, scaling linearly per step (quadratic over a full generation).
- F32 funneling for half dtypes: not in the F32-vs-F32 gap by construction,
  but measured separately on the shipped BF16 path: the same 128-token deers
  prefill costs 0.963 s in BF16 vs 0.524 s in F32 (1.84x), because every CPU
  BF16 matmul upcasts both sides to F32 and downcasts back (`src/storage/cpu.rs`).
- Repeat-before-cache memory traffic: traffic-neutral, so not a win. The GQA
  expansion must materialize once per step either way; moving `repeat` after
  the cache read halves stored cache bytes but not per-step traffic. Take it
  only if long-context cache capacity (not speed) becomes the binding constraint.
- RoPE cache rebuilds: absent. Tables are precomputed once in `Qwen3::new`;
  decode only takes `narrow` views. Nothing to fix.
- Full score-matrix materialization: present in prefill (`[1,16,128,128]`
  scores, 28 layers), absent in decode (one `[1,16,1,T]` row). Chunking it
  needs streaming-attention machinery, which is out of scope; at 0.6B scale
  the GEMM and softmax rows show it costs what the math costs.
- Causal mask rebuilds: checked, negligible. The mask is a scalar fill of T
  elements (`src/nn/functional.rs`); no mask row rises above profiler noise.
  Sharing one mask across layers would save small allocs, not time.
- Sampler and logit reads: checked, negligible. `sample_last` compacts one
  151936-row (108 us) and the sampler walks O(V) once per token; both vanish
  next to the 18.3 ms vocab matmul that feeds them.

## Proposals (teachable only)

1. Hoist `compact` off the matmul path. Measured prize: up to 58.7 ms of
   475 ms (12%) in the 128-token prefill. Sketch: in `project_qkv`, `attend`,
   and `head_logits`, keep permute outputs as views into the stride-tolerant
   CPU matmul and compact once only where an elementwise op needs contiguity.
   Why it stays teachable: it moves existing calls after auditing each
   consumer, adding no type, pass, or operator.
2. Run inference weights in F16 instead of BF16 on CPU. Measured prize: the
   1.84x BF16 funneling tax (0.963 s vs 0.524 s at 128 tokens) drops to near
   zero, because F16 already has a native strided GEMM path while BF16 does
   not. Sketch: convert once at the load boundary, compute in F16, cast to
   F32 only at sampling. Why it stays teachable: one dtype choice at the
   boundary, no new ops; it still needs a parity proof since F16 rounds
   differently than BF16.

## Rejected with measurements

- Fused softmax (~45 ms, ~10% of prefill): needs a dedicated kernel, which
  the teachability constraint forbids. `log_sum_exp` is already fused; the
  remainder is the price of composing softmax from primitives.
- Paged or preallocated KV cache (cat ~4 ms/token at context 130, growing
  linearly per step): needs a slice-write operator deers does not have.
- Allocation traffic (1.21 GB allocated per 128-token prefill): needs an
  allocator or pooling design, i.e. architecture, not a small change.
- Post-cache GQA repeat: traffic-neutral per the analysis above, so it is a
  capacity option, not a speed win.

## Reproduce

```sh
QWEN3_06B_DIR=~/.cache/deers/qwen3-0.6B cargo run --release --example qwen3_head_to_head
cargo run --release --example qwen3_gemm_micro
```

Weights are read-only from the Hugging Face cache and are never committed.
