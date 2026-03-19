# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar18`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `CLAUDE.md` — repository context and design principles.
   - `src/storage/mps.rs` — the MPS backend. **This is the main file you modify.**
   - `src/storage.rs` — the `BackendStorage` trait and dispatch.
   - `examples/bench_compare.rs` — the benchmark (forward pass of a 2-layer MNIST model). Do not modify.
4. **Read the reference implementation**: Read `/Users/roberto/projects/candle` for how candle implements its Metal backend. Key files:
   - `candle-metal-kernels/src/metal/commands.rs` — command buffer pooling and batching.
   - `candle-metal-kernels/src/kernels/mlx_gemm.rs` — tiled matmul dispatch with MLX steel kernels.
   - `candle-core/src/metal_backend/device.rs` — buffer pool with Arc-based reuse.
   - `candle-metal-kernels/src/kernel.rs` — pipeline caching and specialization constants.
   These are read-only references. Do not modify anything in the candle repo.
5. **Establish the baseline**: Build and run the benchmark as-is.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The goal

**Make the deers MPS backend faster**, closing the gap with candle's Metal backend. The benchmark is:

```bash
cargo run --release --example bench_compare
```

This runs a 2-layer MNIST forward pass (256 batch, matmul → relu → matmul → cross_entropy) 50 times each on: deers CPU, deers MPS, candle CPU, candle Metal. It reports forward-only and forward+backward times in microseconds.

**The metric is the deers MPS forward time.** Lower is better. You also want to watch the ratio vs candle Metal — that tells you how close you are to a mature implementation.

## What you CAN do

- Modify `src/storage/mps.rs` — this is the main file you edit. Metal kernels, buffer management, command buffer strategy, everything in here is fair game.
- Modify `src/storage.rs` if the trait needs small changes to support MPS optimizations (e.g. adding a method).
- Modify other `src/` files if needed to support your changes, but keep changes minimal.

## What you CANNOT do

- Modify `examples/bench_compare.rs`. The benchmark is the ground truth.
- Modify anything in `/Users/roberto/projects/candle`. It's read-only reference.
- Add new crate dependencies (you can only use what's already in `Cargo.toml`).
- Break existing tests. Run `cargo test` to verify after each change.

## Known bottlenecks and opportunities

Based on analysis of the current deers MPS code vs candle's Metal backend, here are the main areas to investigate (roughly ordered by expected impact):

### 1. Command buffer per-op overhead (HIGH impact)
Currently, every single operation (unary, binary, matmul, reduce) creates its own command buffer, commits it, and implicitly waits. This is the #1 bottleneck — GPU command submission has high fixed overhead.

**What candle does**: Pools command buffers (5 buffers, up to 50 encoders each). Multiple ops get batched into one command buffer. Only flushes when needed (threshold hit, or explicit sync).

**What to try**: Accumulate multiple compute encoders into a single command buffer. Only commit when results are needed (e.g. `into_cpu()` / `synchronize()`). Start simple — even a single shared command buffer that auto-commits on sync would be a big win.

### 2. Buffer allocation per-op (MEDIUM impact)
Every op allocates a new Metal buffer via `device.new_buffer()`. Buffer allocation has overhead, especially at small sizes.

**What candle does**: Arc-based buffer pool with power-of-two sizing. Reuses buffers when reference count drops to 1.

**What to try**: A simple buffer pool that caches freed buffers by size. Even a basic `Vec<Buffer>` per size bucket would help.

### 3. Naive matmul kernel (MEDIUM-HIGH impact)
The current `matmul_f32` kernel is a naive per-element loop with no tiling, no shared memory, no SIMD. For [256,784] × [784,128], this is leaving a lot of GPU performance on the table.

**What candle does**: Uses MLX's steel_gemm kernels with threadgroup memory, SIMD groups, and sophisticated tiling (32×32, 64×64 tiles depending on matrix size and device).

**What to try**: Start with simple threadgroup tiling (e.g. 16×16 tiles with shared memory). Don't try to match MLX's full complexity — even basic tiling should give 5-10x improvement over the naive kernel. You can also explore using `simdgroup_matrix` intrinsics if the matrix sizes support it.

### 4. Parameter buffer allocation (LOW-MEDIUM impact)
Every dispatch creates a new tiny Metal buffer for the metadata struct (StridedMeta, BinaryMeta, etc.) via `param_buffer()`. These are ~64-byte allocations that happen on every op.

**What to try**: Use `encoder.set_bytes()` instead of allocating a buffer for small metadata. Metal supports setting constant data directly without a buffer for data ≤ 4KB.

### 5. Synchronization on readback (LOW impact for forward-only)
`into_cpu()` calls `synchronize()` which creates an empty command buffer just to fence. If you batch commands (opportunity #1), the sync can just wait on the last real command buffer instead.

## Experimentation

Each experiment modifies code, rebuilds, and runs the benchmark. The benchmark itself takes a few seconds.

**What you CAN do:**
- Modify the MPS backend code (primarily `src/storage/mps.rs`).

**What you CANNOT do:**
- Modify the benchmark.
- Add dependencies.
- Break tests.

**The first run**: Your very first run should always be to establish the baseline, so run the benchmark as-is.

## Output format

The benchmark prints something like:

```
=== deers cpu ===
  forward:                1234 us
  forward+backward:       5678 us
  backward only:          4444 us

=== deers mps ===
  forward:                9876 us
  forward+backward:      19876 us
  backward only:         10000 us

=== candle cpu ===
  forward:                 800 us
  forward+backward:       3200 us
  backward only:          2400 us

=== candle metal ===
  forward:                 200 us
  forward+backward:        800 us
  backward only:           600 us
```

Extract the key numbers:

```bash
grep -E "forward:" bench.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	fwd_us	fwd_bwd_us	candle_metal_fwd_us	status	description
```

1. git commit hash (short, 7 chars)
2. deers MPS forward time in microseconds
3. deers MPS forward+backward time in microseconds
4. candle Metal forward time (for reference — should be stable across runs)
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	fwd_us	fwd_bwd_us	candle_metal_fwd_us	status	description
a1b2c3d	9876	19876	200	keep	baseline
b2c3d4e	3200	8400	198	keep	batch command buffers
c3d4e5f	3100	8200	201	keep	use set_bytes for params
d4e5f6g	0	0	0	crash	tiled matmul (shader compile error)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar18`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Pick an optimization from the opportunities list (or your own ideas based on profiling/reading candle).
3. Implement it by modifying the code.
4. Run `cargo test` to make sure nothing is broken. If tests fail, fix before proceeding.
5. git commit.
6. Run the benchmark: `cargo run --release --example bench_compare > bench.log 2>&1`
7. Read the results: `grep -E "deers mps|candle metal" -A3 bench.log`
8. If the grep output is empty or shows an error, the run crashed. Run `tail -n 30 bench.log` to read the error and attempt a fix.
9. Record the results in the tsv (NOTE: do not commit results.tsv, leave it untracked).
10. If the deers MPS forward time improved (lower), you "advance" the branch, keeping the commit.
11. If it's equal or worse, git reset back to where you started.

**Timeout**: Each benchmark run should take under 30 seconds. If it hangs, kill it and treat as failure.

**Crashes**: If a run crashes (shader compile error, wrong buffer size, etc.), use your judgment: easy fix → fix and re-run. Fundamentally broken → log as crash and move on.

**Tests**: Always run `cargo test` before benchmarking. A "faster" MPS backend that produces wrong results is worthless.

**Order of attack**: Start with the highest-impact, lowest-risk changes first. Suggested order:
1. Batch command buffers (biggest win for least code change)
2. Use `set_bytes` for param metadata
3. Buffer pool
4. Tiled matmul kernel

But use your judgment — if you see a quick win, take it.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, re-read candle's Metal backend for new patterns, try combining previous changes, or try more aggressive kernel optimizations. The loop runs until the human interrupts you, period.
