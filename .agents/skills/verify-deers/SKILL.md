---
name: verify-deers
description: Verify the deers Rust tensor library via cargo build, cargo test, clippy, and example smoke runs. Use when changing tensor ops, autograd, nn modules, optimizers, or models.
---

# Verify deers

Deers is a Rust library, not a service. There is no server or UI. Verification means exercising the real library path with cargo.

Verify against the hardware available. Doctor reports the tier. Drive only that tier. A missing accelerator is a limit of the host, not a failure of the change.

## Launch

No server to keep alive. Build once from the repo root.

```sh
cargo build --locked
```

Ready signal is a successful build with exit code 0. When doctor reports a GPU toolchain, build the CUDA variant.

```sh
cargo build --locked --features cuda
```

On macOS the default build already includes Metal.

## Doctor

Read-only check before driving anything.

```sh
./.agents/skills/verify-deers/scripts/doctor.sh
```

Doctor prints cargo and rustc versions, OS, GPU presence, and the expected tier. The tier is `cpu-only`, `cuda`, or `metal`. If the build fails, fix that first. Do not write a proof against a broken base.

## Drive

Pick the tier doctor reported. On `cuda` or `metal` run the full suite.

```sh
./.agents/skills/verify-deers/scripts/drive.sh /tmp/deers-verify.log
cargo clippy --all-targets --all-features -- -D warnings
```

Targeted run for one area (tensor, nn, gpt).

```sh
./.agents/skills/verify-deers/scripts/drive.sh /tmp/deers-verify.log tensor
```

On CPU-only Linux without `nvcc`, `cargo test` cannot pass. It builds the `candle-core` CUDA dev-dependency. The transitive `cudarc` build script shells out to `nvcc --version` and fails when `nvcc` is absent. The same breakage hits `cargo clippy --all-targets --all-features` and example targets. Prove with the CPU tier instead and say the full suite was not runnable on this hardware.

```sh
cargo build --locked
cargo clippy --lib --locked
```

Smoke an example only on matching hardware. Prefer a short help style invocation or a build of the example over a full training run. Training examples auto-download data. On CPU-only Linux skip the example build and stay with the library build and clippy proof.

```sh
cargo build --locked --example mnist_train
```

See `features/` for per-feature entry points and proof states.

## Evidence

The standard proof for numerical behavior is parity with candle. Run the same operation in deers and in candle on the same inputs, then assert close numerically. CPU proof is 1e-4 for f32. Accelerator numbers count only when doctor reported that hardware and the run used it. Those numbers are 2e-3 for accelerator backends and 1e-2 for f16 (see `tests/utils.rs` and `tests/gpt.rs` for the reference helpers). New ops and gradient paths need both a forward and a backward comparison against candle before they count as proven.

- Drive the real user path (`cargo test`, example binaries, public `Tensor` and `nn` APIs), not internal setters or test-only shortcuts.
- Record the command, the exit code, and the tail of the output, not just a final claim.
- Check side effects where relevant (checkpoint files written, loss decreasing in a short run).
- Mocks only where a production boundary already isolates the external system. Dataset auto-download hits the network, so prefer offline unit tests unless the feature under proof is the downloader itself.

## Cleanup

Kill what you started. There are no servers to stop. Cleanup means removing scratch state, never the evidence:

```sh
rm -f /tmp/deers-verify.log
```

Proof logs under `/tmp/` survive teardown. Do not delete the log named in the report during the same run. Dataset downloads under `data/` are verification scaffolding. Leave them or remove them explicitly, and say which.

## Helpers

- `scripts/doctor.sh` runs the read-only build check and prints the hardware tier. No arguments.
- `scripts/drive.sh <out-log> [filter]` runs `cargo test --locked [filter]` and tees output to the log. It refuses fast on CPU-only Linux without `nvcc` unless `DEERS_VERIFY_CPU_OK=1` is set. Both scripts are executable.
