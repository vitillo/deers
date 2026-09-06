---
name: verify-deers
description: Verify the deers Rust tensor library via cargo build, cargo test, clippy, and example smoke runs. Use when changing tensor ops, autograd, nn modules, optimizers, or models.
---

# Verify deers

Deers is a Rust library, not a service. There is no server or UI. Verification means exercising the real library path with cargo.

## Launch

No server to keep alive. Build once from the repo root:

```sh
cargo build --locked
```

Ready signal is a successful build with exit code 0. CUDA variant only on Linux with a GPU toolchain:

```sh
cargo build --locked --features cuda
```

## Doctor

Read-only check before driving anything:

```sh
./.claude/skills/verify-deers/scripts/doctor.sh
```

It prints cargo and rustc versions and runs a locked build. If the build fails, fix that first. Do not write a proof against a broken base.

## Drive

Full suite:

```sh
./.claude/skills/verify-deers/scripts/drive.sh /tmp/deers-verify.log
cargo clippy --all-targets --all-features -- -D warnings
```

Targeted run for one area (tensor, nn, gpt):

```sh
./.claude/skills/verify-deers/scripts/drive.sh /tmp/deers-verify.log tensor
```

Known base limitation on Linux without a CUDA toolkit: `cargo test`
builds the `candle-core` dev-dependency with its `cuda` feature, whose
transitive `cudarc` build script shells out to `nvcc --version` and fails
when `nvcc` is absent. In that environment prove with `cargo build` plus
`cargo clippy --lib`, and say the full suite was not runnable.

Smoke an example without network or long training. On a machine with the
CUDA toolkit installed, prefer `cargo build --example mnist_train` or a
short `--help` style invocation over a full training run, since training
examples auto-download data. On Linux without `nvcc`, example targets also
fail to link through the `candle-core` cuda dev-dependency, so note that
and stay with the library build and clippy proof.

```sh
cargo build --locked --example mnist_train
```

See `features/` for per-feature entry points and proof states.

## Evidence

The standard proof for numerical behavior is parity with candle. Run the
same operation in deers and in candle on the same inputs, then assert close
numerically: 1e-4 for f32 on CPU, 2e-3 for accelerator backends, 1e-2 for
f16 (see `tests/utils.rs` and `tests/gpt.rs` for the reference helpers).
New ops and gradient paths need both a forward and a backward comparison
against candle before they count as proven.

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

- `scripts/doctor.sh` runs the read-only build check. No arguments.
- `scripts/drive.sh <out-log> [filter]` runs `cargo test --locked [filter]` and tees output to the log. Both scripts are executable.
