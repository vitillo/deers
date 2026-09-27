---
name: nvidia-profile
description: Profile deers CUDA workloads with Nsight Systems timelines and Nsight Compute kernel counters. Use when optimizing GPU kernel time, memory movement, or launch overhead on CUDA hardware.
---

# Nvidia profile

Deers is a Rust library, not a service. Profiling means driving a real binary (example, benchmark, or test) under `nsys` or `ncu` on CUDA hardware. Tier one first: fix the timeline before touching kernels.

Run everything from the repo root on the machine holding the GPU. Never profile over a remote mount or a shared checkout.

## Use it when

Use this skill when GPU wall time, kernel selection, host-to-device copies, or launch overhead is the question. Skip it on `cpu-only` and `metal` tiers: without a CUDA device there is nothing to profile. Confirm the tier first.

```sh
./.agents/skills/verify-deers/scripts/doctor.sh
```

Proceed only on tier `cuda`. A missing GPU is a limit of the host, not a failure of the change.

## Prerequisites

Install both toolkits on the profiling host. One without the other leaves half the workflow missing.

```sh
sudo apt install nsight-systems nsight-compute
which nsys ncu
```

Check CPU sampling permission before collecting a timeline.

```sh
cat /proc/sys/kernel/perf_event_paranoid
```

Empty call stacks under `nsys` mean the paranoid level is too high. Lowering it needs root. If you cannot escalate, record host-side stacks as unavailable and keep the CUDA trace, which does not need it.

Check counter permission before collecting kernels. The driver restricts performance counters to admin by default. A first `ncu` run that reports insufficient permissions (`ERR_NVGPUCTRPERM`) proves the restriction is on. Clearing it is an admin-only change: run under `sudo` or set the driver option `NVreg_RestrictProfilingToAdminHost=0` and reboot the driver. Both need a privileged operator. Never work around it with sampled estimates presented as counter data. Report the block and stop.

## Tier one: timeline

Always start here. Build the release binary first: debug builds move the bottleneck.

```sh
cargo build --locked --release --features cuda
```

Capture one timeline of the exact workload under test.

```sh
nsys profile -o /tmp/deers-timeline --trace=cuda,nvtx,osrt --stats=true ./target/release/<binary> [args]
nsys stats /tmp/deers-timeline.nsys-rep
```

Read the report in this order: total CUDA time vs wall time, largest `memcpy`/`memset` blocks, kernel share, then gaps where the GPU idles while the host works. Fix copies and idle gaps before kernel internals. Idle GPU with a busy host is a launch or synchronization problem, not a kernel problem.

Keep the `.nsys-rep` file as evidence. Record the command, the exit code, and the top rows of the stats output.

## Tier two: kernels

Enter tier two only when the timeline names a kernel worth drilling into. Profile that kernel, not the whole run.

```sh
ncu --set detailed -o /tmp/deers-kernel ./target/release/<binary> [args]
```

Start from the `detailed` set: achieved occupancy, SOL memory vs SOL SM, and stall reasons. Chase the lower SOL number first. One metric set per run; stacking every counter in one pass serializes replay and skews timing.

## Compare

Every claim of faster needs a before and an after. Run identical work on both sides: same binary flags, same inputs, same batch and sequence lengths, same GPU clocks and power cap. Lock clocks where the operator allows; otherwise report them unlocked alongside the numbers.

Report the full environment with each comparison: GPU model (`nvidia-smi -L`), driver (`nvidia-smi`), CUDA toolkit (`nvcc --version`), `nsys`/`ncu` versions, clocks and power cap, and the exact profile commands. Numbers without this block do not count as a comparison.

## Gotchas

- NVTX ranges are absent unless the code emits them. Deers links CUDA dynamically through `cudarc` and emits no NVTX markers today, so the timeline shows raw kernels and API calls only. Name regions by binary and kernel name instead of inventing range labels. If markers get added later, they arrive via an NVTX shim crate; a host without the NVTX runtime silently drops them, so missing ranges are an environment gap, not proof of flat structure.
- No CUDA device means no profile. `ncu` and the CUDA trace in `nsys` fail or report empty on `cpu-only` and `metal` hosts. The verify-deers CPU tier applies: prove correctness there and mark GPU performance as not measured on this hardware.
- Profile the release binary with the same feature flags as the claim. A timeline from a debug build or a different `--features` set measures a different program.

## Cleanup

Remove scratch reports, never the evidence named in the report.

```sh
rm -f /tmp/deers-timeline.nsys-rep /tmp/deers-kernel.ncu-rep
```

Proof reports under `/tmp/` survive teardown. Do not delete a report file cited in the same run.
