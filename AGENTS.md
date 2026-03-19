# DEERS

## Purpose

DEERS is a minimal PyTorch clone in Rust for learning. Optimize for readability and understanding over production-level performance or abstraction.

## Design principles

- Prefer the simplest approach that works.
- Keep the code easy to follow for someone learning how tensor libraries are built.
- Compose from existing primitives unless a lower-level implementation avoids a real backend cost.
- Be reasonably efficient on every backend. Avoid unnecessary CPU <-> accelerator copies or materialization roundtrips when work can stay on-device.
- Follow PyTorch and candle conventions for API shape and placement when they fit the project goals.
- Work backwards from `nanochat`: only implement what the target model actually needs.
- Build incrementally: one small piece at a time.
- Prioritize correctness, with tests for new behavior and gradients where applicable.

## Structure

- Keep `Tensor` focused on general tensor operations.
- Keep `nn` focused on modules.
- Put stateless model-building helpers in `nn::functional`.

## Process

- Explain the reasoning behind design choices, not just the diff.
- When multiple valid approaches exist, present the options and tradeoffs.
- Don’t preserve awkward code just to preserve old tests; update tests when the design improves.

## Commits

- Commit messages should explain why the change exists, not restate the diff.
