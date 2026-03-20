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

## Error handling

- Prefer explicit panics for unrecoverable programmer errors where there is no meaningful local recovery path.
- Invalid tensor program states should fail loudly rather than hiding the issue behind defensive abstractions.
- Example: mixing incompatible tensor dtypes in an operation is allowed to panic if the operation cannot sensibly continue.
- Use `Result` for operations that can realistically fail at runtime, such as backend execution or data movement.

## Structure

- Keep `Tensor` focused on general tensor operations.
- Keep `nn` focused on modules.
- Put stateless model-building helpers in `nn::functional`.

## Process

- Explain the reasoning behind design choices, not just the diff.
- When multiple valid approaches exist, present the options and tradeoffs.
- Don’t preserve awkward code just to preserve old tests; update tests when the design improves.
- Before adding a tensor/autograd fallback, check whether it forces CPU <-> accelerator copies on MPS. If the op already lives at storage level, prefer keeping the fallback there or explicitly call out the backend cost.

## Testing

- Structure tests in three explicit phases: `Arrange`, `Act`, and `Assert`.
- Keep test setup small and local so each phase is easy to scan.
- Prefer one clear behavior per test over broad scenario tests.

## Commits

- Commit messages should explain why the change exists, not restate the diff.
