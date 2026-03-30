# Contributing to DEERS

## Design principles

- Prefer the simplest approach that works.
- Keep the code easy to follow for someone learning how tensor libraries are built.
- Compose from existing primitives unless a lower-level implementation avoids a real backend cost.
- Be reasonably efficient on every backend. Avoid unnecessary CPU <-> accelerator copies or materialization roundtrips when work can stay on-device.
- Follow PyTorch and candle conventions for API shape and placement when they fit the project goals.
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

## Testing

- Structure tests in three explicit phases: `Arrange`, `Act`, and `Assert`.
- Keep test setup small and local so each phase is easy to scan.
- Prefer one clear behavior per test over broad scenario tests.
- Run `cargo clippy --all-targets --all-features -- -D warnings` before wrapping up a change.

## Commits

- Commit messages should explain why the change exists, not restate the diff.
