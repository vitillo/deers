use deers::{DType, Device, NoGradGuard, Tensor, no_grad};

#[test]
fn no_grad_is_available_from_the_crate_root() {
    // Arrange
    let input = Tensor::ones((2,), DType::F32, Device::Cpu).attach();

    // Act
    let output = no_grad(|| &input + &input);

    // Assert
    assert!(!output.requires_grad());
    assert!(output.op().is_none());
}

#[test]
fn no_grad_restores_tracking_after_unwinding() {
    // Arrange
    let input = Tensor::ones((2,), DType::F32, Device::Cpu).attach();

    // Act
    let result = std::panic::catch_unwind(|| {
        no_grad(|| panic!("leave no-grad scope"));
    });
    let output = &input + &input;

    // Assert
    assert!(result.is_err());
    assert!(output.requires_grad());
    assert!(output.op().is_some());
}

#[test]
fn nested_no_grad_scopes_restore_at_the_outermost_scope() {
    // Arrange
    let input = Tensor::ones((2,), DType::F32, Device::Cpu).attach();

    // Act
    let outer_output = no_grad(|| {
        let inner_output = no_grad(|| &input + &input);
        let after_inner = &input * 2.0;
        (inner_output, after_inner)
    });
    let after_outer = &input + &input;

    // Assert
    assert!(!outer_output.0.requires_grad());
    assert!(!outer_output.1.requires_grad());
    assert!(after_outer.requires_grad());
}

#[test]
fn no_grad_guard_disables_tracking_until_drop() {
    // Arrange
    let input = Tensor::ones((2,), DType::F32, Device::Cpu).attach();

    // Act
    let scoped_output = {
        let _guard = NoGradGuard::new();
        &input + &input
    };
    let restored_output = &input + &input;

    // Assert
    assert!(!scoped_output.requires_grad());
    assert!(restored_output.requires_grad());
}
