use core::{f32, f64};
mod utils;

use deers::{DType, Device, Tensor};
use utils::Approx;

#[test]
fn test_zeros() {
    let tensor = Tensor::zeros((2, 3), DType::F32, Device::Cpu);

    assert_eq!(tensor.layout().shape, (2, 3).into());
    assert_eq!(tensor.layout().strides, (3, 1).into());
    assert_eq!(tensor.to_vec::<f32>().unwrap(), vec![0.0f32; 6]);
}

#[test]
fn test_ones() {
    let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);

    assert_eq!(tensor.layout().shape, (2, 3).into());
    assert_eq!(tensor.layout().strides, (3, 1).into());
    assert_eq!(tensor.to_vec::<f32>().unwrap(), vec![1.0f32; 6]);
}

#[test]
fn test_mps_zeros() {
    let tensor = Tensor::zeros((2, 3), DType::F32, Device::Mps);

    assert_eq!(tensor.device(), Device::Mps);
    assert_eq!(tensor.layout().shape, (2, 3).into());
    assert_eq!(tensor.to_vec::<f32>().unwrap(), vec![0.0f32; 6]);
}

#[test]
fn test_mps_ewise_add() {
    let a = Tensor::ones((2, 3), DType::F32, Device::Mps);
    let b = Tensor::ones((2, 3), DType::F32, Device::Mps);

    let c = a + b;

    assert_eq!(c.device(), Device::Mps);
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![2.0f32; 6]);
}

#[test]
fn test_mps_matmul_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Mps).attach();
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), Device::Mps).attach();
    let c = a.matmul(&b);

    let grads = c.backward().unwrap();

    assert_eq!(grads.get(a.id()).unwrap().device(), Device::Mps);
    assert_eq!(grads.get(b.id()).unwrap().device(), Device::Mps);
}

#[test]
fn test_ones_like() {
    let tensor = Tensor::zeros((2, 3), DType::F32, Device::Cpu);
    let tensor = tensor.ones_like();

    assert_eq!(tensor.layout().shape, (2, 3).into());
    assert_eq!(tensor.layout().strides, (3, 1).into());
    assert_eq!(tensor.to_vec::<f32>().unwrap(), vec![1.0f32; 6]);
}

#[test]
fn test_zeros_like() {
    let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);
    let tensor = tensor.zeros_like();

    assert_eq!(tensor.layout().shape, (2, 3).into());
    assert_eq!(tensor.layout().strides, (3, 1).into());
    assert_eq!(tensor.to_vec::<f32>().unwrap(), vec![0.0f32; 6]);
}

#[test]
fn test_neg() {
    let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);

    let tensor = -tensor;

    assert_eq!(tensor.to_vec::<f32>().unwrap(), vec![-1.0f32; 6]);
}

#[test]
fn test_neg_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = -&a;

    let grads = b.backward().unwrap();

    assert_eq!(grads.get(a.id()).unwrap(), -a.ones_like());
}

#[test]
fn test_ewise_add() {
    let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);
    let b = Tensor::ones((2, 3), DType::F32, Device::Cpu);

    let c = a + b;

    assert_eq!(c.to_vec::<f32>().unwrap(), vec![2.0f32; 6]);
}

#[test]
fn test_ewise_add_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu).attach();
    let c = &a + b;

    let grads = c.backward().unwrap();

    assert_eq!(grads.get(a.id()).unwrap(), a.ones_like());
}

#[test]
fn test_ewise_sub() {
    let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);
    let b = Tensor::ones((2, 3), DType::F32, Device::Cpu);

    let c = a - b;

    assert_eq!(c.to_vec::<f32>().unwrap(), vec![0.0; 6]);
}

#[test]
fn test_ewise_mul() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

    let c = a * b;

    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]
    );
}

#[test]
fn test_ewise_mul_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu).attach();
    let c = &a * &b;

    let grads = c.backward().unwrap();

    assert_eq!(grads.get(a.id()).unwrap(), b);
    assert_eq!(grads.get(b.id()).unwrap(), a);
}

#[test]
fn test_ewise_div() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

    let c = a / b;

    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_ewise_div_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu).attach();
    let c = &a / &b;

    let grads = c.backward().unwrap();

    Vec::<_>::assert_approx_eq(
        grads.get(a.id()).unwrap().to_vec().unwrap(),
        vec![0.2500, 0.2000, 0.1667],
    );
    Vec::<_>::assert_approx_eq(
        grads.get(b.id()).unwrap().to_vec().unwrap(),
        vec![-0.0625, -0.0800, -0.0833],
    );
}

#[test]
fn test_ewise_powf() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

    let c = a.powf(b);

    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 27.0, 256.0, 3125.0, 46656.0]
    );
}

#[test]
fn test_ewise_powf_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu).attach();
    let c = a.powf(&b);

    let grads = c.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap(),
        Tensor::from_vec(vec![4.0, 80., 1458.], (3,), Device::Cpu)
    );
    Vec::<_>::assert_approx_eq(
        grads.get(b.id()).unwrap().to_vec().unwrap(),
        vec![0., 22.1807, 800.8884],
    );
}

#[test]
fn test_ewise_log() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

    let b = a.log();

    Vec::<_>::assert_approx_eq(
        b.to_vec::<f32>().unwrap(),
        vec![0.0, f32::consts::LN_2, 1.0986],
    );
}

#[test]
fn test_ewise_log_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = &a.log();

    let grads = b.backward().unwrap();

    Vec::<_>::assert_approx_eq(
        grads.get(a.id()).unwrap().to_vec().unwrap(),
        vec![1.0, 0.5, 1.0 / 3.0],
    );
}

#[test]
fn test_ewise_exp() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

    let b = a.exp();

    Vec::<_>::assert_approx_eq(
        b.to_vec::<f32>().unwrap(),
        vec![f32::consts::E, 7.3891, 20.0855],
    );
}

#[test]
fn test_ewise_exp_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = &a.exp();

    let grads = b.backward().unwrap();

    Vec::<_>::assert_approx_eq(
        grads.get(a.id()).unwrap().to_vec().unwrap(),
        vec![f64::consts::E, 7.3891, 20.0855],
    );
}

#[test]
fn test_scalar_add() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

    let b = a + 2.0;

    assert_eq!(b.to_vec::<f32>().unwrap(), vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_backward_scalar_add() {
    let a = Tensor::ones((3,), DType::F32, Device::Cpu).attach();
    let b = &a + 2.0;

    let grads = b.backward().unwrap();

    let expected = Tensor::ones((3,), DType::F32, Device::Cpu);
    assert_eq!(grads.get(a.id()).unwrap(), expected);
}

#[test]
fn test_scalar_sub() {
    let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);

    let b = a - 2.0;

    assert_eq!(b.to_vec::<f32>().unwrap(), vec![-1.0; 6]);
}

#[test]
fn test_scalar_mul() {
    let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);

    let b = a * 2.0;

    assert_eq!(b.to_vec::<f32>().unwrap(), vec![2.0; 6]);
}

#[test]
fn test_backward_scalar_mul() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = &a * 2.0;

    let grads = b.backward().unwrap();

    let expected = Tensor::from_vec(vec![2.0, 2.0, 2.0], (3,), Device::Cpu);
    assert_eq!(grads.get(a.id()).unwrap(), expected);
}

#[test]
fn test_scalar_powf() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

    let b = a.scalar_powf(2.0);

    assert_eq!(b.to_vec::<f32>().unwrap(), vec![1.0, 4.0, 9.0]);
}

#[test]
fn test_scalar_powf_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = a.scalar_powf(3.0);

    let grads = b.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![3.0, 12.0, 27.0]
    );
}

#[test]
fn test_permute() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

    let b = a.permute(vec![1, 0]);

    assert_eq!(
        b.to_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    )
}

#[test]
fn test_permute_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).attach();
    let b = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (2, 3), Device::Cpu).attach();
    let c = a * &b;
    let d = c.permute(vec![1, 0]);

    let grads = d.backward().unwrap();

    assert_eq!(
        grads.get(b.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )
}

#[test]
fn test_permute_backward_uses_inverse_permutation() {
    let a = Tensor::from_vec(
        (0..24).map(|v| v as f32).collect::<Vec<_>>(),
        (2, 3, 4),
        Device::Cpu,
    )
    .attach();
    let b = a.permute(vec![1, 2, 0]);
    let grad = Tensor::from_vec(
        (0..24).map(|v| v as f32).collect::<Vec<_>>(),
        (3, 4, 2),
        Device::Cpu,
    );
    let loss = (&b * &grad).sum(vec![0, 1, 2], false);

    let grads = loss.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 1.0, 3.0, 5.0, 7.0,
            9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0,
        ]
    );
}

#[test]
fn test_broadcast() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

    let b = a.broadcast((2, 3));

    assert_eq!(b.layout().shape, (2, 3).into());
    assert_eq!(
        b.to_vec::<f32>().unwrap(),
        vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn test_broadcast_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = a.broadcast((2, 3));

    let grads = b.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap(),
        Tensor::from_vec(vec![2.0f32, 2.0, 2.0], (3,), Device::Cpu)
    );
}

#[test]
fn test_sum() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

    let b = a.sum(vec![0], false);

    assert_eq!(b.layout().shape, (3,).into());
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_sum_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).attach();
    let b = a.sum(vec![], false);

    let grads = b.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap(),
        Tensor::ones((2, 3), DType::F32, Device::Cpu)
    );
}

#[test]
fn test_mps_sum_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Mps).attach();
    let b = a.sum(vec![], false);

    let grads = b.backward().unwrap();

    assert_eq!(grads.get(a.id()).unwrap().device(), Device::Mps);
    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_sum_keepdims() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

    let b = a.sum(vec![0], true);

    assert_eq!(b.layout().shape, (1, 3).into());
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_reshape() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 6), Device::Cpu);

    let b = a.reshape((2, 3));

    assert_eq!(b.layout().shape, (2, 3).into());
    assert_eq!(
        b.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_reshape_after_permute_materializes_logical_order() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = a.permute(vec![1, 0]);
    let c = b.reshape((2, 3));

    assert_eq!(c.layout().shape, (2, 3).into());
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn test_reshape_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 6), Device::Cpu).attach();
    let b = a.reshape((2, 3));

    let grads = b.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap(),
        Tensor::ones((1, 6), DType::F32, Device::Cpu)
    );
}

#[test]
fn test_transpose() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);
    let b = a.transpose((0, 1).into());

    assert_eq!(b.layout().shape, (2, 2).into());
    assert_eq!(b.layout().strides, (1, 2).into());
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![1.0f32, 3.0, 2.0, 4.0]);
}

#[test]
fn test_transpose_default() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);
    let b = a.transpose(None);

    assert_eq!(b.layout().shape, (2, 2).into());
    assert_eq!(b.layout().strides, (1, 2).into());
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![1.0f32, 3.0, 2.0, 4.0]);
}

#[test]
fn test_matmul() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);

    let c = a.matmul(&b);

    assert_eq!(c.layout().shape, (2, 2).into());
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![7.0, 10.0, 15.0, 22.0]);
}

#[test]
fn test_matmul_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).attach();
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).attach();
    let c = a.matmul(&b);

    let grads = c.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![3.0, 7.0, 3.0, 7.0]
    );
    assert_eq!(
        grads.get(b.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![4.0, 4.0, 6.0, 6.0]
    );
}

#[test]
fn test_mps_matmul_backward_values() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Mps).attach();
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Mps).attach();
    let c = a.matmul(&b);

    let grads = c.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![3.0, 7.0, 3.0, 7.0]
    );
    assert_eq!(
        grads.get(b.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![4.0, 4.0, 6.0, 6.0]
    );
}

#[test]
fn test_mps_matmul_batched_3d() {
    // Same test as test_matmul_batched_3d but on MPS
    let a = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
        Device::Mps,
    );
    let b = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 3, 2],
        Device::Mps,
    );

    let c = a.matmul(&b);
    assert_eq!(c.layout().shape, vec![2, 2, 2].into());
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![4.0, 2.0, 10.0, 5.0, 24.0, 24.0, 33.0, 33.0]
    );
}

#[test]
fn test_matmul_non_square() {
    // (2,3) @ (3,2) = (2,2)
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = Tensor::from_vec(
        vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
        (3, 2),
        Device::Cpu,
    );

    let c = a.matmul(&b);

    assert_eq!(c.layout().shape, (2, 2).into());
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matmul_non_square_backward() {
    // (2,3) @ (3,2) = (2,2) — would have caught the shape bug with square-only tests
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).attach();
    let b = Tensor::from_vec(
        vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
        (3, 2),
        Device::Cpu,
    )
    .attach();
    let c = a.matmul(&b);
    let d = c.sum(vec![0, 1], true);

    let grads = d.backward().unwrap();

    // dA = ones(2,2) @ B^T = [[7+8, 9+10, 11+12], [7+8, 9+10, 11+12]] = [[15, 19, 23], [15, 19, 23]]
    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![15.0, 19.0, 23.0, 15.0, 19.0, 23.0]
    );
    // dB = A^T @ ones(2,2) = [[1+4, 1+4], [2+5, 2+5], [3+6, 3+6]] = [[5, 5], [7, 7], [9, 9]]
    assert_eq!(
        grads.get(b.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
}

#[test]
fn test_matmul_batched_3d() {
    // (2, 2, 3) @ (2, 3, 2) = (2, 2, 2) — two independent matmuls
    let a = Tensor::from_vec(
        vec![
            // batch 0: [[1,2,3],[4,5,6]]
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 1: [[7,8,9],[10,11,12]]
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
        Device::Cpu,
    );
    let b = Tensor::from_vec(
        vec![
            // batch 0: [[1,0],[0,1],[1,0]]
            1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, // batch 1: [[1,1],[1,1],[1,1]]
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 3, 2],
        Device::Cpu,
    );

    let c = a.matmul(&b);
    assert_eq!(c.layout().shape, vec![2, 2, 2].into());
    // batch 0: [[1+0+3, 0+2+0],[4+0+6, 0+5+0]] = [[4,2],[10,5]]
    // batch 1: [[7+8+9, 7+8+9],[10+11+12, 10+11+12]] = [[24,24],[33,33]]
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![4.0, 2.0, 10.0, 5.0, 24.0, 24.0, 33.0, 33.0]
    );
}

#[test]
fn test_matmul_batched_3d_backward() {
    let a = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
        Device::Cpu,
    )
    .attach();
    let b = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 3, 2],
        Device::Cpu,
    )
    .attach();
    let c = a.matmul(&b);
    let loss = c.sum(vec![0, 1, 2], true);
    let grads = loss.backward().unwrap();

    // dA = ones(2,2,2) @ B^T
    // batch 0: ones(2,2) @ [[1,0,1],[0,1,0]] = [[1,1,1],[1,1,1]]
    // batch 1: ones(2,2) @ [[1,1,1],[1,1,1]] = [[2,2,2],[2,2,2]]
    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    );

    // dB = A^T @ ones(2,2,2)
    // batch 0: [[1,4],[2,5],[3,6]]^T... wait, A^T @ ones
    // A^T for batch 0 = [[1,4],[2,5],[3,6]], @ ones(2,2) = [[5,5],[7,7],[9,9]]
    // A^T for batch 1 = [[7,10],[8,11],[9,12]], @ ones(2,2) = [[17,17],[19,19],[21,21]]
    assert_eq!(
        grads.get(b.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 17.0, 17.0, 19.0, 19.0, 21.0, 21.0]
    );
}

#[test]
fn test_max() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);

    let b = a.max(vec![1], false);

    assert_eq!(b.layout().shape, (2,).into());
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![2.0, 4.0]);
}

#[test]
fn test_max_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 3.0, 2.0, 4.0], (2, 2), Device::Cpu).attach();
    let b = a.max(vec![1], false);
    let loss = b.sum(vec![0], false);

    let grads = loss.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![0.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn test_max_backward_routes_gradient_to_all_tied_maxima() {
    let a = Tensor::from_vec(vec![2.0f32, 2.0, 1.0, 3.0], (2, 2), Device::Cpu).attach();
    let b = a.max(vec![1], false);
    let loss = b.sum(vec![0], false);

    let grads = loss.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn test_logsumexp() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);

    let b = a.log_sum_exp(vec![1]);

    assert_eq!(b.layout().shape, (2,).into());
    Vec::<_>::assert_approx_eq(b.to_vec::<f32>().unwrap(), vec![2.3133, 4.3133]);
}

#[test]
fn test_logsumexp_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).attach();
    let b = a.log_sum_exp(vec![1]);

    let grads = b.backward().unwrap();

    Vec::<_>::assert_approx_eq(
        grads.get(a.id()).unwrap().to_vec().unwrap(),
        vec![0.2689f32, 0.7311, 0.2689, 0.7311],
    );
}

#[test]
fn test_mps_logsumexp_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Mps).attach();
    let b = a.log_sum_exp(vec![1]);

    let grads = b.backward().unwrap();

    Vec::<_>::assert_approx_eq(
        grads.get(a.id()).unwrap().to_vec().unwrap(),
        vec![0.2689f32, 0.7311, 0.2689, 0.7311],
    );
}

#[test]
fn test_compact() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);
    let c = t.permute(vec![1, 0]);

    let c = c.compact();

    assert_eq!(c.to_vec::<f32>().unwrap(), vec![1.0f32, 3.0, 2.0, 4.0]);
}

#[test]
fn test_compact_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).attach();
    let b = a.permute(vec![1, 0]);
    let c = b.compact();

    let grads = c.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_rand() {
    let t = Tensor::rand((100,), DType::F32, Device::Cpu);
    let data: Vec<f32> = t.to_vec().unwrap();

    assert_eq!(data.len(), 100);
    assert!(data.iter().all(|&v| (0.0..1.0).contains(&v)));
    // Check it's not all the same value (extremely unlikely with real randomness)
    assert!(data.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn test_randn() {
    let t = Tensor::randn((1000,), DType::F32, Device::Cpu);
    let data: Vec<f32> = t.to_vec().unwrap();

    assert_eq!(data.len(), 1000);
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let variance: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
    // Mean should be close to 0, variance close to 1
    assert!(mean.abs() < 0.2, "mean = {}", mean);
    assert!((variance - 1.0).abs() < 0.2, "variance = {}", variance);
}

#[test]
fn test_relu() {
    let a = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], (2, 3), Device::Cpu);
    let b = a.relu();

    assert_eq!(
        b.to_vec::<f32>().unwrap(),
        vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn test_relu_backward() {
    let a = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], (2, 3), Device::Cpu).attach();
    let b = a.relu();
    let c = b.sum(vec![1], true);
    let d = c.sum(vec![0], true);

    let grads = d.backward().unwrap();

    // Gradient is 1 where input > 0, 0 elsewhere
    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_mps_relu_backward() {
    let a = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], (2, 3), Device::Mps).attach();
    let b = a.relu();
    let c = b.sum(vec![1], true);
    let d = c.sum(vec![0], true);

    let grads = d.backward().unwrap();

    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_relu_backward_with_mul() {
    // Test that gradients flow correctly through relu when multiplied
    let a = Tensor::from_vec(vec![-1.0f32, 2.0, -3.0, 4.0], (2, 2), Device::Cpu).attach();
    let b = a.relu();
    let scale = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);
    let c = b * scale;
    let d = c.sum(vec![0, 1], true);

    let grads = d.backward().unwrap();

    // grad = scale * (input > 0): [0, 2, 0, 4]
    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![0.0, 2.0, 0.0, 4.0]
    );
}

#[test]
fn test_square() {
    let a = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], (2, 3), Device::Cpu);
    let b = a.square();
    assert_eq!(
        b.to_vec::<f32>().unwrap(),
        vec![4.0, 1.0, 0.0, 1.0, 4.0, 9.0]
    );
}

#[test]
fn test_square_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu).attach();
    let b = a.square();
    let loss = b.sum(vec![0], false);
    let grads = loss.backward().unwrap();
    // d(x²)/dx = 2x
    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0]
    );
}

#[test]
fn test_sigmoid() {
    let a = Tensor::from_vec(vec![0.0f32, 1.0, -1.0, 10.0], (4,), Device::Cpu);
    let b = a.sigmoid();
    let result: Vec<f32> = b.to_vec().unwrap();
    // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.7311, sigmoid(-1) ≈ 0.2689, sigmoid(10) ≈ 1.0
    let expected = vec![0.5, 0.7311, 0.2689, 0.99995];
    Vec::assert_approx_eq(&result, &expected);
}

#[test]
fn test_sigmoid_backward() {
    let a = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], (3,), Device::Cpu).attach();
    let b = a.sigmoid();
    let loss = b.sum(vec![0], false);
    let grads = loss.backward().unwrap();
    // d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
    let result: Vec<f32> = grads.get(a.id()).unwrap().to_vec().unwrap();
    let expected = vec![0.25, 0.1966, 0.1966];
    Vec::assert_approx_eq(&result, &expected);
}

#[test]
fn test_tanh() {
    let a = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], (3,), Device::Cpu);
    let b = a.tanh();
    let result: Vec<f32> = b.to_vec().unwrap();
    // tanh(0) = 0, tanh(1) ≈ 0.7616, tanh(-1) ≈ -0.7616
    let expected = vec![0.0, 0.7616, -0.7616];
    Vec::assert_approx_eq(&result, &expected);
}

#[test]
fn test_tanh_large_values_saturate() {
    let a = Tensor::from_vec(vec![1000.0f32, -1000.0], (2,), Device::Cpu);
    let result: Vec<f32> = a.tanh().to_vec().unwrap();
    let expected = vec![1.0, -1.0];
    Vec::assert_approx_eq(&result, &expected);
}

#[test]
fn test_tanh_backward() {
    let a = Tensor::from_vec(vec![0.0f32, 1.0], (2,), Device::Cpu).attach();
    let b = a.tanh();
    let loss = b.sum(vec![0], false);
    let grads = loss.backward().unwrap();
    // d(tanh)/dx = 1 - tanh²(x)
    let result: Vec<f32> = grads.get(a.id()).unwrap().to_vec().unwrap();
    let expected = vec![1.0, 0.4200];
    Vec::assert_approx_eq(&result, &expected);
}

#[test]
fn test_softmax() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], (2, 3), Device::Cpu);
    let b = a.softmax(1);
    let result: Vec<f32> = b.to_vec().unwrap();
    // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
    let expected = vec![0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652];
    Vec::assert_approx_eq(&result, &expected);
}

#[test]
fn test_softmax_sums_to_one() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = a.softmax(1);
    let sums = b.sum(vec![1], false);
    Vec::assert_approx_eq(sums.to_vec::<f32>().unwrap(), vec![1.0, 1.0]);
}

#[test]
fn test_mean() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = a.mean(vec![1], false);
    assert_eq!(b.layout().shape, (2,).into());
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![2.0, 5.0]);
}

#[test]
fn test_mean_keepdims() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = a.mean(vec![1], true);
    assert_eq!(b.layout().shape, (2, 1).into());
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![2.0, 5.0]);
}

#[test]
fn test_mean_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).attach();
    let b = a.mean(vec![1], false);
    let loss = b.sum(vec![0], false);
    let grads = loss.backward().unwrap();
    // d(mean)/dx = 1/n for each element
    let result: Vec<f32> = grads.get(a.id()).unwrap().to_vec().unwrap();
    let expected = vec![1.0 / 3.0; 6];
    Vec::assert_approx_eq(&result, &expected);
}

#[test]
fn test_cat_dim0() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), Device::Cpu);
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], (1, 3), Device::Cpu);
    let c = Tensor::cat(&[a, b], 0);
    assert_eq!(c.layout().shape, (2, 3).into());
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_cat_dim1() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), Device::Cpu);
    let c = Tensor::cat(&[a, b], 1);
    assert_eq!(c.layout().shape, (2, 4).into());
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    );
}

#[test]
fn test_cat_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), Device::Cpu).attach();
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], (1, 3), Device::Cpu).attach();
    let c = Tensor::cat(&[a.clone(), b.clone()], 0);
    let loss = c.sum(vec![0, 1], false);
    let grads = loss.backward().unwrap();
    assert_eq!(
        grads.get(a.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0]
    );
    assert_eq!(
        grads.get(b.id()).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0]
    );
}

#[test]
fn test_cat_mps() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), Device::Mps);
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], (1, 3), Device::Mps);
    let c = Tensor::cat(&[a, b], 0);
    assert_eq!(c.layout().shape, (2, 3).into());
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_log_softmax() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], (2, 3), Device::Cpu);
    let b = a.log_softmax(1);
    let result: Vec<f32> = b.to_vec().unwrap();

    // log_softmax(x) = x - log(sum(exp(x))) per row
    // For [1,2,3]: logsumexp = ln(e^1 + e^2 + e^3) ≈ 3.4076
    let lse = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    let expected = vec![
        1.0 - lse,
        2.0 - lse,
        3.0 - lse,
        1.0 - lse,
        2.0 - lse,
        3.0 - lse,
    ];

    Vec::<_>::assert_approx_eq(result, expected);
}

#[test]
fn test_mps_log_softmax() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], (2, 3), Device::Mps);
    let b = a.log_softmax(1);
    let result: Vec<f32> = b.to_vec().unwrap();

    let lse = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    let expected = vec![
        1.0 - lse,
        2.0 - lse,
        3.0 - lse,
        1.0 - lse,
        2.0 - lse,
        3.0 - lse,
    ];

    Vec::<_>::assert_approx_eq(result, expected);
}

#[test]
fn test_log_softmax_sums_to_one() {
    // exp(log_softmax(x)) should sum to 1 along the softmax axis
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 1.0, -1.0], (2, 3), Device::Cpu);
    let b = a.log_softmax(1).exp();
    let sums: Vec<f32> = b.sum(vec![1], false).to_vec().unwrap();

    Vec::<_>::assert_approx_eq(sums, vec![1.0, 1.0]);
}

#[test]
fn test_log_softmax_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).attach();
    let b = a.log_softmax(1);
    let c = b.sum(vec![0, 1], true);

    let grads = c.backward().unwrap();

    // d(sum(log_softmax))/dx_ij = 1 - n_cols * softmax(x_i)_j
    // For [1,2]: softmax = [0.2689, 0.7311], grad = [1-2*0.2689, 1-2*0.7311] = [0.4622, -0.4622]
    // For [3,4]: same softmax values
    let grad: Vec<f32> = grads.get(a.id()).unwrap().to_vec().unwrap();
    Vec::<_>::assert_approx_eq(grad, vec![0.4622, -0.4622, 0.4622, -0.4622]);
}

#[test]
fn test_mps_log_softmax_backward() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Mps).attach();
    let b = a.log_softmax(1);
    let c = b.sum(vec![0, 1], true);

    let grads = c.backward().unwrap();

    let grad: Vec<f32> = grads.get(a.id()).unwrap().to_vec().unwrap();
    Vec::<_>::assert_approx_eq(grad, vec![0.4622, -0.4622, 0.4622, -0.4622]);
}

#[test]
fn test_gather_forward() {
    let input = Tensor::from_vec(
        vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        (2, 3),
        Device::Cpu,
    );
    let indices = Tensor::from_vec(vec![1u32, 2], (2,), Device::Cpu);
    let out = input.gather(1, &indices);
    let vals: Vec<f32> = out.to_vec().unwrap();
    // out[0] = input[0, 1] = 20.0, out[1] = input[1, 2] = 60.0
    Vec::<_>::assert_approx_eq(vals, vec![20.0, 60.0]);
}

#[test]
fn test_gather_backward() {
    let input =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).attach();
    let indices = Tensor::from_vec(vec![0u32, 2], (2,), Device::Cpu);
    let out = input.gather(1, &indices);
    let loss = out.sum(vec![0, 1], true);
    let grads = loss.backward().unwrap();
    let grad: Vec<f32> = grads.get(input.id()).unwrap().to_vec().unwrap();
    // Gradient is 1 at gathered positions, 0 elsewhere
    Vec::<_>::assert_approx_eq(grad, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_mps_gather_backward() {
    let input =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Mps).attach();
    let indices = Tensor::from_vec(vec![0u32, 2], (2,), Device::Cpu);
    let out = input.gather(1, &indices);
    let loss = out.sum(vec![0, 1], true);
    let grads = loss.backward().unwrap();
    let grad: Vec<f32> = grads.get(input.id()).unwrap().to_vec().unwrap();

    Vec::<_>::assert_approx_eq(grad, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_nll_loss_forward() {
    let log_probs = Tensor::from_vec(
        vec![-0.9076f32, -1.2076, -2.4076, -0.4076, -1.9076, -1.5076],
        (2, 3),
        Device::Cpu,
    );
    let targets = Tensor::from_vec(vec![0u32, 2], (2,), Device::Cpu);
    let loss = deers::loss::nll_loss(&log_probs, &targets);

    // -mean(log_probs[0,0] + log_probs[1,2]) = -mean(-0.9076 + -1.5076) = 1.2076
    let loss_val: Vec<f32> = loss.to_vec().unwrap();
    assert!((loss_val[0] - 1.2076).abs() < 1e-4, "loss={}", loss_val[0]);
}

#[test]
fn test_nll_loss_backward() {
    let log_probs = Tensor::from_vec(
        vec![-0.9f32, -1.2, -2.4, -0.4, -1.9, -1.5],
        (2, 3),
        Device::Cpu,
    )
    .attach();
    let targets = Tensor::from_vec(vec![1u32, 0], (2,), Device::Cpu);
    let loss = deers::loss::nll_loss(&log_probs, &targets);
    let grads = loss.backward().unwrap();
    let grad: Vec<f32> = grads.get(log_probs.id()).unwrap().to_vec().unwrap();

    // Gradient is -1/batch_size at [i, target[i]], 0 elsewhere
    // batch_size=2, so scale = -0.5
    // targets = [1, 0] → grad[0,1]=-0.5, grad[1,0]=-0.5, rest 0
    Vec::<_>::assert_approx_eq(grad, vec![0.0, -0.5, 0.0, -0.5, 0.0, 0.0]);
}

#[test]
fn test_cross_entropy_end_to_end() {
    // Full pipeline: logits -> cross_entropy -> backward
    let logits =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.0, -1.0, 0.0], (2, 3), Device::Cpu).attach();
    let targets = Tensor::from_vec(vec![2u32, 0], (2,), Device::Cpu);
    let loss = deers::loss::cross_entropy(&logits, &targets);
    let grads = loss.backward().unwrap();

    let grad: Vec<f32> = grads.get(logits.id()).unwrap().to_vec().unwrap();
    assert_eq!(grad.len(), 6);

    // Sum of gradients per row should be 0
    let row1_sum: f32 = grad[0..3].iter().sum();
    let row2_sum: f32 = grad[3..6].iter().sum();
    assert!(row1_sum.abs() < 1e-4, "row1_sum={row1_sum}");
    assert!(row2_sum.abs() < 1e-4, "row2_sum={row2_sum}");
}

#[test]
fn test_mps_cross_entropy_end_to_end() {
    let logits =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.0, -1.0, 0.0], (2, 3), Device::Mps).attach();
    let targets = Tensor::from_vec(vec![2u32, 0], (2,), Device::Cpu);
    let loss = deers::loss::cross_entropy(&logits, &targets);
    let grads = loss.backward().unwrap();

    let grad: Vec<f32> = grads.get(logits.id()).unwrap().to_vec().unwrap();
    assert_eq!(grad.len(), 6);

    let row1_sum: f32 = grad[0..3].iter().sum();
    let row2_sum: f32 = grad[3..6].iter().sum();
    assert!(row1_sum.abs() < 1e-4, "row1_sum={row1_sum}");
    assert!(row2_sum.abs() < 1e-4, "row2_sum={row2_sum}");
}

#[test]
fn test_mps_narrow_to_vec() {
    let a = Tensor::from_vec(
        (0..32).map(|v| v as f32).collect::<Vec<_>>(),
        (8, 4),
        Device::Mps,
    );
    let b = a.narrow(0, 2, 3);

    assert_eq!(
        b.to_vec::<f32>().unwrap(),
        vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    );
}

// --- Var & SGD tests ---

use deers::optim::SGD;
use deers::Var;

#[test]
fn test_var_basic() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);
    let var = Var::new(t);
    assert!(var.requires_grad());
    assert_eq!(var.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_sgd_step() {
    // Minimize (x - 3)^2, starting at x=0. After one step with lr=0.1:
    // grad = 2*(0-3) = -6, x_new = 0 - 0.1*(-6) = 0.6
    let x = Var::new(Tensor::from_vec(vec![0.0f32], (1,), Device::Cpu));
    let target = Tensor::from_vec(vec![3.0f32], (1,), Device::Cpu);
    let mut sgd = SGD::new(vec![x.clone()], 0.1);

    let diff = &*x - &target;
    let loss = (&diff * &diff).sum(vec![0], true);
    sgd.backward_step(&loss).unwrap();

    let val: Vec<f32> = x.to_vec().unwrap();
    assert!((val[0] - 0.6).abs() < 1e-5, "x after step = {}", val[0]);
}

#[test]
fn test_sgd_loss_decreases() {
    // Minimize (x - 3)^2 starting from x=0. Loss should decrease each step.
    let x = Var::new(Tensor::from_vec(vec![0.0f32], (1,), Device::Cpu));
    let target = Tensor::from_vec(vec![3.0f32], (1,), Device::Cpu);
    let mut sgd = SGD::new(vec![x.clone()], 0.1);

    let mut prev_loss = f32::MAX;
    for _ in 0..5 {
        let diff = &*x - &target;
        let loss = (&diff * &diff).sum(vec![0], true);
        let loss_val: Vec<f32> = loss.to_vec().unwrap();
        assert!(loss_val[0] < prev_loss, "loss should decrease");
        prev_loss = loss_val[0];
        sgd.backward_step(&loss).unwrap();
    }
}

#[test]
fn test_sgd_preserves_grad_tracking() {
    let x = Var::new(Tensor::from_vec(vec![1.0f32], (1,), Device::Cpu));
    let mut sgd = SGD::new(vec![x.clone()], 0.01);

    let loss = (&*x * &*x).sum(vec![0], true);
    sgd.backward_step(&loss).unwrap();

    assert!(x.requires_grad());
}

#[test]
fn test_auto_broadcast_add() {
    // (2,3) + (3,) should auto-broadcast
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], (3,), Device::Cpu);
    let c = &a + &b;
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
}

#[test]
fn test_auto_broadcast_mul() {
    // (2,3) * (2,1) should auto-broadcast
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
    let b = Tensor::from_vec(vec![2.0f32, 3.0], (2, 1), Device::Cpu);
    let c = &a * &b;
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0, 12.0, 15.0, 18.0]
    );
}

#[test]
fn test_auto_broadcast_gradient() {
    // Verify gradients flow correctly through auto-broadcast
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).attach();
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], (3,), Device::Cpu).attach();
    let c = (&a + &b).sum(vec![0, 1], false);
    let grads = c.backward().unwrap();

    // d(sum(a+b))/da = ones(2,3)
    let ga = grads.get(a.id()).unwrap();
    assert_eq!(ga.to_vec::<f32>().unwrap(), vec![1.0; 6]);

    // d(sum(a+b))/db = sum along broadcast axis 0 of ones(2,3) = [2,2,2]
    let gb = grads.get(b.id()).unwrap();
    assert_eq!(gb.to_vec::<f32>().unwrap(), vec![2.0, 2.0, 2.0]);
}
