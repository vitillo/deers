use core::{f32, f64};
use deers::{Approx, DType, Device, Tensor};

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
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu);
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
