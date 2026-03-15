/// Cross-validate deers backward pass against candle.
/// For each operation, we run the same computation in both frameworks
/// with identical input data and compare the resulting gradients.
use candle_core::{Device as CDevice, Tensor as CTensor, Var};
use deers::{Device, Tensor};

const TOL: f64 = 1e-4;

fn assert_vecs_close(deers: &[f32], candle: &[f32], label: &str) {
    assert_eq!(
        deers.len(),
        candle.len(),
        "{label}: length mismatch ({} vs {})",
        deers.len(),
        candle.len()
    );
    for (i, (d, c)) in deers.iter().zip(candle.iter()).enumerate() {
        assert!(
            (d - c).abs() < TOL as f32,
            "{label}[{i}]: deers={d}, candle={c}"
        );
    }
}

/// Helper: create a candle Var from data with given shape
fn cvar(data: Vec<f32>, shape: &[usize]) -> Var {
    let t = CTensor::from_vec(data, shape, &CDevice::Cpu).unwrap();
    Var::from_tensor(&t).unwrap()
}

/// Helper: create a non-trainable candle tensor
fn ctensor(data: Vec<f32>, shape: &[usize]) -> CTensor {
    CTensor::from_vec(data, shape, &CDevice::Cpu).unwrap()
}

/// Helper: extract gradient as flat Vec<f32>
fn cgrad(grads: &candle_core::backprop::GradStore, var: &Var) -> Vec<f32> {
    grads
        .get(var.as_tensor())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn validate_relu_backward() {
    let data = vec![-2.0f32, -1.0, 0.5, 1.0, -0.5, 3.0];

    // deers
    let da = Tensor::from_vec(data.clone(), (2, 3), Device::Cpu).attach();
    let db = da.relu();
    let dc = db.sum(vec![0, 1], true);
    let dgrads = dc.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    // candle
    let ca = cvar(data, &[2, 3]);
    let cb = ca.relu().unwrap();
    let cc = cb.sum_all().unwrap();
    let cgrads = cc.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "relu");
}

#[test]
fn validate_matmul_backward() {
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    // deers: (2,3) @ (3,2)
    let da = Tensor::from_vec(data_a.clone(), (2, 3), Device::Cpu).attach();
    let db = Tensor::from_vec(data_b.clone(), (3, 2), Device::Cpu).attach();
    let dc = da.matmul(&db);
    let dd = dc.sum(vec![0, 1], true);
    let dgrads = dd.backward().unwrap();
    let dgrad_a: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();
    let dgrad_b: Vec<f32> = dgrads.get(db.id()).unwrap().to_vec().unwrap();

    // candle
    let ca = cvar(data_a, &[2, 3]);
    let cb = cvar(data_b, &[3, 2]);
    let cc = ca.matmul(&cb).unwrap();
    let cd = cc.sum_all().unwrap();
    let cgrads = cd.backward().unwrap();

    assert_vecs_close(&dgrad_a, &cgrad(&cgrads, &ca), "matmul grad_a");
    assert_vecs_close(&dgrad_b, &cgrad(&cgrads, &cb), "matmul grad_b");
}

#[test]
fn validate_exp_backward() {
    let data = vec![0.5f32, 1.0, -0.5, 2.0];

    let da = Tensor::from_vec(data.clone(), (2, 2), Device::Cpu).attach();
    let db = da.exp();
    let dc = db.sum(vec![0, 1], true);
    let dgrads = dc.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data, &[2, 2]);
    let cb = ca.exp().unwrap();
    let cc = cb.sum_all().unwrap();
    let cgrads = cc.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "exp");
}

#[test]
fn validate_log_backward() {
    let data = vec![0.5f32, 1.0, 2.0, 3.0];

    let da = Tensor::from_vec(data.clone(), (2, 2), Device::Cpu).attach();
    let db = da.log();
    let dc = db.sum(vec![0, 1], true);
    let dgrads = dc.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data, &[2, 2]);
    let cb = ca.log().unwrap();
    let cc = cb.sum_all().unwrap();
    let cgrads = cc.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "log");
}

#[test]
fn validate_neg_backward() {
    let data = vec![1.0f32, -2.0, 3.0, -4.0];

    let da = Tensor::from_vec(data.clone(), (2, 2), Device::Cpu).attach();
    let db = -&da;
    let dc = db.sum(vec![0, 1], true);
    let dgrads = dc.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data, &[2, 2]);
    let cb = ca.neg().unwrap();
    let cc = cb.sum_all().unwrap();
    let cgrads = cc.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "neg");
}

#[test]
fn validate_mul_backward() {
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_b = vec![5.0f32, 6.0, 7.0, 8.0];

    let da = Tensor::from_vec(data_a.clone(), (2, 2), Device::Cpu).attach();
    let db = Tensor::from_vec(data_b.clone(), (2, 2), Device::Cpu).attach();
    let dc = &da * &db;
    let dd = dc.sum(vec![0, 1], true);
    let dgrads = dd.backward().unwrap();
    let dgrad_a: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();
    let dgrad_b: Vec<f32> = dgrads.get(db.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data_a, &[2, 2]);
    let cb = cvar(data_b, &[2, 2]);
    let cc = (&*ca * &*cb).unwrap();
    let cd = cc.sum_all().unwrap();
    let cgrads = cd.backward().unwrap();

    assert_vecs_close(&dgrad_a, &cgrad(&cgrads, &ca), "mul grad_a");
    assert_vecs_close(&dgrad_b, &cgrad(&cgrads, &cb), "mul grad_b");
}

#[test]
fn validate_div_backward() {
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_b = vec![5.0f32, 6.0, 7.0, 8.0];

    let da = Tensor::from_vec(data_a.clone(), (2, 2), Device::Cpu).attach();
    let db = Tensor::from_vec(data_b.clone(), (2, 2), Device::Cpu).attach();
    let dc = &da / &db;
    let dd = dc.sum(vec![0, 1], true);
    let dgrads = dd.backward().unwrap();
    let dgrad_a: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();
    let dgrad_b: Vec<f32> = dgrads.get(db.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data_a, &[2, 2]);
    let cb = cvar(data_b, &[2, 2]);
    let cc = (&*ca / &*cb).unwrap();
    let cd = cc.sum_all().unwrap();
    let cgrads = cd.backward().unwrap();

    assert_vecs_close(&dgrad_a, &cgrad(&cgrads, &ca), "div grad_a");
    assert_vecs_close(&dgrad_b, &cgrad(&cgrads, &cb), "div grad_b");
}

#[test]
fn validate_sum_backward() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Sum along axis 1
    let da = Tensor::from_vec(data.clone(), (2, 3), Device::Cpu).attach();
    let db = da.sum(vec![1], true);
    let dc = db.sum(vec![0], true);
    let dgrads = dc.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data, &[2, 3]);
    let cb = ca.sum(1).unwrap();
    let cc = cb.sum_all().unwrap();
    let cgrads = cc.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "sum");
}

#[test]
fn validate_broadcast_backward() {
    let data = vec![1.0f32, 2.0, 3.0];

    // (1,3) broadcast to (2,3), then sum
    let da = Tensor::from_vec(data.clone(), (1, 3), Device::Cpu).attach();
    let db = da.broadcast((2, 3));
    let dc = db.sum(vec![0, 1], true);
    let dgrads = dc.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data, &[1, 3]);
    let cb = ca.broadcast_as(&[2, 3]).unwrap();
    let cc = cb.sum_all().unwrap();
    let cgrads = cc.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "broadcast");
}

#[test]
fn validate_mlp_forward_backward() {
    // Simulate a small MLP: x @ w1 -> relu -> @ w2 -> sum
    // This validates the gradient flow through a realistic computation graph.
    // Use values where x @ w1 has no exact zeros (to avoid ReLU subgradient convention differences)
    let x_data = vec![1.1f32, 0.7, -0.9, 2.1, 0.3, 1.4]; // (2, 3)
    let w1_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, -0.1, 0.2, 0.3, -0.1, 0.6, -0.4, 0.2]; // (3, 4)
    let w2_data = vec![0.3f32, -0.2, 0.5, 0.1]; // (4, 1)

    // deers
    let dx = Tensor::from_vec(x_data.clone(), (2, 3), Device::Cpu);
    let dw1 = Tensor::from_vec(w1_data.clone(), (3, 4), Device::Cpu).attach();
    let dw2 = Tensor::from_vec(w2_data.clone(), (4, 1), Device::Cpu).attach();
    let dh = dx.matmul(&dw1).relu();
    let dout = dh.matmul(&dw2);
    let dloss = dout.sum(vec![0, 1], true);
    let dgrads = dloss.backward().unwrap();
    let dgrad_w1: Vec<f32> = dgrads.get(dw1.id()).unwrap().to_vec().unwrap();
    let dgrad_w2: Vec<f32> = dgrads.get(dw2.id()).unwrap().to_vec().unwrap();

    // candle
    let cx = ctensor(x_data, &[2, 3]);
    let cw1 = cvar(w1_data, &[3, 4]);
    let cw2 = cvar(w2_data, &[4, 1]);
    let ch = cx.matmul(&cw1).unwrap().relu().unwrap();
    let cout = ch.matmul(&cw2).unwrap();
    let closs = cout.sum_all().unwrap();
    let cgrads = closs.backward().unwrap();

    assert_vecs_close(&dgrad_w1, &cgrad(&cgrads, &cw1), "mlp grad_w1");
    assert_vecs_close(&dgrad_w2, &cgrad(&cgrads, &cw2), "mlp grad_w2");
}
