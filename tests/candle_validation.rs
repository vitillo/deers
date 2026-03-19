/// Cross-validate deers backward pass against candle.
/// For each operation, we run the same computation in both frameworks
/// with identical input data and compare the resulting gradients.
use candle_core::{Device as CDevice, Tensor as CTensor, Var};
use candle_nn::{self, Optimizer};
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
        assert!((d - c).abs() < TOL as f32, "{label}[{i}]: deers={d}, candle={c}");
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
    grads.get(var.as_tensor()).unwrap().flatten_all().unwrap().to_vec1().unwrap()
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
fn validate_scalar_powf_backward() {
    let data = vec![1.0f32, 2.0, 3.0];

    let da = Tensor::from_vec(data.clone(), (3,), Device::Cpu).attach();
    let db = da.scalar_powf(3.0);
    let dc = db.sum(vec![0], false);
    let dgrads = dc.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data, &[3]);
    let cb = ca.powf(3.0).unwrap();
    let cc = cb.sum_all().unwrap();
    let cgrads = cc.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "scalar_powf");
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
fn validate_max_backward() {
    let data = vec![1.0f32, 3.0, 2.0, 4.0];

    let da = Tensor::from_vec(data.clone(), (2, 2), Device::Cpu).attach();
    let db = da.max(vec![1], false);
    let dc = db.sum(vec![0], false);
    let dgrads = dc.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data, &[2, 2]);
    let cb = ca.max(1).unwrap();
    let cc = cb.sum_all().unwrap();
    let cgrads = cc.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "max");
}

#[test]
fn validate_permute_backward() {
    let data = (0..24).map(|v| v as f32).collect::<Vec<_>>();
    let grad = (0..24).map(|v| v as f32).collect::<Vec<_>>();

    let da = Tensor::from_vec(data.clone(), (2, 3, 4), Device::Cpu).attach();
    let dgrad = Tensor::from_vec(grad.clone(), (3, 4, 2), Device::Cpu);
    let db = da.permute(vec![1, 2, 0]);
    let dloss = (&db * &dgrad).sum(vec![0, 1, 2], false);
    let dgrads = dloss.backward().unwrap();
    let dgrad_a: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    let ca = cvar(data, &[2, 3, 4]);
    let cgrad_tensor = ctensor(grad, &[3, 4, 2]);
    let cb = ca.permute((1, 2, 0)).unwrap();
    let closs = cb.mul(&cgrad_tensor).unwrap().sum_all().unwrap();
    let cgrads = closs.backward().unwrap();

    assert_vecs_close(&dgrad_a, &cgrad(&cgrads, &ca), "permute");
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

#[test]
fn validate_log_softmax_backward() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, -1.0];

    // deers
    let da = Tensor::from_vec(data.clone(), (2, 3), Device::Cpu).attach();
    let db = da.log_softmax(1);
    let dc = db.sum(vec![0, 1], true);
    let dgrads = dc.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    // candle
    let ca = cvar(data, &[2, 3]);
    let cb = candle_nn::ops::log_softmax(&ca, 1).unwrap();
    let cc = cb.sum_all().unwrap();
    let cgrads = cc.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "log_softmax");
}

#[test]
fn validate_nll_loss_backward() {
    let logits = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, -1.0];
    let targets = vec![2i64, 0];

    // deers: logits -> log_softmax -> nll_loss -> backward
    let da = Tensor::from_vec(logits.clone(), (2, 3), Device::Cpu).attach();
    let db = da.log_softmax(1);
    let dloss = deers::loss::nll_loss(&db, &Tensor::from_vec(vec![2i64, 0], (2,), Device::Cpu));
    let dgrads = dloss.backward().unwrap();
    let dgrad: Vec<f32> = dgrads.get(da.id()).unwrap().to_vec().unwrap();

    // candle: logits -> log_softmax -> nll -> backward
    let ca = cvar(logits, &[2, 3]);
    let cb = candle_nn::ops::log_softmax(&ca, 1).unwrap();
    let ct = CTensor::from_vec(targets, 2, &CDevice::Cpu).unwrap();
    let closs = candle_nn::loss::nll(&cb, &ct).unwrap();
    let cgrads = closs.backward().unwrap();

    assert_vecs_close(&dgrad, &cgrad(&cgrads, &ca), "nll_loss");
}

#[test]
fn validate_sgd_matches_candle() {
    // Same MLP: x @ w -> relu -> sum as loss, one SGD step, compare updated weights.
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0]; // (2, 2)
    let w_data = vec![0.5f32, -0.3, 0.8, 0.1]; // (2, 2)
    let lr = 0.01;

    // --- deers ---
    let dx = deers::Tensor::from_vec(x_data.clone(), (2, 2), deers::Device::Cpu);
    let dw = deers::Var::new(deers::Tensor::from_vec(w_data.clone(), (2, 2), deers::Device::Cpu));
    let mut dsgd = deers::optim::SGD::new(vec![dw.clone()], lr);

    let dout = dx.matmul(&dw).relu();
    let dloss = dout.sum(vec![0, 1], true);
    dsgd.backward_step(&dloss).unwrap();
    let dw_updated: Vec<f32> = dw.to_vec().unwrap();

    // --- candle ---
    let cx = ctensor(x_data, &[2, 2]);
    let cw = cvar(w_data, &[2, 2]);
    let mut csgd = candle_nn::SGD::new(vec![cw.clone()], lr).unwrap();

    let cout = cx.matmul(&cw).unwrap().relu().unwrap();
    let closs = cout.sum_all().unwrap();
    csgd.backward_step(&closs).unwrap();
    let cw_updated: Vec<f32> = cw.flatten_all().unwrap().to_vec1().unwrap();

    assert_vecs_close(&dw_updated, &cw_updated, "sgd updated weights");
}

#[test]
fn validate_linear_cross_entropy_step() {
    // One full forward+backward through: Linear(4,3) -> relu -> Linear(3,2) -> cross_entropy
    // Compare every intermediate against candle.
    let x_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // (2, 4)
    let w1_data = vec![0.1f32, -0.2, 0.3, 0.4, 0.1, -0.1, -0.3, 0.2, 0.5, 0.2, -0.4, 0.1]; // (4, 3)
    let b1_data = vec![0.01f32, -0.02, 0.03]; // (3,)
    let w2_data = vec![0.2f32, -0.1, 0.3, 0.4, -0.2, 0.1]; // (3, 2)
    let b2_data = vec![0.05f32, -0.05]; // (2,)
    let targets_i64 = vec![1i64, 0];
    let targets_i64_deers = vec![1i64, 0];

    // --- deers ---
    let dx = Tensor::from_vec(x_data.clone(), (2, 4), Device::Cpu);
    let dw1 = deers::Var::new(Tensor::from_vec(w1_data.clone(), (4, 3), Device::Cpu));
    let db1 = deers::Var::new(Tensor::from_vec(b1_data.clone(), (3,), Device::Cpu));
    let dw2 = deers::Var::new(Tensor::from_vec(w2_data.clone(), (3, 2), Device::Cpu));
    let db2 = deers::Var::new(Tensor::from_vec(b2_data.clone(), (2,), Device::Cpu));

    // Forward: linear1
    let dh = dx.matmul(&dw1);
    let dh_vals: Vec<f32> = dh.to_vec().unwrap();

    let db1_bc = db1.reshape((1, 3)).broadcast((2, 3));
    let dh_bias = &dh + &db1_bc;
    let dh_bias_vals: Vec<f32> = dh_bias.to_vec().unwrap();

    // relu
    let dh_relu = dh_bias.relu();
    let dh_relu_vals: Vec<f32> = dh_relu.to_vec().unwrap();

    // Forward: linear2
    let dlogits = dh_relu.matmul(&dw2);
    let db2_bc = db2.reshape((1, 2)).broadcast((2, 2));
    let dlogits_bias = &dlogits + &db2_bc;
    let dlogits_vals: Vec<f32> = dlogits_bias.to_vec().unwrap();

    // cross_entropy
    let dtargets = Tensor::from_vec(targets_i64_deers.clone(), (2,), Device::Cpu);
    let dloss = deers::loss::cross_entropy(&dlogits_bias, &dtargets);
    let dloss_val: Vec<f32> = dloss.to_vec().unwrap();

    // backward
    let dgrads = dloss.backward().unwrap();
    let dgrad_w1: Vec<f32> = dgrads.get(dw1.id()).unwrap().to_vec().unwrap();
    let dgrad_b1: Vec<f32> = dgrads.get(db1.id()).unwrap().to_vec().unwrap();
    let dgrad_w2: Vec<f32> = dgrads.get(dw2.id()).unwrap().to_vec().unwrap();
    let dgrad_b2: Vec<f32> = dgrads.get(db2.id()).unwrap().to_vec().unwrap();

    // --- candle ---
    let cx = ctensor(x_data, &[2, 4]);
    let cw1 = cvar(w1_data, &[4, 3]);
    let cb1 = cvar(b1_data, &[3]);
    let cw2 = cvar(w2_data, &[3, 2]);
    let cb2 = cvar(b2_data, &[2]);

    // candle Linear stores weight as (out, in) and transposes, but we do (in, out) directly.
    // So we replicate our forward manually: x @ w + b
    let ch = cx.matmul(&cw1).unwrap();
    let ch_vals: Vec<f32> = ch.flatten_all().unwrap().to_vec1().unwrap();

    let ch_bias = ch.broadcast_add(&cb1).unwrap();
    let ch_bias_vals: Vec<f32> = ch_bias.flatten_all().unwrap().to_vec1().unwrap();

    let ch_relu = ch_bias.relu().unwrap();
    let ch_relu_vals: Vec<f32> = ch_relu.flatten_all().unwrap().to_vec1().unwrap();

    let clogits = ch_relu.matmul(&cw2).unwrap();
    let clogits_bias = clogits.broadcast_add(&cb2).unwrap();
    let clogits_vals: Vec<f32> = clogits_bias.flatten_all().unwrap().to_vec1().unwrap();

    let ct = CTensor::from_vec(targets_i64, 2, &CDevice::Cpu).unwrap();
    let closs = candle_nn::loss::cross_entropy(&clogits_bias, &ct).unwrap();
    let closs_val: Vec<f32> = closs.to_vec0::<f32>().map(|v| vec![v]).unwrap();

    let cgrads = closs.backward().unwrap();
    let cgrad_w1 = cgrad(&cgrads, &cw1);
    let cgrad_b1 = cgrad(&cgrads, &cb1);
    let cgrad_w2 = cgrad(&cgrads, &cw2);
    let cgrad_b2 = cgrad(&cgrads, &cb2);

    // --- compare stage by stage ---
    assert_vecs_close(&dh_vals, &ch_vals, "matmul1");
    assert_vecs_close(&dh_bias_vals, &ch_bias_vals, "linear1 (matmul+bias)");
    assert_vecs_close(&dh_relu_vals, &ch_relu_vals, "relu");
    assert_vecs_close(&dlogits_vals, &clogits_vals, "linear2 (matmul+bias)");
    assert_vecs_close(&dloss_val, &closs_val, "cross_entropy loss");

    assert_vecs_close(&dgrad_w2, &cgrad_w2, "grad_w2");
    assert_vecs_close(&dgrad_b2, &cgrad_b2, "grad_b2");
    assert_vecs_close(&dgrad_w1, &cgrad_w1, "grad_w1");
    assert_vecs_close(&dgrad_b1, &cgrad_b1, "grad_b1");
}
