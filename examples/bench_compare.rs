use std::time::Instant;

fn main() {
    let batch_size = 256;
    let iterations = 50;

    let x_data: Vec<f32> = (0..batch_size * 784).map(|i| (i as f32 % 256.0) / 255.0).collect();
    let w1_data: Vec<f32> = (0..784 * 128).map(|i| (i as f32 * 0.001).sin() * 0.05).collect();
    let b1_data: Vec<f32> = vec![0.0; 128];
    let w2_data: Vec<f32> = (0..128 * 10).map(|i| (i as f32 * 0.002).sin() * 0.05).collect();
    let b2_data: Vec<f32> = vec![0.0; 10];
    let targets_u32: Vec<u32> = (0..batch_size).map(|i| (i % 10) as u32).collect();

    bench_deers(
        "deers cpu",
        deers::Device::Cpu,
        batch_size,
        iterations,
        &x_data,
        &w1_data,
        &b1_data,
        &w2_data,
        &b2_data,
        &targets_u32,
    );

    bench_deers(
        "deers mps",
        deers::Device::Mps,
        batch_size,
        iterations,
        &x_data,
        &w1_data,
        &b1_data,
        &w2_data,
        &b2_data,
        &targets_u32,
    );

    bench_candle_cpu(
        batch_size,
        iterations,
        &x_data,
        &w1_data,
        &b1_data,
        &w2_data,
        &b2_data,
        &targets_u32,
    );

    bench_candle_metal(
        batch_size,
        iterations,
        &x_data,
        &w1_data,
        &b1_data,
        &w2_data,
        &b2_data,
        &targets_u32,
    );
}

fn bench_deers(
    name: &str,
    device: deers::Device,
    batch_size: usize,
    iterations: usize,
    x_data: &[f32],
    w1_data: &[f32],
    b1_data: &[f32],
    w2_data: &[f32],
    b2_data: &[f32],
    targets_u32: &[u32],
) {
    use deers::{Tensor, Var};

    let x = Tensor::from_vec(x_data.to_vec(), (batch_size, 784), device);
    let w1 = Var::new(Tensor::from_vec(w1_data.to_vec(), (784, 128), device));
    let b1 = Var::new(Tensor::from_vec(b1_data.to_vec(), (128,), device));
    let w2 = Var::new(Tensor::from_vec(w2_data.to_vec(), (128, 10), device));
    let b2 = Var::new(Tensor::from_vec(b2_data.to_vec(), (10,), device));
    let targets = Tensor::from_vec(targets_u32.to_vec(), (batch_size,), device);

    let h = (x.matmul(&w1) + &*b1).relu();
    let logits = h.matmul(&w2) + &*b2;
    let loss = deers::loss::cross_entropy(&logits, &targets);
    let _ = loss.backward();

    let t0 = Instant::now();
    for _ in 0..iterations {
        let h = (x.matmul(&w1) + &*b1).relu();
        let logits = h.matmul(&w2) + &*b2;
        let _loss = deers::loss::cross_entropy(&logits, &targets);
    }
    let fwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    let t0 = Instant::now();
    for _ in 0..iterations {
        let h = (x.matmul(&w1) + &*b1).relu();
        let logits = h.matmul(&w2) + &*b2;
        let loss = deers::loss::cross_entropy(&logits, &targets);
        let _ = loss.backward();
    }
    let fwd_bwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    println!("=== {name} ===");
    println!("  forward:          {fwd_us:>10.0} us");
    println!("  forward+backward: {fwd_bwd_us:>10.0} us");
    println!("  backward only:    {:>10.0} us", fwd_bwd_us - fwd_us);
}

fn bench_candle_cpu(
    batch_size: usize,
    iterations: usize,
    x_data: &[f32],
    w1_data: &[f32],
    b1_data: &[f32],
    w2_data: &[f32],
    b2_data: &[f32],
    targets_u32: &[u32],
) {
    use candle_core::{Device as CDevice, Tensor as CTensor, Var};

    let x = CTensor::from_vec(x_data.to_vec(), (batch_size, 784), &CDevice::Cpu).unwrap();
    let w1 =
        Var::from_tensor(&CTensor::from_vec(w1_data.to_vec(), (784, 128), &CDevice::Cpu).unwrap())
            .unwrap();
    let b1 = Var::from_tensor(&CTensor::from_vec(b1_data.to_vec(), (128,), &CDevice::Cpu).unwrap())
        .unwrap();
    let w2 =
        Var::from_tensor(&CTensor::from_vec(w2_data.to_vec(), (128, 10), &CDevice::Cpu).unwrap())
            .unwrap();
    let b2 = Var::from_tensor(&CTensor::from_vec(b2_data.to_vec(), (10,), &CDevice::Cpu).unwrap())
        .unwrap();
    let targets = CTensor::from_vec(targets_u32.to_vec(), batch_size, &CDevice::Cpu).unwrap();

    let h = x.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().relu().unwrap();
    let logits = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
    let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
    let _ = loss.backward();

    let t0 = Instant::now();
    for _ in 0..iterations {
        let h = x.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().relu().unwrap();
        let logits = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
        let _loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
    }
    let fwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    let t0 = Instant::now();
    for _ in 0..iterations {
        let h = x.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().relu().unwrap();
        let logits = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
        let _ = loss.backward();
    }
    let fwd_bwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    println!("\n=== candle cpu ===");
    println!("  forward:          {fwd_us:>10.0} us");
    println!("  forward+backward: {fwd_bwd_us:>10.0} us");
    println!("  backward only:    {:>10.0} us", fwd_bwd_us - fwd_us);
}

fn bench_candle_metal(
    batch_size: usize,
    iterations: usize,
    x_data: &[f32],
    w1_data: &[f32],
    b1_data: &[f32],
    w2_data: &[f32],
    b2_data: &[f32],
    targets_u32: &[u32],
) {
    use candle_core::{Device as CDevice, Tensor as CTensor, Var};

    let device = match CDevice::new_metal(0) {
        Ok(device) => device,
        Err(err) => {
            println!("\n=== candle metal ===");
            println!("  skipped: {err}");
            return;
        }
    };

    let x = CTensor::from_vec(x_data.to_vec(), (batch_size, 784), &device).unwrap();
    let w1 = Var::from_tensor(&CTensor::from_vec(w1_data.to_vec(), (784, 128), &device).unwrap())
        .unwrap();
    let b1 =
        Var::from_tensor(&CTensor::from_vec(b1_data.to_vec(), (128,), &device).unwrap()).unwrap();
    let w2 = Var::from_tensor(&CTensor::from_vec(w2_data.to_vec(), (128, 10), &device).unwrap())
        .unwrap();
    let b2 =
        Var::from_tensor(&CTensor::from_vec(b2_data.to_vec(), (10,), &device).unwrap()).unwrap();
    let targets = CTensor::from_vec(targets_u32.to_vec(), batch_size, &device).unwrap();

    let h = x.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().relu().unwrap();
    let logits = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
    let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
    let _ = loss.backward();

    let t0 = Instant::now();
    for _ in 0..iterations {
        let h = x.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().relu().unwrap();
        let logits = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
        let _loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
    }
    let fwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    let t0 = Instant::now();
    for _ in 0..iterations {
        let h = x.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().relu().unwrap();
        let logits = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
        let _ = loss.backward();
    }
    let fwd_bwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

    println!("\n=== candle metal ===");
    println!("  forward:          {fwd_us:>10.0} us");
    println!("  forward+backward: {fwd_bwd_us:>10.0} us");
    println!("  backward only:    {:>10.0} us", fwd_bwd_us - fwd_us);
}
