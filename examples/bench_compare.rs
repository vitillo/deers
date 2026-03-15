use std::time::Instant;

fn main() {
    let batch_size = 256;
    let iterations = 50;

    // Shared data
    let x_data: Vec<f32> = (0..batch_size * 784).map(|i| (i as f32 % 256.0) / 255.0).collect();
    let w1_data: Vec<f32> = (0..784 * 128).map(|i| ((i as f32 * 0.001).sin() * 0.05)).collect();
    let b1_data: Vec<f32> = vec![0.0; 128];
    let w2_data: Vec<f32> = (0..128 * 10).map(|i| ((i as f32 * 0.002).sin() * 0.05)).collect();
    let b2_data: Vec<f32> = vec![0.0; 10];
    let targets_f32: Vec<f32> = (0..batch_size).map(|i| (i % 10) as f32).collect();
    let targets_u32: Vec<u32> = (0..batch_size).map(|i| (i % 10) as u32).collect();

    // --- Deers ---
    {
        use deers::{Device, Tensor, Var};

        let x = Tensor::from_vec(x_data.clone(), (batch_size, 784), Device::Cpu);
        let w1 = Var::new(Tensor::from_vec(w1_data.clone(), (784, 128), Device::Cpu));
        let b1 = Var::new(Tensor::from_vec(b1_data.clone(), (128,), Device::Cpu));
        let w2 = Var::new(Tensor::from_vec(w2_data.clone(), (128, 10), Device::Cpu));
        let b2 = Var::new(Tensor::from_vec(b2_data.clone(), (10,), Device::Cpu));
        let targets = Tensor::from_vec(targets_f32.clone(), (batch_size,), Device::Cpu);

        // Warmup
        let h = (x.matmul(&w1) + &*b1).relu();
        let logits = h.matmul(&w2) + &*b2;
        let loss = deers::loss::cross_entropy(&logits, &targets);
        let _ = loss.backward();

        // Forward benchmark
        let t0 = Instant::now();
        for _ in 0..iterations {
            let h = (x.matmul(&w1) + &*b1).relu();
            let logits = h.matmul(&w2) + &*b2;
            let _loss = deers::loss::cross_entropy(&logits, &targets);
        }
        let fwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

        // Forward + backward benchmark
        let t0 = Instant::now();
        for _ in 0..iterations {
            let h = (x.matmul(&w1) + &*b1).relu();
            let logits = h.matmul(&w2) + &*b2;
            let loss = deers::loss::cross_entropy(&logits, &targets);
            let _ = loss.backward();
        }
        let fwd_bwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

        println!("=== deers ===");
        println!("  forward:          {fwd_us:>10.0} us");
        println!("  forward+backward: {fwd_bwd_us:>10.0} us");
        println!("  backward only:    {:>10.0} us", fwd_bwd_us - fwd_us);
    }

    // --- Candle ---
    {
        use candle_core::{Device as CDevice, Tensor as CTensor, Var};
        use candle_nn::Optimizer;

        let x = CTensor::from_vec(x_data.clone(), (batch_size, 784), &CDevice::Cpu).unwrap();
        let w1 = Var::from_tensor(&CTensor::from_vec(w1_data.clone(), (784, 128), &CDevice::Cpu).unwrap()).unwrap();
        let b1 = Var::from_tensor(&CTensor::from_vec(b1_data.clone(), (128,), &CDevice::Cpu).unwrap()).unwrap();
        let w2 = Var::from_tensor(&CTensor::from_vec(w2_data.clone(), (128, 10), &CDevice::Cpu).unwrap()).unwrap();
        let b2 = Var::from_tensor(&CTensor::from_vec(b2_data.clone(), (10,), &CDevice::Cpu).unwrap()).unwrap();
        let targets = CTensor::from_vec(targets_u32.clone(), batch_size, &CDevice::Cpu).unwrap();

        // Warmup
        let h = x.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().relu().unwrap();
        let logits = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
        let _ = loss.backward();

        // Forward benchmark
        let t0 = Instant::now();
        for _ in 0..iterations {
            let h = x.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().relu().unwrap();
            let logits = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
            let _loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
        }
        let fwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

        // Forward + backward benchmark
        let t0 = Instant::now();
        for _ in 0..iterations {
            let h = x.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().relu().unwrap();
            let logits = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
            let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
            let _ = loss.backward();
        }
        let fwd_bwd_us = t0.elapsed().as_micros() as f64 / iterations as f64;

        println!("\n=== candle ===");
        println!("  forward:          {fwd_us:>10.0} us");
        println!("  forward+backward: {fwd_bwd_us:>10.0} us");
        println!("  backward only:    {:>10.0} us", fwd_bwd_us - fwd_us);
    }
}
