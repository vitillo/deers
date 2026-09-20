//! GEMM parity microbenchmark: is deers matmul itself slower than candle's,
//! or does the head-to-head gap live in framework overhead?
//!
//! Times one representative prefill GEMM ([128,1024] @ [1024,3072]) and one
//! decode GEMM ([1,1024] @ [1024,1024]) on both sides, F32, warmed up.
//!
//! Run:
//!   cargo run --release --example qwen3_gemm_micro

use std::time::Instant;

use candle_core::{Device as CDevice, Tensor as CTensor};
use deers::{Device, Tensor, no_grad};

const REPS: usize = 20;

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn bench(name: &str, m: usize, k: usize, n: usize) {
    let lhs: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.125 - 0.75).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.25 - 0.5).collect();
    let a = Tensor::from_vec(lhs.clone(), vec![m, k], Device::Cpu);
    let b = Tensor::from_vec(rhs.clone(), vec![k, n], Device::Cpu);
    let ca = CTensor::from_vec(lhs, (m, k), &CDevice::Cpu).unwrap();
    let cb = CTensor::from_vec(rhs, (k, n), &CDevice::Cpu).unwrap();

    no_grad(|| a.matmul(&b));
    ca.matmul(&cb).unwrap();

    let mut deers_times = Vec::with_capacity(REPS);
    let mut candle_times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        no_grad(|| a.matmul(&b));
        deers_times.push(start.elapsed().as_secs_f64());
        let start = Instant::now();
        ca.matmul(&cb).unwrap();
        candle_times.push(start.elapsed().as_secs_f64());
    }
    let deers = median(&mut deers_times) * 1e3;
    let candle = median(&mut candle_times) * 1e3;
    println!("{name}: deers {deers:.3}ms, candle {candle:.3}ms, ratio {:.2}x", deers / candle);
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("run with --release");
        std::process::exit(2);
    }
    println!("machine threads: {}", std::thread::available_parallelism().unwrap().get());
    bench("prefill MLP-up [128,1024]@[1024,3072]", 128, 1024, 3072);
    bench("decode proj     [1,1024]@[1024,1024]  ", 1, 1024, 1024);
}
