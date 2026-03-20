use std::time::Instant;
use std::{env, process};

use deers::dataset::MNISTDataset;
use deers::nn::{self, Module};
use deers::optim::SGD;
use deers::{Device, Tensor, loss};

fn main() {
    let device = parse_device_arg();

    println!("Loading MNIST...");
    let t0 = Instant::now();
    let dataset = MNISTDataset::load().unwrap();
    println!("Loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // Flatten images: (N, 28, 28) -> (N, 784)
    let train_images = dataset.train_images.reshape((60000, 784)).to_device(device).unwrap();
    let test_images = dataset.test_images.reshape((10000, 784)).to_device(device).unwrap();
    let train_labels = dataset.train_labels;
    let test_labels = dataset.test_labels;

    let model =
        nn::seq().add(nn::Linear::new(784, 128)).add(nn::ReLU).add(nn::Linear::new(128, 10));
    model.to_device(device).unwrap();

    println!("Device: {device:?}");
    println!("Model: 784 -> 128 (relu) -> 10");
    println!("Parameters: {}", model.parameters().len());

    let lr = 0.1;
    let mut sgd = SGD::new(model.parameters(), lr);
    let batch_size = 256;
    let num_batches = 60000 / batch_size;
    let epochs = 3;

    println!("Training: {epochs} epochs, batch_size={batch_size}, lr={lr}\n");

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let images = train_images.narrow(0, start, batch_size);
            let labels = train_labels.narrow(0, start, batch_size);

            let logits = model.forward(&images).unwrap();
            let batch_loss = loss::cross_entropy(&logits, &labels);
            let loss_val: Vec<f32> = batch_loss.to_vec().unwrap();
            epoch_loss += loss_val[0];
            sgd.backward_step(&batch_loss).unwrap();

            if batch_idx % 25 == 0 || batch_idx + 1 == num_batches {
                println!("batch {}/{num_batches} | loss: {:.4}", batch_idx + 1, loss_val[0]);
            }
        }

        let elapsed = epoch_start.elapsed().as_secs_f64();
        let avg_loss = epoch_loss / num_batches as f32;
        let accuracy = evaluate(&model, &test_images, &test_labels, batch_size);
        println!(
            "\r  epoch {}/{epochs} | avg_loss: {avg_loss:.4} | test_acc: {accuracy:.2}% | {elapsed:.1}s",
            epoch + 1
        );
    }

    println!("\nDone.");
}

fn parse_device_arg() -> Device {
    let mut args = env::args().skip(1);
    let mut device = Device::Cpu;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                let value = args.next().unwrap_or_else(|| usage("missing value for --device"));
                device = match value.as_str() {
                    "cpu" => Device::Cpu,
                    "mps" => Device::Mps,
                    other => usage(&format!("unsupported device: {other}")),
                };
            }
            "--help" | "-h" => usage(""),
            other => usage(&format!("unexpected argument: {other}")),
        }
    }

    device
}

fn usage(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("{message}");
        eprintln!();
    }
    eprintln!("Usage: cargo run --example mnist_train -- [--device cpu|mps]");
    process::exit(if message.is_empty() { 0 } else { 1 });
}

fn evaluate(model: &impl Module, images: &Tensor, labels: &Tensor, batch_size: usize) -> f64 {
    let num_batches = 10000 / batch_size;
    let mut correct = 0usize;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let batch_images = images.narrow(0, start, batch_size);
        let batch_labels: Vec<i64> = labels.narrow(0, start, batch_size).to_vec().unwrap();

        let logits = model.forward(&batch_images).unwrap();
        let logit_vals: Vec<f32> = logits.to_vec().unwrap();

        for i in 0..batch_size {
            let row = &logit_vals[i * 10..(i + 1) * 10];
            let pred =
                row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == batch_labels[i] as usize {
                correct += 1;
            }
        }
    }

    let total = num_batches * batch_size;
    correct as f64 / total as f64 * 100.0
}
