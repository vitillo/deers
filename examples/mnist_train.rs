use std::time::Instant;

use deers::dataset::MNISTDataset;
use deers::nn::{self, Module};
use deers::optim::SGD;
use deers::{loss, Device, Tensor};

fn main() {
    println!("Loading MNIST...");
    let t0 = Instant::now();
    let dataset = MNISTDataset::load().unwrap();
    println!("Loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // Flatten images: (N, 28, 28) -> (N, 784)
    let train_images = dataset.train_images.reshape((60000, 784));
    let test_images = dataset.test_images.reshape((10000, 784));

    let model = nn::seq()
        .add(nn::Linear::new(784, 128))
        .add(nn::ReLU)
        .add(nn::Linear::new(128, 10));

    println!("Model: 784 -> 128 (relu) -> 10");
    println!("Parameters: {} vars", model.vars().len());

    let lr = 0.1;
    let mut sgd = SGD::new(model.vars(), lr);
    let batch_size = 256;
    let num_batches = 60000 / batch_size;
    let epochs = 3;

    println!("Training: {epochs} epochs, batch_size={batch_size}, lr={lr}\n");

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let images = slice_rows(&train_images, start, batch_size);
            let labels = slice_rows(&dataset.train_labels, start, batch_size);

            let logits = model.forward(&images).unwrap();
            let batch_loss = loss::cross_entropy(&logits, &labels);

            let loss_val: Vec<f32> = batch_loss.to_vec().unwrap();
            epoch_loss += loss_val[0];

            sgd.backward_step(&batch_loss).unwrap();

            println!(
                "batch {}/{num_batches} | loss: {:.4}",
                batch_idx + 1,
                loss_val[0]
            );
        }

        let elapsed = epoch_start.elapsed().as_secs_f64();
        let avg_loss = epoch_loss / num_batches as f32;
        let accuracy = evaluate(&model, &test_images, &dataset.test_labels, batch_size);
        println!(
            "\r  epoch {}/{epochs} | avg_loss: {avg_loss:.4} | test_acc: {accuracy:.2}% | {elapsed:.1}s",
            epoch + 1
        );
    }

    println!("\nDone.");
}

fn evaluate(model: &impl Module, images: &Tensor, labels: &Tensor, batch_size: usize) -> f64 {
    let num_batches = 10000 / batch_size;
    let mut correct = 0usize;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let batch_images = slice_rows(images, start, batch_size);
        let batch_labels: Vec<f32> = slice_rows(labels, start, batch_size).to_vec().unwrap();

        let logits = model.forward(&batch_images).unwrap();
        let logit_vals: Vec<f32> = logits.to_vec().unwrap();

        for i in 0..batch_size {
            let row = &logit_vals[i * 10..(i + 1) * 10];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if pred == batch_labels[i] as usize {
                correct += 1;
            }
        }
    }

    let total = num_batches * batch_size;
    correct as f64 / total as f64 * 100.0
}

/// Extracts `count` rows starting at `start` along dimension 0.
fn slice_rows(t: &Tensor, start: usize, count: usize) -> Tensor {
    let shape = t.layout().shape();
    let row_size: usize = (1..shape.ndim()).map(|i| shape[i]).product();
    let data: Vec<f32> = t.to_vec().unwrap();
    let slice = data[start * row_size..(start + count) * row_size].to_vec();

    let mut new_shape: Vec<usize> = (0..shape.ndim()).map(|i| shape[i]).collect();
    new_shape[0] = count;
    Tensor::from_vec(slice, new_shape, Device::Cpu)
}
