#![cfg(test)]

pub trait Approx {
    fn approx_eq(&self, other: impl AsRef<Self>) -> bool;
}

impl Approx for Vec<f32> {
    fn approx_eq(&self, other: impl AsRef<Self>) -> bool {
        self.iter()
            .zip(other.as_ref().iter())
            .all(|(a, b)| (a - b).abs() < 1e-4)
    }
}

impl Approx for Vec<f64> {
    fn approx_eq(&self, other: impl AsRef<Self>) -> bool {
        self.iter()
            .zip(other.as_ref().iter())
            .all(|(a, b)| (a - b).abs() < 1e-4)
    }
}
