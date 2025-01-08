pub trait Approx {
    fn assert_approx_eq(_: impl AsRef<Self>, _: impl AsRef<Self>);
}

impl Approx for Vec<f32> {
    fn assert_approx_eq(arg1: impl AsRef<Self>, arg2: impl AsRef<Self>) {
        assert!(arg1
            .as_ref()
            .iter()
            .zip(arg2.as_ref().iter())
            .all(|(a, b)| (a - b).abs() < 1e-4))
    }
}

impl Approx for Vec<f64> {
    fn assert_approx_eq(arg1: impl AsRef<Self>, arg2: impl AsRef<Self>) {
        assert!(arg1
            .as_ref()
            .iter()
            .zip(arg2.as_ref().iter())
            .all(|(a, b)| (a - b).abs() < 1e-4))
    }
}
