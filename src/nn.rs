//! Neural network modules: traits, layers, and composition.

/// Stateless neural-network helper functions.
pub mod functional;
/// Trainable tensor wrapper used by modules and optimizers.
pub mod parameter;

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;

use crate::checkpoint;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::{DType, Device};
pub use parameter::Parameter;

/// A named registry of parameters for checkpoint save/load.
#[derive(Clone, Debug, Default)]
pub struct ParamStore {
    params: Rc<RefCell<BTreeMap<String, Parameter>>>,
}

impl ParamStore {
    /// Creates an empty parameter store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a root builder for registering parameters under hierarchical names.
    pub fn root(&self) -> ParamBuilder {
        ParamBuilder { store: self.clone(), prefix: String::new() }
    }

    /// Returns every registered parameter with its stable checkpoint name.
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.params
            .borrow()
            .iter()
            .map(|(name, parameter)| (name.clone(), parameter.clone()))
            .collect()
    }

    /// Returns every registered parameter without its names.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.named_parameters().into_iter().map(|(_, parameter)| parameter).collect()
    }

    /// Saves the current parameter values as a safetensors checkpoint.
    pub fn save(&self, path: &Path) -> Result<()> {
        let tensors = self
            .params
            .borrow()
            .iter()
            .map(|(name, parameter)| (name.clone(), parameter.detach()))
            .collect::<BTreeMap<_, _>>();
        checkpoint::save_tensors(path, &tensors)
    }

    /// Loads parameter values from a safetensors checkpoint onto `device`.
    pub fn load(&self, path: &Path, device: Device) -> Result<()> {
        let loaded = checkpoint::load_tensors(path, device)?;
        let params = self.params.borrow();
        if loaded.len() != params.len() {
            return Err(crate::error::Error::Checkpoint(format!(
                "checkpoint tensor count mismatch: expected {}, found {}",
                params.len(),
                loaded.len()
            )));
        }

        for (name, parameter) in params.iter() {
            let tensor = loaded.get(name).ok_or_else(|| {
                crate::error::Error::Checkpoint(format!("missing parameter in checkpoint: {name}"))
            })?;
            parameter.set(tensor)?;
        }

        Ok(())
    }

    fn register(&self, name: String, tensor: Tensor) -> Parameter {
        let parameter = Parameter::new(tensor);
        let old = self.params.borrow_mut().insert(name.clone(), parameter.clone());
        assert!(old.is_none(), "duplicate parameter name: {name}");
        parameter
    }
}

/// A lightweight builder that prefixes parameter names during construction.
#[derive(Clone, Debug)]
pub struct ParamBuilder {
    store: ParamStore,
    prefix: String,
}

impl ParamBuilder {
    /// Returns a child builder under `segment`, like `blocks.0` or `fc1`.
    pub fn pp(&self, segment: impl AsRef<str>) -> Self {
        let segment = segment.as_ref();
        assert!(!segment.is_empty(), "parameter path segment must not be empty");

        let prefix = if self.prefix.is_empty() {
            segment.to_owned()
        } else {
            format!("{}.{}", self.prefix, segment)
        };
        Self { store: self.store.clone(), prefix }
    }

    /// Registers `tensor` as a parameter named `prefix.name`.
    pub fn param(&self, name: &str, tensor: Tensor) -> Parameter {
        self.store.register(self.full_name(name), tensor)
    }

    fn full_name(&self, name: &str) -> String {
        assert!(!name.is_empty(), "parameter name must not be empty");
        if self.prefix.is_empty() { name.to_owned() } else { format!("{}.{}", self.prefix, name) }
    }
}

/// A neural network layer or model.
pub trait Module {
    /// Runs the module on the input tensor.
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Returns all trainable parameters in this module.
    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }

    /// Moves every parameter in the module to `device`.
    fn to_device(&self, device: Device) -> Result<()> {
        for parameter in self.parameters() {
            parameter.to_device(device)?;
        }
        Ok(())
    }

    /// Sets the module and its children to training mode.
    fn train(&self) {}

    /// Sets the module and its children to evaluation mode.
    fn eval(&self) {}

    /// Returns true when the module is in training mode.
    fn is_training(&self) -> bool {
        true
    }
}

/// Fully connected layer: `y = x @ weight` (+ optional bias).
#[derive(Debug)]
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    transpose_weight: bool,
    training: Cell<bool>,
}

impl Linear {
    /// Creates a Linear layer with uniform [-k, k] initialization (k = 1/sqrt(in)).
    pub fn new(builder: ParamBuilder, in_features: usize, out_features: usize) -> Self {
        Self::new_inner(builder, in_features, out_features, true)
    }

    /// Creates a Linear layer without bias.
    pub fn no_bias(builder: ParamBuilder, in_features: usize, out_features: usize) -> Self {
        Self::new_inner(builder, in_features, out_features, false)
    }

    /// Creates a bias-free Linear layer that reuses an existing weight
    /// parameter instead of registering a new one (for LM head weight tying).
    ///
    /// The shared weight keeps its embedding `[out, in]` layout and is
    /// transposed on the fly, so `y = x @ weight.t()`.
    pub fn shared_weight(weight: Parameter) -> Self {
        Self { weight, bias: None, transpose_weight: true, training: Cell::new(true) }
    }

    /// Returns the layer weight parameter.
    pub fn weight(&self) -> Parameter {
        self.weight.clone()
    }

    fn new_inner(
        builder: ParamBuilder,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Self {
        let k = 1.0 / (in_features as f64).sqrt();
        let weight = builder.param(
            "weight",
            Tensor::rand((in_features, out_features), DType::F32, Device::Cpu) * 2.0 * k - k,
        );
        let bias = if bias {
            Some(builder.param(
                "bias",
                Tensor::rand((out_features,), DType::F32, Device::Cpu) * 2.0 * k - k,
            ))
        } else {
            None
        };
        Self { weight, bias, transpose_weight: false, training: Cell::new(true) }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = self.weight.transpose(None);
        let out = if self.transpose_weight { x.matmul(&weight) } else { x.matmul(&self.weight) };
        match &self.bias {
            Some(bias) => Ok(&out + &**bias),
            None => Ok(out),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            parameters.push(bias.clone());
        }
        parameters
    }

    fn train(&self) {
        self.training.set(true);
    }

    fn eval(&self) {
        self.training.set(false);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Embedding lookup table: maps integer indices to dense vectors.
#[derive(Debug)]
pub struct Embedding {
    weight: Parameter,
    training: Cell<bool>,
}

impl Embedding {
    /// Creates an embedding table with standard normal initialization.
    pub fn new(builder: ParamBuilder, vocab_size: usize, hidden_size: usize) -> Self {
        let weight = builder
            .param("weight", Tensor::randn((vocab_size, hidden_size), DType::F32, Device::Cpu));
        Self { weight, training: Cell::new(true) }
    }

    /// Returns the embedding table parameter (e.g. for LM head weight tying).
    pub fn weight(&self) -> Parameter {
        self.weight.clone()
    }
}

impl Module for Embedding {
    fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        let hidden_size = self.weight.layout().shape()[1];
        let mut final_dims: Vec<usize> = indices.layout().shape().iter().copied().collect();
        final_dims.push(hidden_size);
        let flat = indices.reshape(vec![indices.layout().size()]);
        let selected = self.weight.index_select(0, &flat);
        Ok(selected.reshape(final_dims))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }

    fn train(&self) {
        self.training.set(true);
    }

    fn eval(&self) {
        self.training.set(false);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// RMSNorm: `x / sqrt(mean(x²) + eps) * weight`.
///
/// The weightless `new` constructor is kept for blocks that own no
/// parameters. Prefer `new_affine` so the scale is trainable.
#[derive(Debug)]
pub struct RMSNorm {
    weight: Option<Parameter>,
    eps: f64,
    training: Cell<bool>,
}

impl RMSNorm {
    /// Creates a weightless RMSNorm layer with epsilon `eps`.
    pub fn new(eps: f64) -> Self {
        Self { weight: None, eps, training: Cell::new(true) }
    }

    /// Creates an RMSNorm layer with a trainable scale initialized to ones.
    pub fn new_affine(builder: ParamBuilder, hidden_size: usize, eps: f64) -> Self {
        let weight = builder.param("weight", Tensor::ones((hidden_size,), DType::F32, Device::Cpu));
        Self { weight: Some(weight), eps, training: Cell::new(true) }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_axis = x.layout().ndim() - 1;
        let mean_sq = (x * x).mean(vec![last_axis], true);
        let inv_norm = (mean_sq + self.eps).scalar_powf(-0.5);
        let normed = x * &inv_norm;
        match &self.weight {
            Some(weight) => Ok(&normed * &**weight),
            None => Ok(normed),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.weight.clone().into_iter().collect()
    }

    fn train(&self) {
        self.training.set(true);
    }

    fn eval(&self) {
        self.training.set(false);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// LayerNorm over the last dimension: `(x - mean) / sqrt(var + eps) * weight + bias`.
///
/// Matches PyTorch `LayerNorm` with `elementwise_affine` and the candle
/// `LayerNorm` forward path.
#[derive(Debug)]
pub struct LayerNorm {
    weight: Parameter,
    bias: Option<Parameter>,
    eps: f64,
    training: Cell<bool>,
}

impl LayerNorm {
    /// Creates a LayerNorm with weight initialized to ones and bias to zeros.
    pub fn new(builder: ParamBuilder, normalized_shape: usize, eps: f64) -> Self {
        let weight =
            builder.param("weight", Tensor::ones((normalized_shape,), DType::F32, Device::Cpu));
        let bias =
            builder.param("bias", Tensor::zeros((normalized_shape,), DType::F32, Device::Cpu));
        Self { weight, bias: Some(bias), eps, training: Cell::new(true) }
    }

    /// Creates a LayerNorm with weight initialized to ones and no bias.
    pub fn no_bias(builder: ParamBuilder, normalized_shape: usize, eps: f64) -> Self {
        let weight =
            builder.param("weight", Tensor::ones((normalized_shape,), DType::F32, Device::Cpu));
        Self { weight, bias: None, eps, training: Cell::new(true) }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_axis = x.layout().ndim() - 1;
        let hidden = x.layout().shape()[last_axis];
        assert_eq!(
            hidden,
            self.weight.layout().shape()[0],
            "LayerNorm input last dim must match normalized shape"
        );
        let mean = x.mean(vec![last_axis], true);
        let centered = x - &mean;
        let var = (&centered * &centered).mean(vec![last_axis], true);
        let normed = &centered * &(var + self.eps).scalar_powf(-0.5);
        let scaled = &normed * &*self.weight;
        match &self.bias {
            Some(bias) => Ok(&scaled + &**bias),
            None => Ok(scaled),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            parameters.push(bias.clone());
        }
        parameters
    }

    fn train(&self) {
        self.training.set(true);
    }

    fn eval(&self) {
        self.training.set(false);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Element-wise ReLU activation.
#[derive(Debug)]
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.relu())
    }
}

/// Inverted dropout: zeroes elements with probability `p` while training.
///
/// Evaluation is an exact identity. Has no parameters.
#[derive(Debug)]
pub struct Dropout {
    p: f64,
    training: Cell<bool>,
}

impl Dropout {
    /// Creates a Dropout layer with drop probability `p` in `[0, 1)`.
    pub fn new(p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout probability must be in [0, 1), got {p}");
        Self { p, training: Cell::new(true) }
    }

    /// Returns the drop probability.
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(functional::dropout(x, self.p, self.training.get()))
    }

    fn train(&self) {
        self.training.set(true);
    }

    fn eval(&self) {
        self.training.set(false);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// A sequence of modules applied in order.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    training: Cell<bool>,
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sequential({} layers)", self.layers.len())
    }
}

impl Sequential {
    /// Appends a layer to the sequence.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }

    fn train(&self) {
        self.training.set(true);
        for layer in &self.layers {
            layer.train();
        }
    }

    fn eval(&self) {
        self.training.set(false);
        for layer in &self.layers {
            layer.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Creates an empty Sequential to build with `.add()`.
pub fn seq() -> Sequential {
    Sequential { layers: vec![], training: Cell::new(true) }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{Linear, Module, ParamStore};
    use crate::{Device, Tensor};

    #[test]
    fn test_param_store_registers_hierarchical_names() {
        // Arrange
        let store = ParamStore::new();
        let root = store.root();

        // Act
        let _fc1 = Linear::new(root.pp("fc1"), 4, 3);
        let _fc2 = Linear::no_bias(root.pp("fc2"), 3, 2);
        let names = store.named_parameters().into_iter().map(|(name, _)| name).collect::<Vec<_>>();

        // Assert
        assert_eq!(names, vec!["fc1.bias", "fc1.weight", "fc2.weight"]);
    }

    #[test]
    fn test_param_store_save_and_load_roundtrip() {
        // Arrange
        let path = std::env::temp_dir().join(format!(
            "deers-param-store-{}-{}.safetensors",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let store = ParamStore::new();
        let root = store.root();
        let linear = Linear::new(root.pp("linear"), 2, 2);
        let original = linear.parameters()[0].to_vec::<f32>().unwrap();
        linear.parameters()[0]
            .set(&Tensor::from_vec(vec![9.0f32, 8.0, 7.0, 6.0], (2, 2), Device::Cpu))
            .unwrap();

        // Act
        store.save(&path).unwrap();
        linear.parameters()[0]
            .set(&Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], (2, 2), Device::Cpu))
            .unwrap();
        store.load(&path, Device::Cpu).unwrap();
        let restored = linear.parameters()[0].to_vec::<f32>().unwrap();

        // Assert
        assert_eq!(restored, vec![9.0, 8.0, 7.0, 6.0]);
        assert_ne!(restored, original);

        let _ = fs::remove_file(path);
    }
}
