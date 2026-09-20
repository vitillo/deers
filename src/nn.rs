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
use crate::ops::{self, TensorOp};
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
    ///
    /// Names, counts, shapes, and dtypes must match the registered parameters.
    /// Every mismatch fails loudly with the file and tensor named.
    pub fn load(&self, path: &Path, device: Device) -> Result<()> {
        let loaded = checkpoint::load_tensors(path, device)?;
        let origins = loaded
            .keys()
            .map(|name| (name.clone(), path.to_path_buf()))
            .collect::<BTreeMap<_, _>>();
        self.assign(loaded, &origins, &path.display().to_string())
    }

    /// Loads parameter values from a sharded checkpoint directory onto `device`.
    ///
    /// Shard files resolve through [`checkpoint::discover_shard_files`] and
    /// Qwen3 tensor names map onto deers names. Validation matches [`load`].
    pub fn load_sharded(&self, dir: &Path, device: Device) -> Result<()> {
        let (loaded, origins) = checkpoint::load_sharded_tracked(dir, device)?;
        self.assign(loaded, &origins, &dir.display().to_string())
    }

    /// Assigns loaded tensors to the registered parameters after validation.
    ///
    /// Unexpected names fail first so a stray tensor cannot hide behind the
    /// count check. Counts, missing names, dtypes, and shapes follow in order.
    /// `origins` maps each tensor to the shard file it came from. `source`
    /// names the checkpoint file or directory for count-level errors.
    fn assign(
        &self,
        loaded: BTreeMap<String, Tensor>,
        origins: &BTreeMap<String, std::path::PathBuf>,
        source: &str,
    ) -> Result<()> {
        let params = self.params.borrow();
        for name in loaded.keys() {
            if !params.contains_key(name) {
                let file = origins
                    .get(name)
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| source.to_owned());
                return Err(crate::error::Error::Checkpoint(format!(
                    "unexpected tensor '{name}' in '{file}': no matching parameter"
                )));
            }
        }
        if loaded.len() != params.len() {
            return Err(crate::error::Error::Checkpoint(format!(
                "checkpoint tensor count mismatch: expected {}, found {} in '{source}'",
                params.len(),
                loaded.len()
            )));
        }

        for (name, parameter) in params.iter() {
            let tensor = loaded.get(name).ok_or_else(|| {
                crate::error::Error::Checkpoint(format!("missing parameter '{name}' in '{source}'"))
            })?;
            let file = origins
                .get(name)
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| source.to_owned());
            if tensor.dtype() != parameter.dtype() {
                return Err(crate::error::Error::Checkpoint(format!(
                    "dtype mismatch for tensor '{name}' in '{file}': expected {}, found {}",
                    parameter.dtype(),
                    tensor.dtype()
                )));
            }
            if tensor.layout().shape() != parameter.layout().shape() {
                return Err(crate::error::Error::Checkpoint(format!(
                    "shape mismatch for tensor '{name}' in '{file}': expected {}, found {}",
                    parameter.layout().shape(),
                    tensor.layout().shape()
                )));
            }
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
        Self { weight, bias, training: Cell::new(true) }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight);
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
        let requested = x.dtype();
        // F16 tops out at 65504, so squaring large activations overflows to
        // infinity before the mean runs. Accumulate the variance in F32 and
        // convert back to the requested dtype; the casts stay in the graph so
        // low-precision training keeps gradient flow. Other dtypes keep the
        // direct path.
        let normed = if matches!(requested, DType::F16 | DType::BF16) {
            let acc = ops::Cast::new((*x).clone(), DType::F32)?.forward()?;
            let mean_sq = (&acc * &acc).mean(vec![last_axis], true);
            let inv_norm = (mean_sq + self.eps).scalar_powf(-0.5);
            ops::Cast::new(&acc * &inv_norm, requested)?.forward()?
        } else {
            let mean_sq = (x * x).mean(vec![last_axis], true);
            let inv_norm = (mean_sq + self.eps).scalar_powf(-0.5);
            x * &inv_norm
        };
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

/// Element-wise GELU activation (tanh approximation).
#[derive(Debug)]
pub struct GELU;

impl Module for GELU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.gelu())
    }
}

/// Element-wise SiLU (swish) activation.
#[derive(Debug)]
pub struct SiLU;

impl Module for SiLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.silu())
    }
}

/// Gated SiLU MLP: `down(silu(gate(x)) * up(x))`.
///
/// Follows the Llama feed-forward recipe with three bias-free projections.
#[derive(Debug)]
pub struct SwiGLU {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    out_features: usize,
}

impl SwiGLU {
    /// Creates a SwiGLU module whose projections are registered under `builder`.
    pub fn new(
        builder: ParamBuilder,
        in_features: usize,
        hidden_dim: usize,
        out_features: usize,
    ) -> Self {
        Self {
            gate_proj: Linear::no_bias(builder.pp("gate_proj"), in_features, hidden_dim),
            up_proj: Linear::no_bias(builder.pp("up_proj"), in_features, hidden_dim),
            down_proj: Linear::no_bias(builder.pp("down_proj"), hidden_dim, out_features),
            out_features,
        }
    }

    /// Runs the gate, up, and down projections on a flat `[rows, in]` input.
    ///
    /// The core stays fixed rank. Only the wrapper below folds leading dims,
    /// and that fold is rank-polymorphic through the ellipsis form.
    fn forward_flat(&self, x_flat: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x_flat)?.silu();
        let up = self.up_proj.forward(x_flat)?;
        self.down_proj.forward(&(&gate * &up))
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.layout().shape();
        assert!(shape.ndim() >= 2, "SwiGLU expects input with rank >= 2");
        let mut out_shape: Vec<usize> = (0..shape.ndim() - 1).map(|i| shape[i]).collect();
        let x_flat = x.rearrange("... d -> (...) d", &[]);
        let y = self.forward_flat(&x_flat)?;
        out_shape.push(self.out_features);
        Ok(y.reshape(out_shape))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = self.gate_proj.parameters();
        parameters.extend(self.up_proj.parameters());
        parameters.extend(self.down_proj.parameters());
        parameters
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
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::{Embedding, Linear, Module, ParamStore};
    use crate::checkpoint::save_tensors;
    use crate::{Device, Tensor};

    fn sharded_case(case: &str) -> (ParamStore, PathBuf) {
        let store = ParamStore::new();
        let root = store.root();
        let _wte = Embedding::new(root.pp("wte"), 4, 2);
        let _lm_head = Linear::no_bias(root.pp("lm_head"), 2, 3);
        let dir =
            std::env::temp_dir().join(format!("deers-param-sharded-{case}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        (store, dir)
    }

    fn write_shard(dir: &Path, filename: &str, tensors: &BTreeMap<String, Tensor>) {
        save_tensors(&dir.join(filename), tensors).unwrap();
    }

    fn write_index(dir: &Path, entries: &[(&str, &str)]) {
        let weight_map: BTreeMap<&str, &str> = entries.iter().copied().collect();
        let index = serde_json::json!({"metadata": {"total_size": 0}, "weight_map": weight_map});
        fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&index).unwrap())
            .unwrap();
    }

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

    #[test]
    fn test_param_store_load_sharded_maps_qwen_names() {
        // Arrange: Qwen3 names split across two indexed shards.
        let (store, dir) = sharded_case("mapped");
        let mut first = BTreeMap::new();
        first.insert(
            "model.embed_tokens.weight".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (4, 2), Device::Cpu),
        );
        let mut second = BTreeMap::new();
        second.insert(
            "lm_head.weight".to_owned(),
            Tensor::from_vec(vec![9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0], (2, 3), Device::Cpu),
        );
        write_shard(&dir, "model-00001-of-00002.safetensors", &first);
        write_shard(&dir, "model-00002-of-00002.safetensors", &second);
        write_index(
            &dir,
            &[
                ("model.embed_tokens.weight", "model-00001-of-00002.safetensors"),
                ("lm_head.weight", "model-00002-of-00002.safetensors"),
            ],
        );

        // Act
        store.load_sharded(&dir, Device::Cpu).unwrap();

        // Assert: values land on the deers-named parameters on the CPU.
        let params: BTreeMap<_, _> = store.named_parameters().into_iter().collect();
        assert_eq!(
            params["wte.weight"].to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            params["lm_head.weight"].to_vec::<f32>().unwrap(),
            vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        );
        assert_eq!(params["wte.weight"].device(), Device::Cpu);
        assert_eq!(params["lm_head.weight"].device(), Device::Cpu);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_param_store_load_sharded_rejects_unexpected_name() {
        // Arrange: a tensor outside the Qwen weight layout next to a valid shard.
        let (store, dir) = sharded_case("unexpected");
        let mut first = BTreeMap::new();
        first.insert(
            "model.embed_tokens.weight".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (4, 2), Device::Cpu),
        );
        let mut second = BTreeMap::new();
        second.insert(
            "model.layers.0.self_attn.q_proj.bias".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), Device::Cpu),
        );
        write_shard(&dir, "a.safetensors", &first);
        write_shard(&dir, "b.safetensors", &second);

        // Act
        let error = store.load_sharded(&dir, Device::Cpu).unwrap_err().to_string();

        // Assert
        assert_eq!(
            error,
            format!(
                "unexpected tensor 'model.layers.0.self_attn.q_proj.bias' in '{}': no matching parameter",
                dir.join("b.safetensors").display()
            )
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_param_store_load_sharded_rejects_count_mismatch() {
        // Arrange: one shard while the store holds two parameters.
        let (store, dir) = sharded_case("count");
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (4, 2), Device::Cpu),
        );
        write_shard(&dir, "model.safetensors", &tensors);

        // Act
        let error = store.load_sharded(&dir, Device::Cpu).unwrap_err().to_string();

        // Assert
        assert_eq!(
            error,
            format!("checkpoint tensor count mismatch: expected 2, found 1 in '{}'", dir.display())
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_param_store_load_sharded_rejects_shape_mismatch() {
        // Arrange: the embedding shard carries the wrong second dimension.
        let (store, dir) = sharded_case("shape");
        let mut first = BTreeMap::new();
        first.insert(
            "model.embed_tokens.weight".to_owned(),
            Tensor::from_vec(
                vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                (4, 3),
                Device::Cpu,
            ),
        );
        let mut second = BTreeMap::new();
        second.insert(
            "lm_head.weight".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu),
        );
        write_shard(&dir, "a.safetensors", &first);
        write_shard(&dir, "b.safetensors", &second);

        // Act
        let error = store.load_sharded(&dir, Device::Cpu).unwrap_err().to_string();

        // Assert
        assert_eq!(
            error,
            format!(
                "shape mismatch for tensor 'wte.weight' in '{}': expected [4, 2], found [4, 3]",
                dir.join("a.safetensors").display()
            )
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_param_store_load_sharded_rejects_dtype_mismatch() {
        // Arrange: the embedding shard stores BF16 while the parameter is F32.
        let (store, dir) = sharded_case("dtype");
        let mut first = BTreeMap::new();
        first.insert(
            "model.embed_tokens.weight".to_owned(),
            Tensor::from_vec(vec![half::bf16::from_f32(1.0); 8], (4, 2), Device::Cpu),
        );
        let mut second = BTreeMap::new();
        second.insert(
            "lm_head.weight".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu),
        );
        write_shard(&dir, "a.safetensors", &first);
        write_shard(&dir, "b.safetensors", &second);

        // Act
        let error = store.load_sharded(&dir, Device::Cpu).unwrap_err().to_string();

        // Assert
        assert_eq!(
            error,
            format!(
                "dtype mismatch for tensor 'wte.weight' in '{}': expected f32, found bf16",
                dir.join("a.safetensors").display()
            )
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
