//! Safetensors-based checkpoint serialization for model and optimizer state.
//!
//! Sharded checkpoints follow the Hugging Face layout. A directory holds one
//! or more `model-00001-of-00002.safetensors` shard files. When present,
//! `model.safetensors.index.json` is authoritative: its `weight_map` object
//! assigns every tensor name to its shard file, and loading reads exactly the
//! files the map lists. Without an index, loading reads every `*.safetensors`
//! file in the directory in sorted order.
//!
//! Qwen3 weight names map onto deers GPT names for the layers both models
//! share. `model.embed_tokens.weight` becomes `wte.weight`,
//! `model.layers.{l}.self_attn.{q,k,v}_proj.weight` become
//! `blocks.{l}.attn.{q,k,v}_proj.weight`, `o_proj` becomes `out_proj`, the
//! `q_norm` and `k_norm` weights keep their names under `attn`, the MLP
//! `up_proj` and `down_proj` weights keep theirs under `mlp`, and
//! `lm_head.weight` is unchanged. Qwen-only tensors (`gate_proj`, both
//! layernorms, `model.norm`) have no deers counterpart and fail loudly at
//! assignment with the shard file and tensor named.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use safetensors::tensor::View;
use safetensors::{Dtype as SafeDtype, SafeTensors, serialize_to_file};

use crate::error::{Error, Result};
use crate::{DType, Device, Tensor};

struct TensorBlob {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &TensorBlob {
    fn dtype(&self) -> SafeDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Saves a named tensor map to a safetensors file at `path`.
pub fn save_tensors(path: &Path, tensors: &BTreeMap<String, Tensor>) -> Result<()> {
    let blobs: Vec<_> = tensors
        .iter()
        .map(|(name, tensor)| Ok((name.clone(), tensor_blob(tensor)?)))
        .collect::<Result<_>>()?;
    let views = blobs.iter().map(|(name, tensor)| (name.as_str(), tensor)).collect::<Vec<_>>();
    serialize_to_file(views, None, path)?;
    Ok(())
}

/// Loads a named tensor map from a safetensors file onto `device`.
pub fn load_tensors(path: &Path, device: Device) -> Result<BTreeMap<String, Tensor>> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut loaded = BTreeMap::new();

    for name in tensors.names() {
        let tensor = read_tensor(&tensors, name, device)?;
        loaded.insert(name.to_owned(), tensor);
    }

    Ok(loaded)
}

/// Maps a Hugging Face Qwen3 tensor name onto a deers GPT parameter name.
///
/// Returns `None` for Qwen-only tensors with no deers counterpart
/// (`gate_proj`, both layernorms, `model.norm`) and for names outside the
/// Qwen layout. Callers keep unmapped names as-is so deers-native shards
/// load through the same path.
pub fn map_qwen_name(hf_name: &str) -> Option<String> {
    if hf_name == "model.embed_tokens.weight" {
        return Some("wte.weight".to_owned());
    }
    if hf_name == "lm_head.weight" {
        return Some("lm_head.weight".to_owned());
    }
    let rest = hf_name.strip_prefix("model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    if layer.parse::<usize>().is_err() {
        return None;
    }
    let mapped = match rest {
        "self_attn.q_proj.weight" => format!("blocks.{layer}.attn.q_proj.weight"),
        "self_attn.k_proj.weight" => format!("blocks.{layer}.attn.k_proj.weight"),
        "self_attn.v_proj.weight" => format!("blocks.{layer}.attn.v_proj.weight"),
        "self_attn.o_proj.weight" => format!("blocks.{layer}.attn.out_proj.weight"),
        "self_attn.q_norm.weight" => format!("blocks.{layer}.attn.q_norm.weight"),
        "self_attn.k_norm.weight" => format!("blocks.{layer}.attn.k_norm.weight"),
        "mlp.up_proj.weight" => format!("blocks.{layer}.mlp.up_proj.weight"),
        "mlp.down_proj.weight" => format!("blocks.{layer}.mlp.down_proj.weight"),
        _ => return None,
    };
    Some(mapped)
}

/// Index file parsed from `model.safetensors.index.json`. Only `weight_map`
/// matters here. Size metadata stays unread.
#[derive(serde::Deserialize)]
struct ShardIndex {
    weight_map: BTreeMap<String, String>,
}

/// Discovers shard files in `dir` in load order.
///
/// Reads `model.safetensors.index.json` when present and returns exactly the
/// files its `weight_map` lists. Otherwise returns every `*.safetensors` file
/// in sorted order. A missing shard listed by the index, an empty weight map,
/// and a directory with no shards all fail loudly with the path named.
pub fn discover_shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let bytes = fs::read(&index_path)?;
        let index: ShardIndex = serde_json::from_slice(&bytes).map_err(|e| {
            Error::Checkpoint(format!(
                "failed to parse shard index '{}': {e}",
                index_path.display()
            ))
        })?;
        let files = index
            .weight_map
            .values()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|name| dir.join(name))
            .collect::<Vec<_>>();
        if files.is_empty() {
            return Err(Error::Checkpoint(format!(
                "empty weight map in shard index '{}'",
                index_path.display()
            )));
        }
        for file in &files {
            if !file.exists() {
                return Err(Error::Checkpoint(format!(
                    "missing shard file '{}' listed in index '{}'",
                    file.display(),
                    index_path.display()
                )));
            }
        }
        return Ok(files);
    }

    let mut files = vec![];
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(Error::Checkpoint(format!(
            "no safetensors shard files in '{}'",
            dir.display()
        )));
    }
    Ok(files)
}

/// Loads a sharded checkpoint directory onto `device`, keyed by deers names.
///
/// Qwen3 tensor names map through [`map_qwen_name`]. A tensor listed twice
/// fails loudly with both shard files named. File dtypes convert to matching
/// deers dtypes including BF16. See [`load_sharded_tracked`] for per-tensor
/// file origins.
pub fn load_sharded(dir: &Path, device: Device) -> Result<BTreeMap<String, Tensor>> {
    Ok(load_sharded_tracked(dir, device)?.0)
}

/// Loads a sharded checkpoint with the shard file each tensor came from.
///
/// The [`ParamStore`](crate::nn::ParamStore) sharded loader uses the origins
/// to name the exact file in mismatch errors.
pub(crate) fn load_sharded_tracked(
    dir: &Path,
    device: Device,
) -> Result<(BTreeMap<String, Tensor>, BTreeMap<String, PathBuf>)> {
    let mut merged: BTreeMap<String, Tensor> = BTreeMap::new();
    let mut origins: BTreeMap<String, PathBuf> = BTreeMap::new();
    for file in discover_shard_files(dir)? {
        let bytes = fs::read(&file).map_err(|e| {
            Error::Checkpoint(format!("failed to read shard file '{}': {e}", file.display()))
        })?;
        let shards = SafeTensors::deserialize(&bytes).map_err(|e| {
            Error::Checkpoint(format!("failed to parse shard file '{}': {e}", file.display()))
        })?;
        for name in shards.names() {
            let deers_name = map_qwen_name(name).unwrap_or_else(|| name.to_owned());
            if let Some(prev) = origins.get(&deers_name) {
                return Err(Error::Checkpoint(format!(
                    "duplicate tensor '{deers_name}' in shard files '{}' and '{}'",
                    prev.display(),
                    file.display()
                )));
            }
            let tensor = read_tensor(&shards, name, device).map_err(|e| {
                Error::Checkpoint(format!(
                    "tensor '{name}' in shard file '{}': {e}",
                    file.display()
                ))
            })?;
            origins.insert(deers_name.clone(), file.clone());
            merged.insert(deers_name, tensor);
        }
    }
    Ok((merged, origins))
}

fn read_tensor(tensors: &SafeTensors<'_>, name: &str, device: Device) -> Result<Tensor> {
    let view = tensors.tensor(name)?;
    let shape = view.shape().to_vec();
    let tensor = match view.dtype() {
        SafeDtype::F16 => {
            let values = view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| f16::from_bits(u16::from_le_bytes(*chunk)))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)
        }
        SafeDtype::BF16 => {
            let values = view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| bf16::from_bits(u16::from_le_bytes(*chunk)))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)
        }
        SafeDtype::F32 => {
            let values = view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)
        }
        SafeDtype::I64 => {
            let values = view
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| i64::from_le_bytes(*chunk))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)
        }
        other => {
            return Err(Error::Checkpoint(format!(
                "unsupported safetensors dtype in checkpoint: {other:?}"
            )));
        }
    };
    Ok(tensor)
}

fn tensor_blob(tensor: &Tensor) -> Result<TensorBlob> {
    let shape = tensor.layout().shape().iter().copied().collect::<Vec<_>>();
    let blob = match tensor.dtype() {
        DType::F16 => {
            let data = tensor
                .to_vec::<f16>()?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            TensorBlob { dtype: SafeDtype::F16, shape, data }
        }
        DType::BF16 => {
            let data = tensor
                .to_vec::<bf16>()?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            TensorBlob { dtype: SafeDtype::BF16, shape, data }
        }
        DType::F32 => {
            let data = tensor
                .to_vec::<f32>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            TensorBlob { dtype: SafeDtype::F32, shape, data }
        }
        DType::I64 => {
            let data = tensor
                .to_vec::<i64>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            TensorBlob { dtype: SafeDtype::I64, shape, data }
        }
    };
    Ok(blob)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use half::bf16;

    use super::{discover_shard_files, load_sharded, load_tensors, map_qwen_name, save_tensors};
    use crate::{DType, Device, Tensor};

    fn shard_dir(case: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("deers-sharded-{case}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
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
    fn test_save_and_load_tensors_roundtrip() {
        // Arrange
        let path = std::env::temp_dir().join(format!(
            "deers-checkpoint-roundtrip-{}-{}.safetensors",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "linear.weight".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu),
        );
        tensors.insert("tokens".to_owned(), Tensor::from_vec(vec![1i64, 2, 3], (3,), Device::Cpu));

        // Act
        save_tensors(&path, &tensors).unwrap();
        let loaded = load_tensors(&path, Device::Cpu).unwrap();

        // Assert
        assert_eq!(loaded["linear.weight"].to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded["tokens"].to_vec::<i64>().unwrap(), vec![1, 2, 3]);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_save_and_load_bf16_tensors_roundtrip() {
        // Arrange
        let path = std::env::temp_dir().join(format!(
            "deers-checkpoint-bf16-roundtrip-{}-{}.safetensors",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let weight = vec![bf16::from_f32(1.5), bf16::from_f32(-2.0), bf16::from_f32(3.140625)];
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "linear.weight".to_owned(),
            Tensor::from_vec(weight.clone(), (3,), Device::Cpu),
        );
        tensors.insert("linear.bias".to_owned(), Tensor::from_vec(vec![0.5f32], (1,), Device::Cpu));

        // Act
        save_tensors(&path, &tensors).unwrap();
        let loaded = load_tensors(&path, Device::Cpu).unwrap();

        // Assert: the BF16 file dtype maps back to BF16 with identical bits,
        // and the F32 control tensor is untouched.
        assert_eq!(loaded["linear.weight"].dtype(), DType::BF16);
        assert_eq!(loaded["linear.weight"].to_vec::<bf16>().unwrap(), weight);
        let as_f32: Vec<f32> =
            loaded["linear.weight"].to_vec::<bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
        assert_eq!(as_f32, vec![1.5, -2.0, 3.140625]);
        assert_eq!(loaded["linear.bias"].dtype(), DType::F32);
        assert_eq!(loaded["linear.bias"].to_vec::<f32>().unwrap(), vec![0.5]);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_map_qwen_name_covers_shared_layers() {
        // Arrange: Qwen3 names for the layers both models share.
        let cases = [
            ("model.embed_tokens.weight", "wte.weight"),
            ("lm_head.weight", "lm_head.weight"),
            ("model.layers.0.self_attn.q_proj.weight", "blocks.0.attn.q_proj.weight"),
            ("model.layers.2.self_attn.k_proj.weight", "blocks.2.attn.k_proj.weight"),
            ("model.layers.1.self_attn.v_proj.weight", "blocks.1.attn.v_proj.weight"),
            ("model.layers.0.self_attn.o_proj.weight", "blocks.0.attn.out_proj.weight"),
            ("model.layers.3.self_attn.q_norm.weight", "blocks.3.attn.q_norm.weight"),
            ("model.layers.3.self_attn.k_norm.weight", "blocks.3.attn.k_norm.weight"),
            ("model.layers.0.mlp.up_proj.weight", "blocks.0.mlp.up_proj.weight"),
            ("model.layers.0.mlp.down_proj.weight", "blocks.0.mlp.down_proj.weight"),
        ];

        // Act
        let mapped: Vec<Option<String>> = cases.iter().map(|(hf, _)| map_qwen_name(hf)).collect();

        // Assert
        let expected: Vec<Option<String>> =
            cases.iter().map(|(_, deers)| Some((*deers).to_owned())).collect();
        assert_eq!(mapped, expected);
    }

    #[test]
    fn test_map_qwen_name_rejects_qwen_only_and_foreign_names() {
        // Arrange: Qwen-only tensors plus names outside the Qwen layout.
        let cases = [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
            "model.layers.x.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "custom.weight",
            "",
        ];

        // Act
        let mapped: Vec<Option<String>> = cases.iter().map(|name| map_qwen_name(name)).collect();

        // Assert
        assert_eq!(mapped, vec![None, None, None, None, None, None, None, None]);
    }

    #[test]
    fn test_load_sharded_maps_qwen_names_and_places_on_device() {
        // Arrange: two shards with Qwen3 names plus an index assigning each
        // tensor to its shard file.
        let dir = shard_dir("mapped");
        let mut first = BTreeMap::new();
        first.insert(
            "model.embed_tokens.weight".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu),
        );
        let mut second = BTreeMap::new();
        second.insert(
            "model.layers.0.self_attn.q_proj.weight".to_owned(),
            Tensor::from_vec(vec![5.0f32, 6.0], (1, 2), Device::Cpu),
        );
        second.insert(
            "lm_head.weight".to_owned(),
            Tensor::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], (2, 3), Device::Cpu),
        );
        write_shard(&dir, "model-00001-of-00002.safetensors", &first);
        write_shard(&dir, "model-00002-of-00002.safetensors", &second);
        write_index(
            &dir,
            &[
                ("model.embed_tokens.weight", "model-00001-of-00002.safetensors"),
                ("model.layers.0.self_attn.q_proj.weight", "model-00002-of-00002.safetensors"),
                ("lm_head.weight", "model-00002-of-00002.safetensors"),
            ],
        );

        // Act
        let loaded = load_sharded(&dir, Device::Cpu).unwrap();

        // Assert: deers names, literal values, and CPU placement.
        let names: Vec<&str> = loaded.keys().map(String::as_str).collect();
        assert_eq!(names, vec!["blocks.0.attn.q_proj.weight", "lm_head.weight", "wte.weight"]);
        assert_eq!(loaded["wte.weight"].to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded["blocks.0.attn.q_proj.weight"].to_vec::<f32>().unwrap(), vec![5.0, 6.0]);
        assert_eq!(
            loaded["lm_head.weight"].to_vec::<f32>().unwrap(),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        for tensor in loaded.values() {
            assert_eq!(tensor.device(), Device::Cpu);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_sharded_preserves_bf16_values_within_error_bound() {
        // Arrange: one BF16 shard with exact values plus values BF16 rounds.
        let dir = shard_dir("bf16");
        let exact = vec![bf16::from_f32(1.5), bf16::from_f32(-2.0), bf16::from_f32(3.140625)];
        let rounded = [0.1f32, std::f32::consts::PI, 1.0 / 3.0];
        let mut shard = BTreeMap::new();
        shard.insert(
            "model.embed_tokens.weight".to_owned(),
            Tensor::from_vec(exact.clone(), (3,), Device::Cpu),
        );
        shard.insert(
            "lm_head.weight".to_owned(),
            Tensor::from_vec(
                rounded.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>(),
                (3,),
                Device::Cpu,
            ),
        );
        write_shard(&dir, "model.safetensors", &shard);
        write_index(
            &dir,
            &[
                ("model.embed_tokens.weight", "model.safetensors"),
                ("lm_head.weight", "model.safetensors"),
            ],
        );

        // Act
        let loaded = load_sharded(&dir, Device::Cpu).unwrap();

        // Assert: exact values round-trip bit for bit, and rounded values
        // stay within the BF16 quantization bound of 0.01.
        assert_eq!(loaded["wte.weight"].dtype(), DType::BF16);
        assert_eq!(loaded["wte.weight"].to_vec::<bf16>().unwrap(), exact);
        let got: Vec<f32> =
            loaded["lm_head.weight"].to_vec::<bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
        let errors: Vec<f32> = got.iter().zip(rounded.iter()).map(|(g, w)| (g - w).abs()).collect();
        let max_error = errors.iter().copied().fold(0.0f32, f32::max);
        assert!(errors.iter().all(|&e| e <= 0.01));
        assert!(max_error > 0.0);
        assert!(max_error <= 0.01);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_sharded_without_index_reads_every_shard() {
        // Arrange: two shards with deers-native names and no index file.
        let dir = shard_dir("no-index");
        let mut first = BTreeMap::new();
        first.insert(
            "wte.weight".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cpu),
        );
        let mut second = BTreeMap::new();
        second.insert(
            "lm_head.weight".to_owned(),
            Tensor::from_vec(vec![3.0f32, 4.0], (2,), Device::Cpu),
        );
        write_shard(&dir, "a.safetensors", &first);
        write_shard(&dir, "b.safetensors", &second);

        // Act
        let loaded = load_sharded(&dir, Device::Cpu).unwrap();

        // Assert
        assert_eq!(loaded["wte.weight"].to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
        assert_eq!(loaded["lm_head.weight"].to_vec::<f32>().unwrap(), vec![3.0, 4.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_shard_files_prefers_index_over_strays() {
        // Arrange: an index naming one shard plus a stray file outside the map.
        let dir = shard_dir("index-wins");
        let mut tensors = BTreeMap::new();
        tensors.insert("wte.weight".to_owned(), Tensor::from_vec(vec![1.0f32], (1,), Device::Cpu));
        write_shard(&dir, "model-00001-of-00001.safetensors", &tensors);
        write_shard(&dir, "stray.safetensors", &tensors);
        write_index(&dir, &[("wte.weight", "model-00001-of-00001.safetensors")]);

        // Act
        let files = discover_shard_files(&dir).unwrap();

        // Assert: only the indexed file loads.
        assert_eq!(files, vec![dir.join("model-00001-of-00001.safetensors")]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_sharded_rejects_duplicate_tensor_across_shards() {
        // Arrange: two shards holding the same tensor name, no index.
        let dir = shard_dir("duplicate");
        let mut tensors = BTreeMap::new();
        tensors.insert("wte.weight".to_owned(), Tensor::from_vec(vec![1.0f32], (1,), Device::Cpu));
        write_shard(&dir, "a.safetensors", &tensors);
        write_shard(&dir, "b.safetensors", &tensors);

        // Act
        let error = load_sharded(&dir, Device::Cpu).unwrap_err().to_string();

        // Assert
        assert_eq!(
            error,
            format!(
                "duplicate tensor 'wte.weight' in shard files '{}' and '{}'",
                dir.join("a.safetensors").display(),
                dir.join("b.safetensors").display()
            )
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_sharded_rejects_empty_directory() {
        // Arrange: a directory with no shard files.
        let dir = shard_dir("empty");

        // Act
        let error = load_sharded(&dir, Device::Cpu).unwrap_err().to_string();

        // Assert
        assert_eq!(error, format!("no safetensors shard files in '{}'", dir.display()));

        let _ = fs::remove_dir_all(&dir);
    }
}
