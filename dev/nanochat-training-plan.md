# nanochat Training Parity Plan

Goal: train a GPT model in DEERS using the same data, optimizer, and evaluation
as nanochat. Each step is independently testable.

## Reference

- nanochat: `/Users/roberto/projects/nanochat`
- Data: ClimbMix-400B parquets (HuggingFace, `karpathy/climbmix-400b-shuffle`)
- Tokenizer: custom BPE, 32K vocab, tiktoken-compatible
- Optimizer: MuonAdamW (Muon for weight matrices, AdamW for everything else)
- Sequence length: 2048, batch size: 524,288 tokens
- LR schedule: linear warmup (40 steps) → constant → linear warmdown (65% of training)
- Loss metric: bits-per-byte (BPB), vocab-size-invariant
- Checkpoints: PyTorch `.pt` format

## Steps

### 1. AdamW optimizer
Smallest self-contained piece. Testable with the existing MNIST setup to verify
convergence vs SGD.
- First and second moment buffers per parameter
- Decoupled weight decay (not L2 reg)
- Bias correction for moments
- Per-parameter-group hyperparameters (lr, betas, eps, weight_decay)

### 2. Tokenizer
Needed for both data loading and generation. nanochat uses a custom BPE tokenizer
trained with tiktoken's split pattern. The `tiktoken-rs` crate can load it.
- Integrate tiktoken-rs (or minimal BPE loader)
- Load nanochat's trained tokenizer files
- Encode/decode text ↔ token ids

### 3. LR scheduler
Pairs with AdamW to complete the optimizer story.
- Linear warmup over N steps
- Constant phase
- Linear warmdown to `final_lr_frac` (0.05) over last 65% of training
- Sqrt batch-size scaling: `lr * sqrt(batch_size / B_ref)`

### 4. Data pipeline
Parquet → tokenize → pack → batch. Can test the training loop on small data first.
- Read parquet files (Arrow crate) — text column extraction
- Tokenize with BOS prepending
- Best-fit sequence packing (no padding, 100% utilization)
- Batch creation: inputs `[B, T]`, targets `[B, T]` shifted by one
- Train/val split (last parquet = validation)

### 5. Training loop
Wire everything together for a real training run.
- Gradient accumulation (`loss / accum_steps` before backward)
- `zero_grad` (set gradients to None between steps)
- Gradient clipping
- Step logging (loss, lr, tokens/sec)

### 6. BPB evaluation
Vocab-invariant eval metric used by nanochat for apples-to-apples comparison.
- Token-to-bytes mapping (each token → number of UTF-8 bytes)
- `bpb = (nats_loss / ln(2)) / avg_bytes_per_token`
- Evaluate on held-out validation split

### 7. Muon optimizer
Upgrades from plain AdamW to the full MuonAdamW hybrid nanochat uses.
- Newton-Schulz polar decomposition (5 iterations) for weight matrix updates
- Momentum with ramp-up (0.95 → 0.97 over 400 steps)
- Automatic param group routing: 2D weight matrices → Muon, everything else → AdamW
- Per-group LR/decay: embeddings, lm_head, scalars each get different hyperparams

### 8. Weight loading
Load pretrained nanochat checkpoints to validate forward pass correctness.
- Parse PyTorch `.pt` files (pickle + tensor storage) or convert via Python script
  to safetensors first
- Map nanochat weight names → DEERS model parameters
- dtype conversion (bf16/f16 ↔ f32)

## Separate track: nanochat model features

These improve final loss quality but don't block training. Add as needed:
- Group-Query Attention (GQA): `n_kv_head` < `n_head`
- Sliding window attention: per-layer window sizes from pattern string
- Value embeddings (ResFormer): learned per-layer value embeddings with gating
- Smear: previous-token mixing with learned gate
- Backout: mid-layer residual subtraction
- Logit soft-capping: `softcap * tanh(logits / softcap)`
