use deers::{Device, Tensor};

fn cpu(values: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec(values, shape, Device::Cpu)
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<f32>().unwrap()
}

fn shape_of(tensor: &Tensor) -> Vec<usize> {
    tensor.layout().shape().iter().copied().collect()
}

#[test]
fn split_names_the_hidden_product() {
    // Arrange
    let input = cpu((0..8).map(|v| v as f32).collect(), vec![2, 4]);

    // Act
    let split = input.rearrange("(b t) (h d) -> b t h d", &[("b", 1), ("t", 2), ("h", 2)]);

    // Assert
    assert_eq!(shape_of(&split), vec![1, 2, 2, 2]);
    assert_eq!(values(&split), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn transpose_reads_as_axis_names() {
    // Arrange
    let input = cpu((0..8).map(|v| v as f32).collect(), vec![1, 2, 2, 2]);

    // Act
    let transposed = input.rearrange("b t h d -> b h t d", &[]);

    // Assert
    assert_eq!(shape_of(&transposed), vec![1, 2, 2, 2]);
    assert_eq!(values(&transposed), vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn merge_inverts_the_split() {
    // Arrange
    let input = cpu((0..8).map(|v| v as f32).collect(), vec![1, 2, 2, 2]);

    // Act
    let merged = input.rearrange("b h t d -> b t (h d)", &[]);

    // Assert
    assert_eq!(shape_of(&merged), vec![1, 2, 4]);
    assert_eq!(values(&merged), vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn attention_sites_match_manual_sequences() {
    // Arrange
    let (b, t, c, h, d) = (2usize, 4, 12, 3, 4);
    let input = cpu((0..(b * t * c)).map(|v| v as f32 * 0.125).collect(), vec![b, t, c]);

    // Act
    let split = input.rearrange("b t (h d) -> b t h d", &[("h", h)]);
    let transposed = split.rearrange("b t h d -> b h t d", &[]);
    let merged = transposed.rearrange("b h t d -> b t (h d)", &[]);

    // Assert
    assert_eq!(values(&split), values(&input.reshape(vec![b, t, h, d])));
    assert_eq!(values(&transposed), values(&split.permute(vec![0, 2, 1, 3])));
    assert_eq!(
        values(&merged),
        values(&transposed.permute(vec![0, 2, 1, 3]).reshape(vec![b, t, c]))
    );
}

#[test]
fn mlp_flatten_roundtrips() {
    // Arrange
    let (b, t, c) = (2usize, 4, 12);
    let input = cpu((0..(b * t * c)).map(|v| v as f32).collect(), vec![b, t, c]);

    // Act
    let flat = input.rearrange("b t c -> (b t) c", &[]);
    let back = flat.rearrange("(b t) c -> b t c", &[("b", b), ("t", t)]);

    // Assert
    assert_eq!(shape_of(&flat), vec![b * t, c]);
    assert_eq!(values(&flat), values(&input.reshape(vec![b * t, c])));
    assert_eq!(values(&back), values(&input));
}

#[test]
fn anonymous_axes_pair_positionally() {
    // Arrange
    let input = cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]);

    // Act
    let flat = input.rearrange("_ _ c -> (_ _) c", &[]);

    // Assert
    assert_eq!(shape_of(&flat), vec![6, 1]);
    assert_eq!(values(&flat), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reduce_mean_matches_manual_mean() {
    // Arrange
    let input = cpu(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);

    // Act
    let reduced = input.reduce("b t c -> b c", "mean", &[]);

    // Assert
    assert_eq!(shape_of(&reduced), vec![1, 2]);
    assert_eq!(values(&reduced), vec![2.0, 3.0]);
}

#[test]
fn multi_axis_reduce_matches_manual_ops() {
    // Arrange
    let input = cpu((0..8).map(|v| v as f32).collect(), vec![1, 2, 2, 2]);

    // Act
    let summed = input.reduce("a b c d -> a d", "sum", &[]);
    let averaged = input.reduce("a b c d -> a d", "mean", &[]);
    let maxed = input.reduce("a b c d -> a d", "max", &[]);

    // Assert
    assert_eq!(values(&summed), values(&input.sum(vec![1, 2], false)));
    assert_eq!(values(&summed), vec![12.0, 16.0]);
    assert_eq!(values(&averaged), values(&input.mean(vec![1, 2], false)));
    assert_eq!(values(&averaged), vec![3.0, 4.0]);
    assert_eq!(values(&maxed), values(&input.max(vec![1, 2], false)));
    assert_eq!(values(&maxed), vec![6.0, 7.0]);
}

#[test]
fn multi_axis_reduce_gradients_match_manual_ops() {
    // Arrange
    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let via_pattern = cpu(data.clone(), vec![1, 2, 2, 2]).attach();
    let via_manual = cpu(data.clone(), vec![1, 2, 2, 2]).attach();

    // Act
    let loss_pattern = via_pattern.reduce("a b c d -> a d", "sum", &[]).sum(vec![0, 1], false);
    let loss_manual = via_manual.sum(vec![1, 2], false).sum(vec![0, 1], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        values(&grads_manual.get(via_manual.id()).unwrap())
    );
    assert_eq!(values(&grads_pattern.get(via_pattern.id()).unwrap()), vec![1.0; 8]);
}

#[test]
fn max_tie_shares_gradient_on_both_paths() {
    // Arrange
    let data = vec![1.0, 5.0, 5.0, 2.0, 0.0, 5.0, 3.0, 3.0];
    let via_pattern = cpu(data.clone(), vec![2, 4]).attach();
    let via_manual = cpu(data.clone(), vec![2, 4]).attach();

    // Act
    let reduced = via_pattern.reduce("b t -> b", "max", &[]);
    let loss_pattern = reduced.sum(vec![0], false);
    let loss_manual = via_manual.max(vec![1], false).sum(vec![0], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(values(&reduced), vec![5.0, 5.0]);
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        values(&grads_manual.get(via_manual.id()).unwrap())
    );
}

#[test]
fn rearrange_plus_reduce_gradients_match_manual_path() {
    // Arrange
    let (b, t, c, h) = (2usize, 4, 12, 3);
    let data: Vec<f32> = (0..(b * t * c)).map(|v| v as f32 * 0.125).collect();
    let via_pattern = cpu(data.clone(), vec![b, t, c]).attach();
    let via_manual = cpu(data.clone(), vec![b, t, c]).attach();

    // Act
    let loss_pattern = via_pattern
        .rearrange("b t (h d) -> b h t d", &[("h", h)])
        .reduce("b h t d -> b h d", "mean", &[])
        .sum(vec![0, 1, 2], false);
    let loss_manual = via_manual
        .reshape(vec![b, t, h, 4])
        .permute(vec![0, 2, 1, 3])
        .mean(vec![2], false)
        .sum(vec![0, 1, 2], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(values(&loss_pattern), vec![142.5]);
    assert_eq!(values(&loss_pattern), values(&loss_manual));
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        values(&grads_manual.get(via_manual.id()).unwrap())
    );
}

#[test]
fn repeat_tiles_a_whole_new_axis() {
    // Arrange
    let input = cpu(vec![7.0, 8.0], vec![1, 1, 2]);

    // Act
    let repeated = input.repeat("b t c -> b t c h", &[("h", 2)]);

    // Assert
    assert_eq!(shape_of(&repeated), vec![1, 1, 2, 2]);
    assert_eq!(values(&repeated), vec![7.0, 7.0, 8.0, 8.0]);
    assert_eq!(
        values(&repeated),
        values(&input.reshape(vec![1, 1, 2, 1]).broadcast(vec![1, 1, 2, 2]))
    );
}

#[test]
fn repeat_tiles_a_merged_new_axis() {
    // Arrange
    let input = cpu(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);

    // Act
    let repeated = input.repeat("b t c -> b t (c h)", &[("h", 3)]);

    // Assert
    assert_eq!(shape_of(&repeated), vec![1, 2, 6]);
    assert_eq!(values(&repeated), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]);
}

#[test]
fn repeat_gradients_match_manual_broadcast() {
    // Arrange
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let via_pattern = cpu(data.clone(), vec![1, 2, 2]).attach();
    let via_manual = cpu(data.clone(), vec![1, 2, 2]).attach();

    // Act
    let loss_pattern =
        via_pattern.repeat("b t c -> b t (c h)", &[("h", 3)]).sum(vec![0, 1, 2], false);
    let loss_manual = via_manual
        .reshape(vec![1, 2, 2, 1])
        .broadcast(vec![1, 2, 2, 3])
        .reshape(vec![1, 2, 6])
        .sum(vec![0, 1, 2], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(values(&loss_pattern), vec![30.0]);
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        values(&grads_manual.get(via_manual.id()).unwrap())
    );
    assert_eq!(values(&grads_pattern.get(via_pattern.id()).unwrap()), vec![3.0; 4]);
}

#[test]
#[should_panic(expected = "must contain exactly one '->'")]
fn missing_arrow_panics() {
    // Arrange
    let input = cpu(vec![1.0], vec![1]);

    // Act
    let _ = input.rearrange("b t c", &[]);
}

#[test]
#[should_panic(expected = "needs sizes for h, d")]
fn split_without_sizes_panics() {
    // Arrange
    let input = cpu(vec![0.0; 12], vec![1, 1, 12]);

    // Act
    let _ = input.rearrange("b t (h d) -> b t h d", &[]);
}

#[test]
#[should_panic(expected = "must preserve the axis multiset")]
fn rearrange_dropping_an_axis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.rearrange("b t c -> b t", &[]);
}

#[test]
#[should_panic(expected = "rhs axis 'h' not present on lhs")]
fn unknown_rhs_axis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.rearrange("b t c -> b c h", &[]);
}

#[test]
#[should_panic(expected = "not divisible by 5")]
fn indivisible_split_panics() {
    // Arrange
    let input = cpu(vec![0.0; 12], vec![1, 1, 12]);

    // Act
    let _ = input.rearrange("b t (h d) -> b t h d", &[("h", 5)]);
}

#[test]
#[should_panic(expected = "needs a size for new axis 'h'")]
fn repeat_without_size_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.repeat("b t c -> b t c h", &[]);
}

#[test]
#[should_panic(expected = "drops lhs axis 'c'")]
fn repeat_dropping_an_axis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.repeat("b t c -> b t h", &[("h", 2)]);
}

#[test]
#[should_panic(expected = "duplicate axis 'b' on rhs")]
fn duplicate_rhs_axis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.rearrange("b t c -> b b c", &[]);
}

#[test]
#[should_panic(expected = "duplicate axis 'b' on lhs")]
fn duplicate_lhs_axis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.rearrange("b b c -> b c", &[]);
}

#[test]
#[should_panic(expected = "drops no axis")]
fn reduce_dropping_nothing_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.reduce("b t c -> c t b", "sum", &[]);
}

#[test]
#[should_panic(expected = "adds no axis")]
fn repeat_adding_nothing_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.repeat("b t c -> c t b", &[("h", 2)]);
}

#[test]
#[should_panic(expected = "unsupported reduction 'median'")]
fn unknown_reduction_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.reduce("b t c -> b c", "median", &[]);
}

#[test]
#[should_panic(expected = "anonymous axis count differs")]
fn mismatched_anonymous_count_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.rearrange("_ t c -> (_ _) c", &[]);
}

#[test]
#[should_panic(expected = "one group per input dim")]
fn rank_mismatch_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![1, 2, 4]);

    // Act
    let _ = input.rearrange("b t -> b t", &[]);
}

#[test]
fn einsum_scores_match_manual_matmul() {
    // Arrange
    let q = cpu(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let k = cpu(vec![5.0, 6.0, 7.0, 8.0], vec![1, 1, 2, 2]);

    // Act
    let scores = Tensor::einsum("b h t d, b h s d -> b h t s", &q, &k);

    // Assert
    assert_eq!(shape_of(&scores), vec![1, 1, 2, 2]);
    assert_eq!(values(&scores), vec![17.0, 23.0, 39.0, 53.0]);
    assert_eq!(values(&scores), values(&q.matmul(&k.transpose(Some((2, 3))))));
}

#[test]
fn einsum_values_match_manual_matmul() {
    // Arrange
    let attn = cpu(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let v = cpu(vec![5.0, 6.0, 7.0, 8.0], vec![1, 1, 2, 2]);

    // Act
    let out = Tensor::einsum("b h t s, b h s d -> b h t d", &attn, &v);

    // Assert
    assert_eq!(shape_of(&out), vec![1, 1, 2, 2]);
    assert_eq!(values(&out), vec![19.0, 22.0, 43.0, 50.0]);
    assert_eq!(values(&out), values(&attn.matmul(&v)));
}

#[test]
fn einsum_projection_matches_manual_matmul() {
    // Arrange
    let x = cpu(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let w = cpu(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], vec![2, 3]);

    // Act
    let proj = Tensor::einsum("t c, c e -> t e", &x, &w);

    // Assert
    assert_eq!(shape_of(&proj), vec![2, 3]);
    assert_eq!(values(&proj), vec![1.0, 2.0, 3.0, 3.0, 4.0, 7.0]);
    assert_eq!(values(&proj), values(&x.matmul(&w)));
}

#[test]
fn einsum_output_order_permutes_the_product() {
    // Arrange
    let q = cpu(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let k = cpu(vec![5.0, 6.0, 7.0, 8.0], vec![1, 1, 2, 2]);

    // Act
    let swapped = Tensor::einsum("b h t d, b h s d -> b h s t", &q, &k);

    // Assert
    assert_eq!(shape_of(&swapped), vec![1, 1, 2, 2]);
    assert_eq!(values(&swapped), vec![17.0, 39.0, 23.0, 53.0]);
    assert_eq!(
        values(&swapped),
        values(&q.matmul(&k.transpose(Some((2, 3)))).permute(vec![0, 1, 3, 2]))
    );
}

#[test]
fn einsum_scores_gradients_match_manual_path() {
    // Arrange
    let q_data = vec![1.0, 2.0, 3.0, 4.0];
    let k_data = vec![5.0, 6.0, 7.0, 8.0];
    let q_pattern = cpu(q_data.clone(), vec![1, 1, 2, 2]).attach();
    let k_pattern = cpu(k_data.clone(), vec![1, 1, 2, 2]).attach();
    let q_manual = cpu(q_data, vec![1, 1, 2, 2]).attach();
    let k_manual = cpu(k_data, vec![1, 1, 2, 2]).attach();

    // Act
    let loss_pattern = Tensor::einsum("b h t d, b h s d -> b h t s", &q_pattern, &k_pattern)
        .sum(vec![0, 1, 2, 3], false);
    let loss_manual =
        q_manual.matmul(&k_manual.transpose(Some((2, 3)))).sum(vec![0, 1, 2, 3], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(
        values(&grads_pattern.get(q_pattern.id()).unwrap()),
        values(&grads_manual.get(q_manual.id()).unwrap())
    );
    assert_eq!(
        values(&grads_pattern.get(k_pattern.id()).unwrap()),
        values(&grads_manual.get(k_manual.id()).unwrap())
    );
    assert_eq!(values(&grads_pattern.get(q_pattern.id()).unwrap()), vec![12.0, 14.0, 12.0, 14.0]);
    assert_eq!(values(&grads_pattern.get(k_pattern.id()).unwrap()), vec![4.0, 6.0, 4.0, 6.0]);
}

#[test]
fn einsum_values_gradients_match_manual_path() {
    // Arrange
    let attn_data = vec![1.0, 2.0, 3.0, 4.0];
    let v_data = vec![5.0, 6.0, 7.0, 8.0];
    let attn_pattern = cpu(attn_data.clone(), vec![1, 1, 2, 2]).attach();
    let v_pattern = cpu(v_data.clone(), vec![1, 1, 2, 2]).attach();
    let attn_manual = cpu(attn_data, vec![1, 1, 2, 2]).attach();
    let v_manual = cpu(v_data, vec![1, 1, 2, 2]).attach();

    // Act
    let loss_pattern = Tensor::einsum("b h t s, b h s d -> b h t d", &attn_pattern, &v_pattern)
        .sum(vec![0, 1, 2, 3], false);
    let loss_manual = attn_manual.matmul(&v_manual).sum(vec![0, 1, 2, 3], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(
        values(&grads_pattern.get(attn_pattern.id()).unwrap()),
        values(&grads_manual.get(attn_manual.id()).unwrap())
    );
    assert_eq!(
        values(&grads_pattern.get(v_pattern.id()).unwrap()),
        values(&grads_manual.get(v_manual.id()).unwrap())
    );
}

#[test]
fn einsum_projection_gradients_match_manual_path() {
    // Arrange
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let w_data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let x_pattern = cpu(x_data.clone(), vec![2, 2]).attach();
    let w_pattern = cpu(w_data.clone(), vec![2, 3]).attach();
    let x_manual = cpu(x_data, vec![2, 2]).attach();
    let w_manual = cpu(w_data, vec![2, 3]).attach();

    // Act
    let loss_pattern =
        Tensor::einsum("t c, c e -> t e", &x_pattern, &w_pattern).sum(vec![0, 1], false);
    let loss_manual = x_manual.matmul(&w_manual).sum(vec![0, 1], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(
        values(&grads_pattern.get(x_pattern.id()).unwrap()),
        values(&grads_manual.get(x_manual.id()).unwrap())
    );
    assert_eq!(
        values(&grads_pattern.get(w_pattern.id()).unwrap()),
        values(&grads_manual.get(w_manual.id()).unwrap())
    );
}

#[test]
#[should_panic(expected = "output label 'x' is not present in either input")]
fn einsum_unknown_output_label_panics() {
    // Arrange
    let q = cpu(vec![0.0; 8], vec![1, 1, 2, 2]);
    let k = cpu(vec![0.0; 8], vec![1, 1, 2, 2]);

    // Act
    let _ = Tensor::einsum("b h t d, b h s d -> b h t x", &q, &k);
}

#[test]
#[should_panic(expected = "left side has 4 labels but the first input is 3-d")]
fn einsum_rank_mismatch_panics() {
    // Arrange
    let q = cpu(vec![0.0; 8], vec![1, 2, 4]);
    let k = cpu(vec![0.0; 8], vec![1, 1, 2, 2]);

    // Act
    let _ = Tensor::einsum("b h t d, b h s d -> b h t s", &q, &k);
}

#[test]
#[should_panic(expected = "contracted axis 'd' has size 2 in the first input but 3 in the second")]
fn einsum_contract_size_mismatch_panics() {
    // Arrange
    let q = cpu(vec![0.0; 8], vec![1, 1, 2, 2]);
    let k = cpu(vec![0.0; 12], vec![1, 1, 2, 3]);

    // Act
    let _ = Tensor::einsum("b h t d, b h s d -> b h t s", &q, &k);
}

#[test]
#[should_panic(expected = "batch axis 'b' has size 1 in the first input but 2 in the second")]
fn einsum_batch_size_mismatch_panics() {
    // Arrange
    let q = cpu(vec![0.0; 8], vec![1, 1, 2, 2]);
    let k = cpu(vec![0.0; 16], vec![2, 1, 2, 2]);

    // Act
    let _ = Tensor::einsum("b h t d, b h s d -> b h t s", &q, &k);
}

#[test]
#[should_panic(expected = "no contracted axis")]
fn einsum_without_contraction_panics() {
    // Arrange
    let a = cpu(vec![0.0; 4], vec![2, 2]);
    let b = cpu(vec![0.0; 4], vec![2, 2]);

    // Act
    let _ = Tensor::einsum("b t, b t -> b t", &a, &b);
}

#[test]
#[should_panic(expected = "axes c, d are all contracted")]
fn einsum_multi_axis_contraction_panics() {
    // Arrange
    let a = cpu(vec![0.0; 8], vec![2, 2, 2]);
    let b = cpu(vec![0.0; 8], vec![2, 2, 2]);

    // Act
    let _ = Tensor::einsum("b c d, b c d -> b", &a, &b);
}

#[test]
#[should_panic(expected = "input label 'e' is missing from the output")]
fn einsum_dropped_input_label_panics() {
    // Arrange
    let a = cpu(vec![0.0; 8], vec![2, 2, 2]);
    let b = cpu(vec![0.0; 8], vec![2, 2, 2]);

    // Act
    let _ = Tensor::einsum("b t c, b c e -> b t", &a, &b);
}

#[test]
#[should_panic(expected = "left input keeps axes b, t")]
fn einsum_two_kept_axes_panics() {
    // Arrange
    let a = cpu(vec![0.0; 24], vec![2, 3, 4]);
    let b = cpu(vec![0.0; 20], vec![4, 5]);

    // Act
    let _ = Tensor::einsum("b t c, c e -> b t e", &a, &b);
}

#[test]
#[should_panic(expected = "duplicate label 't' in input side 1")]
fn einsum_duplicate_label_panics() {
    // Arrange
    let q = cpu(vec![0.0; 8], vec![1, 1, 2, 2]);
    let k = cpu(vec![0.0; 8], vec![1, 1, 2, 2]);

    // Act
    let _ = Tensor::einsum("b h t t, b h s d -> b h t s", &q, &k);
}

#[test]
#[should_panic(expected = "must contain exactly one '->'")]
fn einsum_missing_arrow_panics() {
    // Arrange
    let a = cpu(vec![0.0; 4], vec![2, 2]);
    let b = cpu(vec![0.0; 4], vec![2, 2]);

    // Act
    let _ = Tensor::einsum("b t, b t", &a, &b);
}

#[test]
#[should_panic(expected = "must contain exactly two inputs separated by ','")]
fn einsum_single_input_panics() {
    // Arrange
    let a = cpu(vec![0.0; 4], vec![2, 2]);
    let b = cpu(vec![0.0; 4], vec![2, 2]);

    // Act
    let _ = Tensor::einsum("b t -> b t", &a, &b);
}

#[test]
fn ellipsis_splits_the_last_dim_at_rank_three() {
    // Arrange
    let input = cpu((0..16).map(|v| v as f32).collect(), vec![2, 2, 4]);

    // Act
    let split = input.rearrange("... (h d) -> ... h d", &[("h", 2)]);

    // Assert
    assert_eq!(shape_of(&split), vec![2, 2, 2, 2]);
    assert_eq!(values(&split), values(&input.reshape(vec![2, 2, 2, 2])));
    assert_eq!(values(&split), (0..16).map(|v| v as f32).collect::<Vec<_>>());
}

#[test]
fn ellipsis_binds_zero_dims_as_identity() {
    // Arrange
    let input = cpu(vec![1.0, 2.0, 3.0], vec![3]);

    // Act
    let same = input.rearrange("... c -> ... c", &[]);

    // Assert
    assert_eq!(shape_of(&same), vec![3]);
    assert_eq!(values(&same), vec![1.0, 2.0, 3.0]);
}

#[test]
fn rope_form_moves_a_split_block_past_the_batch() {
    // Arrange
    let input = cpu((0..8).map(|v| v as f32).collect(), vec![1, 2, 4]);

    // Act
    let rotated = input.rearrange("... (half_d xy) -> xy ... half_d", &[("half_d", 2)]);

    // Assert
    assert_eq!(shape_of(&rotated), vec![2, 1, 2, 2]);
    assert_eq!(values(&rotated), vec![0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]);
}

#[test]
fn rope_form_gradients_match_the_manual_path() {
    // Arrange
    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let via_pattern = cpu(data.clone(), vec![1, 2, 4]).attach();
    let via_manual = cpu(data.clone(), vec![1, 2, 4]).attach();

    // Act
    let loss_pattern = via_pattern
        .rearrange("... (half_d xy) -> xy ... half_d", &[("half_d", 2)])
        .sum(vec![0, 1, 2, 3], false);
    let loss_manual = via_manual
        .reshape(vec![1, 2, 2, 2])
        .permute(vec![3, 0, 1, 2])
        .sum(vec![0, 1, 2, 3], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(values(&loss_pattern), values(&loss_manual));
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        values(&grads_manual.get(via_manual.id()).unwrap())
    );
    assert_eq!(values(&grads_pattern.get(via_pattern.id()).unwrap()), vec![1.0; 8]);
}

#[test]
fn flatten_form_merges_the_batch_into_one_dim() {
    // Arrange
    let input = cpu((0..24).map(|v| v as f32).collect(), vec![2, 3, 4]);

    // Act
    let flat = input.rearrange("... d -> (...) d", &[]);

    // Assert
    assert_eq!(shape_of(&flat), vec![6, 4]);
    assert_eq!(values(&flat), values(&input.reshape(vec![6, 4])));
    assert_eq!(values(&flat), (0..24).map(|v| v as f32).collect::<Vec<_>>());
}

#[test]
fn flatten_form_gradients_match_reshape() {
    // Arrange
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let via_pattern = cpu(data.clone(), vec![2, 3, 4]).attach();
    let via_manual = cpu(data.clone(), vec![2, 3, 4]).attach();

    // Act
    let loss_pattern = via_pattern.rearrange("... d -> (...) d", &[]).sum(vec![0, 1], false);
    let loss_manual = via_manual.reshape(vec![6, 4]).sum(vec![0, 1], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(values(&loss_pattern), vec![276.0]);
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        values(&grads_manual.get(via_manual.id()).unwrap())
    );
    assert_eq!(values(&grads_pattern.get(via_pattern.id()).unwrap()), vec![1.0; 24]);
}

#[test]
fn anonymous_axes_pair_across_an_ellipsis() {
    // Arrange
    let input = cpu((0..12).map(|v| v as f32).collect(), vec![2, 2, 3]);

    // Act
    let swapped = input.rearrange("... _ c -> ... c _", &[]);

    // Assert
    assert_eq!(shape_of(&swapped), vec![2, 3, 2]);
    assert_eq!(values(&swapped), values(&input.permute(vec![0, 2, 1])));
    assert_eq!(
        values(&swapped),
        vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 9.0, 7.0, 10.0, 8.0, 11.0]
    );
}

#[test]
fn ellipsis_reduce_sums_inside_the_batch() {
    // Arrange
    let input = cpu((0..24).map(|v| v as f32).collect(), vec![2, 3, 4]);

    // Act
    let reduced = input.reduce("... t c -> ... c", "sum", &[]);

    // Assert
    assert_eq!(shape_of(&reduced), vec![2, 4]);
    assert_eq!(values(&reduced), vec![12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0]);
}

#[test]
fn ellipsis_reduce_binds_zero_batch_dims() {
    // Arrange
    let input = cpu((0..12).map(|v| v as f32).collect(), vec![3, 4]);

    // Act
    let reduced = input.reduce("... t c -> ... c", "sum", &[]);

    // Assert
    assert_eq!(shape_of(&reduced), vec![4]);
    assert_eq!(values(&reduced), vec![12.0, 15.0, 18.0, 21.0]);
}

#[test]
fn ellipsis_reduce_gradients_match_manual_mean() {
    // Arrange
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let via_pattern = cpu(data.clone(), vec![2, 3, 4]).attach();
    let via_manual = cpu(data.clone(), vec![2, 3, 4]).attach();

    // Act
    let loss_pattern =
        via_pattern.reduce("... t c -> ... c", "mean", &[]).sum(vec![0, 1], false);
    let loss_manual = via_manual.mean(vec![1], false).sum(vec![0, 1], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(values(&loss_pattern), values(&loss_manual));
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        values(&grads_manual.get(via_manual.id()).unwrap())
    );
}

#[test]
fn ellipsis_repeat_tiles_inside_the_batch() {
    // Arrange
    let input = cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    // Act
    let repeated = input.repeat("... c -> ... c h", &[("h", 2)]);

    // Assert
    assert_eq!(shape_of(&repeated), vec![2, 3, 2]);
    assert_eq!(
        values(&repeated),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
    );
}

#[test]
fn ellipsis_repeat_gradients_match_manual_broadcast() {
    // Arrange
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let via_pattern = cpu(data.clone(), vec![2, 3]).attach();
    let via_manual = cpu(data.clone(), vec![2, 3]).attach();

    // Act
    let loss_pattern =
        via_pattern.repeat("... c -> ... c h", &[("h", 2)]).sum(vec![0, 1, 2], false);
    let loss_manual = via_manual
        .reshape(vec![2, 3, 1])
        .broadcast(vec![2, 3, 2])
        .sum(vec![0, 1, 2], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(values(&loss_pattern), vec![42.0]);
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        values(&grads_manual.get(via_manual.id()).unwrap())
    );
    assert_eq!(values(&grads_pattern.get(via_pattern.id()).unwrap()), vec![2.0; 6]);
}

#[test]
fn ellipsis_attention_scores_match_named_einsum() {
    // Arrange
    let q = cpu(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let k = cpu(vec![5.0, 6.0, 7.0, 8.0], vec![1, 1, 2, 2]);

    // Act
    let scores = Tensor::einsum("... q d, ... k d -> ... q k", &q, &k);

    // Assert
    assert_eq!(shape_of(&scores), vec![1, 1, 2, 2]);
    assert_eq!(values(&scores), vec![17.0, 23.0, 39.0, 53.0]);
}

#[test]
fn ellipsis_attention_binds_zero_batch_dims() {
    // Arrange
    let q = cpu(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let k = cpu(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // Act
    let scores = Tensor::einsum("... q d, ... k d -> ... q k", &q, &k);

    // Assert
    assert_eq!(shape_of(&scores), vec![2, 2]);
    assert_eq!(values(&scores), vec![17.0, 23.0, 39.0, 53.0]);
}

#[test]
fn ellipsis_attention_gradients_match_manual_path() {
    // Arrange
    let q_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let k_data = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let q_pattern = cpu(q_data.clone(), vec![2, 2, 2]).attach();
    let k_pattern = cpu(k_data.clone(), vec![2, 2, 2]).attach();
    let q_manual = cpu(q_data, vec![2, 2, 2]).attach();
    let k_manual = cpu(k_data, vec![2, 2, 2]).attach();

    // Act
    let loss_pattern = Tensor::einsum("... q d, ... k d -> ... q k", &q_pattern, &k_pattern)
        .sum(vec![0, 1, 2], false);
    let loss_manual =
        q_manual.matmul(&k_manual.transpose(Some((1, 2)))).sum(vec![0, 1, 2], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(
        values(&grads_pattern.get(q_pattern.id()).unwrap()),
        values(&grads_manual.get(q_manual.id()).unwrap())
    );
    assert_eq!(
        values(&grads_pattern.get(k_pattern.id()).unwrap()),
        values(&grads_manual.get(k_manual.id()).unwrap())
    );
}

#[test]
fn ellipsis_linear_broadcasts_the_weight_over_batch() {
    // Arrange
    let x = cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let w = cpu(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], vec![2, 3]);

    // Act
    let y = Tensor::einsum("... d_in, d_out d_in -> ... d_out", &x, &w);

    // Assert
    assert_eq!(shape_of(&y), vec![2, 2]);
    assert_eq!(values(&y), vec![4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn ellipsis_linear_keeps_extra_batch_dims() {
    // Arrange
    let x = cpu((0..12).map(|v| v as f32).collect(), vec![2, 2, 3]);
    let w = cpu(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], vec![2, 3]);

    // Act
    let y = Tensor::einsum("... d_in, d_out d_in -> ... d_out", &x, &w);

    // Assert
    assert_eq!(shape_of(&y), vec![2, 2, 2]);
    assert_eq!(values(&y), vec![2.0, 3.0, 8.0, 9.0, 14.0, 15.0, 20.0, 21.0]);
}

#[test]
fn ellipsis_linear_gradients_match_manual_matmul() {
    // Arrange
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let x_pattern = cpu(x_data.clone(), vec![2, 3]).attach();
    let w_pattern = cpu(w_data.clone(), vec![2, 3]).attach();
    let x_manual = cpu(x_data, vec![2, 3]).attach();
    let w_manual = cpu(w_data, vec![2, 3]).attach();

    // Act
    let loss_pattern =
        Tensor::einsum("... d_in, d_out d_in -> ... d_out", &x_pattern, &w_pattern)
            .sum(vec![0, 1], false);
    let loss_manual = x_manual.matmul(&w_manual.transpose(None)).sum(vec![0, 1], false);
    let grads_pattern = loss_pattern.backward().unwrap();
    let grads_manual = loss_manual.backward().unwrap();

    // Assert
    assert_eq!(
        values(&grads_pattern.get(x_pattern.id()).unwrap()),
        values(&grads_manual.get(x_manual.id()).unwrap())
    );
    assert_eq!(
        values(&grads_pattern.get(w_pattern.id()).unwrap()),
        values(&grads_manual.get(w_manual.id()).unwrap())
    );
}

#[test]
#[should_panic(expected = "both sides must carry '...' together")]
fn ellipsis_on_lhs_only_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![2, 4]);

    // Act
    let _ = input.rearrange("... b c -> b c", &[]);
}

#[test]
#[should_panic(expected = "both sides must carry '...' together")]
fn ellipsis_on_rhs_only_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![2, 4]);

    // Act
    let _ = input.rearrange("b c -> ... b c", &[]);
}

#[test]
#[should_panic(expected = "at most one ellipsis per side")]
fn two_ellipses_on_one_side_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![2, 4]);

    // Act
    let _ = input.rearrange("... ... b -> ... b", &[]);
}

#[test]
#[should_panic(expected = "only supported on the rhs")]
fn flatten_on_lhs_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![2, 4]);

    // Act
    let _ = input.rearrange("(...) d -> ... d", &[]);
}

#[test]
#[should_panic(expected = "must match at least one dim to flatten")]
fn flatten_binds_zero_dims_panics() {
    // Arrange
    let input = cpu(vec![0.0; 4], vec![4]);

    // Act
    let _ = input.rearrange("... d -> (...) d", &[]);
}

#[test]
#[should_panic(expected = "binds zero or more dims")]
fn ellipsis_with_too_many_named_groups_panics() {
    // Arrange
    let input = cpu(vec![0.0; 2], vec![2]);

    // Act
    let _ = input.rearrange("... a b -> ... a b", &[]);
}

#[test]
#[should_panic(expected = "batch dims are preserved in the output")]
fn einsum_ellipsis_missing_from_output_panics() {
    // Arrange
    let q = cpu(vec![0.0; 8], vec![2, 2, 2]);
    let k = cpu(vec![0.0; 8], vec![2, 2, 2]);

    // Act
    let _ = Tensor::einsum("... q d, ... k d -> q k", &q, &k);
}

#[test]
#[should_panic(expected = "batch rank must match")]
fn einsum_ellipsis_rank_mismatch_panics() {
    // Arrange
    let q = cpu(vec![0.0; 16], vec![2, 2, 2, 2]);
    let k = cpu(vec![0.0; 8], vec![2, 2, 2]);

    // Act
    let _ = Tensor::einsum("... q d, ... k d -> ... q k", &q, &k);
}

#[test]
#[should_panic(expected = "batch '...' dim 0 has size 2 in the first input but 3 in the second")]
fn einsum_ellipsis_size_mismatch_panics() {
    // Arrange
    let q = cpu(vec![0.0; 8], vec![2, 2, 2]);
    let k = cpu(vec![0.0; 12], vec![3, 2, 2]);

    // Act
    let _ = Tensor::einsum("... q d, ... k d -> ... q k", &q, &k);
}

#[test]
#[should_panic(expected = "must lead")]
fn einsum_ellipsis_not_leading_panics() {
    // Arrange
    let q = cpu(vec![0.0; 8], vec![2, 2, 2]);
    let k = cpu(vec![0.0; 8], vec![2, 2, 2]);

    // Act
    let _ = Tensor::einsum("q ... d, ... k d -> ... q k", &q, &k);
}

#[test]
#[should_panic(expected = "neither input keeps an axis")]
fn einsum_vector_dot_without_kept_axis_panics() {
    // Arrange
    let a = cpu(vec![1.0, 2.0], vec![2]);
    let b = cpu(vec![3.0, 4.0], vec![2]);

    // Act
    let _ = Tensor::einsum("... d, ... d -> ...", &a, &b);
}

#[test]
#[should_panic(expected = "duplicate axis 'b' on lhs")]
fn duplicate_axis_with_ellipsis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 8], vec![2, 2, 2]);

    // Act
    let _ = input.rearrange("... b b -> ... b", &[]);
}

#[test]
#[should_panic(expected = "must preserve the axis multiset")]
fn rearrange_drop_with_ellipsis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 24], vec![2, 3, 4]);

    // Act
    let _ = input.rearrange("... b c -> ... b", &[]);
}

#[test]
#[should_panic(expected = "drops no axis")]
fn reduce_no_drop_with_ellipsis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 24], vec![2, 3, 4]);

    // Act
    let _ = input.reduce("... b c -> ... c b", "sum", &[]);
}

#[test]
#[should_panic(expected = "adds no axis")]
fn repeat_no_add_with_ellipsis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 24], vec![2, 3, 4]);

    // Act
    let _ = input.repeat("... b c -> ... c b", &[("h", 2)]);
}

#[test]
#[should_panic(expected = "drops lhs axis 'c'")]
fn repeat_drop_with_ellipsis_panics() {
    // Arrange
    let input = cpu(vec![0.0; 24], vec![2, 3, 4]);

    // Act
    let _ = input.repeat("... b c -> ... b h", &[("h", 2)]);
}
