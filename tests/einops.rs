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
    assert_eq!(
        values(&grads_pattern.get(via_pattern.id()).unwrap()),
        vec![1.0; 8]
    );
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
    let loss_pattern = via_pattern
        .repeat("b t c -> b t (c h)", &[("h", 3)])
        .sum(vec![0, 1, 2], false);
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
