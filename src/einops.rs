//! Einops-style shape expressions over [`Tensor`](crate::Tensor).
//!
//! `rearrange`, `reduce`, and `repeat` rewrite tensor shapes through named
//! axes instead of raw index vectors. Each method lowers to the existing
//! primitives (`reshape`, `permute`, `broadcast`, `sum`, `mean`, `max`), so
//! autograd flows through unchanged and no new operator exists.
//!
//! A pattern is `lhs -> rhs`. Each side holds space separated groups. A group
//! is one axis name, one `_` placeholder, or a parenthesised merge of two or
//! more names such as `(h d)`. Names pair across the arrow. Each `_` pairs
//! positionally with the `_` at the same flat position on the other side.
//! Sizes for split or new axes arrive in `sizes`. All other sizes come from
//! the input shape.
//!
//! Bad patterns panic. A bad pattern is a programmer error in the same sense
//! as a wrong index vector, so each panic names the pattern, the axis, and
//! the expectation.

use std::collections::HashMap;

use crate::Tensor;
use crate::layout::Layout;

/// One flat axis inside a pattern group.
#[derive(Debug, Clone, PartialEq)]
enum Axis {
    /// A user written name such as `b` or `head`.
    Named(String),
    /// An anonymous `_`, identified by its flat position on its own side.
    Anon(usize),
}

/// One side of a `lhs -> rhs` pattern: named groups around one ellipsis.
///
/// `...` binds zero or more whole leading dims in input order. A bare `...`
/// passes them through. `(...)` merges them into one dim and is only valid
/// on the rhs, since the split rank has no names for `sizes` to fill.
#[derive(Debug, Clone)]
struct Side {
    pre: Vec<Vec<Axis>>,
    ellipsis: Option<EllipsisKind>,
    post: Vec<Vec<Axis>>,
}

/// The two ellipsis forms: passthrough `...` or merged `(...)`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum EllipsisKind {
    Pass,
    Flatten,
}

/// A parsed `lhs -> rhs` pattern plus the source text for panic messages.
#[derive(Debug, Clone)]
struct Pattern {
    lhs: Side,
    rhs: Side,
    source: String,
}

impl Axis {
    /// Resolves to the shared key used in the sizes table.
    fn key(&self) -> String {
        match self {
            Axis::Named(name) => name.clone(),
            // `#` cannot appear in a parsed name, so generated keys never
            // collide with user axes.
            Axis::Anon(position) => format!("_anon#{position}"),
        }
    }
}

/// Parses one side of a pattern into groups around one ellipsis.
fn parse_side(source: &str, pattern: &str) -> Side {
    let mut pre: Vec<Vec<Axis>> = Vec::new();
    let mut post: Vec<Vec<Axis>> = Vec::new();
    let mut ellipsis: Option<EllipsisKind> = None;
    let mut anon = 0;
    let mut token = String::new();
    let mut chars = source.chars();
    let push_group = |group: Vec<Axis>,
                      pre: &mut Vec<Vec<Axis>>,
                      post: &mut Vec<Vec<Axis>>,
                      ellipsis: &Option<EllipsisKind>| {
        if ellipsis.is_some() { post.push(group) } else { pre.push(group) }
    };
    let flush = |token: &mut String,
                 pre: &mut Vec<Vec<Axis>>,
                 post: &mut Vec<Vec<Axis>>,
                 ellipsis: &mut Option<EllipsisKind>,
                 anon: &mut usize| {
        let word = token.trim().to_string();
        token.clear();
        if word.is_empty() {
            return;
        }
        if word == "..." {
            if ellipsis.is_some() {
                panic!(
                    "einops pattern '{pattern}': multiple '...' on one side; at most one ellipsis per side"
                );
            }
            *ellipsis = Some(EllipsisKind::Pass);
            return;
        }
        let group = if word == "_" {
            let axis = Axis::Anon(*anon);
            *anon += 1;
            vec![axis]
        } else {
            vec![Axis::Named(check_name(&word, pattern))]
        };
        push_group(group, pre, post, ellipsis);
    };
    while let Some(next) = chars.next() {
        match next {
            '(' => {
                flush(&mut token, &mut pre, &mut post, &mut ellipsis, &mut anon);
                let mut inner = String::new();
                let mut closed = false;
                for inner_next in chars.by_ref() {
                    if inner_next == ')' {
                        closed = true;
                        break;
                    }
                    inner.push(inner_next);
                }
                if !closed {
                    panic!("einops pattern '{pattern}': '(' has no closing ')' in '{source}'");
                }
                let words: Vec<&str> = inner.split_whitespace().collect();
                if words == ["..."] {
                    if ellipsis.is_some() {
                        panic!(
                            "einops pattern '{pattern}': multiple '...' on one side; at most one ellipsis per side"
                        );
                    }
                    ellipsis = Some(EllipsisKind::Flatten);
                    continue;
                }
                let mut group = Vec::new();
                for word in words {
                    if word == "..." {
                        panic!(
                            "einops pattern '{pattern}': '...' must be a whole group or '(...)', got '({inner})'"
                        );
                    }
                    if word == "_" {
                        group.push(Axis::Anon(anon));
                        anon += 1;
                    } else {
                        group.push(Axis::Named(check_name(word, pattern)));
                    }
                }
                if group.is_empty() {
                    panic!("einops pattern '{pattern}': empty parentheses in '{source}'");
                }
                push_group(group, &mut pre, &mut post, &ellipsis);
            }
            current if current.is_whitespace() => {
                flush(&mut token, &mut pre, &mut post, &mut ellipsis, &mut anon)
            }
            _ => token.push(next),
        }
    }
    flush(&mut token, &mut pre, &mut post, &mut ellipsis, &mut anon);
    if pre.is_empty() && post.is_empty() && ellipsis.is_none() {
        panic!("einops pattern '{pattern}': empty pattern side '{source}'");
    }
    Side { pre, ellipsis, post }
}

/// Accepts plain axis names: ascii alphanumeric plus `_`, never digit first.
fn valid_name(word: &str) -> bool {
    !word.is_empty()
        && word.chars().all(|current| current.is_ascii_alphanumeric() || current == '_')
        && !word.chars().next().is_some_and(|current| current.is_ascii_digit())
}

/// Rejects tokens that are not plain axis names.
fn check_name(word: &str, pattern: &str) -> String {
    if !valid_name(word) {
        panic!("einops pattern '{pattern}': '{word}' is not a valid axis name");
    }
    word.to_string()
}

/// Parses `lhs -> rhs` and pairs anonymous axes positionally across the arrow.
///
/// Both sides number `_` from zero in encounter order, so the nth `_` on
/// each side shares one key and pairing falls out of key equality.
fn parse_pattern(pattern: &str) -> Pattern {
    let parts: Vec<&str> = pattern.split("->").collect();
    if parts.len() != 2 {
        panic!("einops pattern '{pattern}': pattern must contain exactly one '->'");
    }
    let lhs = parse_side(parts[0], pattern);
    let rhs = parse_side(parts[1], pattern);
    let lhs_anons = anon_count(&lhs);
    let rhs_anons = anon_count(&rhs);
    if lhs_anons != rhs_anons {
        panic!(
            "einops pattern '{pattern}': anonymous axis count differs with {lhs_anons} on lhs vs {rhs_anons} on rhs; '_' pairs positionally"
        );
    }
    Pattern { lhs, rhs, source: pattern.to_string() }
}

/// Flat axis keys in group order, skipping the ellipsis.
fn flat_keys(side: &Side) -> Vec<String> {
    flat_group_keys(&side.pre)
        .into_iter()
        .chain(flat_group_keys(&side.post))
        .collect()
}

/// Flat keys of a group list.
fn flat_group_keys(groups: &[Vec<Axis>]) -> Vec<String> {
    groups.iter().flatten().map(Axis::key).collect()
}

/// Counts explicit `_` axes on one side; ellipsis dims never enter the count.
fn anon_count(side: &Side) -> usize {
    side.pre
        .iter()
        .chain(side.post.iter())
        .flatten()
        .filter(|axis| matches!(axis, Axis::Anon(_)))
        .count()
}

/// Named groups on one side in order.
fn explicit_groups(side: &Side) -> Vec<&Vec<Axis>> {
    side.pre.iter().chain(side.post.iter()).collect()
}

/// Synthetic keys for the N dims bound by `...`, in input order.
fn ellipsis_keys(rank: usize) -> Vec<String> {
    (0..rank).map(|index| format!("_ellipsis#{index}")).collect()
}

/// Binds `...` on both sides against the input rank.
///
/// The lhs fixes N as input dims minus named lhs groups. Both sides must
/// carry `...` together, the lhs keeps the passthrough form, and a rhs
/// `(...)` needs at least one bound dim to merge.
fn bind_ellipsis(pattern: &Pattern, input_shape: &[usize]) -> usize {
    let source = pattern.source.as_str();
    match (pattern.lhs.ellipsis, pattern.rhs.ellipsis) {
        (None, None) => {
            if explicit_groups(&pattern.lhs).len() != input_shape.len() {
                panic!(
                    "einops pattern '{source}': lhs has {} groups but input is {}-d (shape {input_shape:?}); one group per input dim",
                    explicit_groups(&pattern.lhs).len(),
                    input_shape.len()
                );
            }
            0
        }
        (Some(_), None) => panic!(
            "einops pattern '{source}': lhs has '...' but rhs does not; both sides must carry '...' together"
        ),
        (None, Some(_)) => panic!(
            "einops pattern '{source}': rhs has '...' but lhs does not; both sides must carry '...' together"
        ),
        (Some(EllipsisKind::Flatten), Some(_)) => panic!(
            "einops pattern '{source}': '(...)' is only supported on the rhs; lhs uses '...'"
        ),
        (Some(EllipsisKind::Pass), Some(EllipsisKind::Flatten)) => {
            let rank = bind_rank(&pattern.lhs, input_shape, source);
            if rank == 0 {
                panic!(
                    "einops pattern '{source}': '(...)' binds zero dims; '...' must match at least one dim to flatten"
                );
            }
            rank
        }
        (Some(EllipsisKind::Pass), Some(EllipsisKind::Pass)) => {
            bind_rank(&pattern.lhs, input_shape, source)
        }
    }
}

/// Ranks the lhs ellipsis as input dims minus named lhs groups.
fn bind_rank(lhs: &Side, input_shape: &[usize], source: &str) -> usize {
    let named = explicit_groups(lhs).len();
    if named > input_shape.len() {
        panic!(
            "einops pattern '{source}': lhs has {named} named groups plus '...' but input is {}-d (shape {input_shape:?}); '...' binds zero or more dims",
            input_shape.len()
        );
    }
    input_shape.len() - named
}

/// Rejects duplicated names within one side, which would alias one position.
fn check_unique(side: &Side, side_name: &str, pattern: &str) {
    let mut seen = Vec::new();
    for key in flat_keys(side) {
        if seen.contains(&key) {
            panic!(
                "einops pattern '{pattern}': duplicate axis '{key}' on {side_name}; axis names must be unique per side"
            );
        }
        seen.push(key);
    }
}

/// Resolves every flat lhs axis to a size from the input shape or `sizes`.
///
/// Split groups need sizes for all but at most one member. The missing member
/// divides the input dim with a divisibility check.
fn resolve_sizes(
    pattern: &Pattern,
    input_shape: &[usize],
    hints: &[(&str, usize)],
    ellipsis_rank: usize,
) -> HashMap<String, usize> {
    let source = pattern.source.as_str();
    let lhs_groups = explicit_groups(&pattern.lhs);
    let dims: Vec<usize> = if pattern.lhs.ellipsis.is_some() {
        let pre = pattern.lhs.pre.len();
        lhs_groups
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if index < pre { input_shape[index] } else { input_shape[index + ellipsis_rank] }
            })
            .collect()
    } else {
        input_shape.to_vec()
    };
    let mut sizes: HashMap<String, usize> = HashMap::new();
    for (name, size) in hints {
        if sizes.insert(name.to_string(), *size).is_some() {
            panic!("einops pattern '{source}': duplicate size for axis '{name}'");
        }
    }
    for (group, &dim) in lhs_groups.iter().zip(dims.iter()) {
        if group.len() == 1 {
            let key = group[0].key();
            match sizes.get(&key) {
                Some(&known) if known != dim => {
                    panic!(
                        "einops pattern '{source}': axis '{key}' size {known} does not match input dim {dim}"
                    );
                }
                Some(_) => {}
                None => {
                    sizes.insert(key, dim);
                }
            }
            continue;
        }
        let unknown: Vec<String> =
            group.iter().map(Axis::key).filter(|key| !sizes.contains_key(key)).collect();
        if unknown.len() > 1 {
            panic!(
                "einops pattern '{source}': split of dim {dim} needs sizes for {}; pass them in sizes",
                unknown.join(", ")
            );
        }
        let mut product = 1;
        for axis in group.iter() {
            if let Some(&size) = sizes.get(&axis.key()) {
                product *= size;
            }
        }
        if let Some(name) = unknown.into_iter().next() {
            if product == 0 || dim % product != 0 {
                panic!(
                    "einops pattern '{source}': cannot infer '{name}': dim {dim} not divisible by {product}"
                );
            }
            sizes.insert(name, dim / product);
            product = dim;
        }
        if product != dim {
            panic!(
                "einops pattern '{source}': split sizes product {product} does not match input dim {dim}"
            );
        }
    }
    sizes
}

/// Shape of one side after merging its groups.
///
/// A passthrough `...` contributes its bound dims in order. A rhs `(...)`
/// contributes their product as one dim.
fn grouped_shape(
    side: &Side,
    sizes: &HashMap<String, usize>,
    bound: &[usize],
    source: &str,
) -> Vec<usize> {
    let mut shape = Vec::new();
    for group in &side.pre {
        shape.push(group_shape(group, sizes, source));
    }
    match side.ellipsis {
        None => {}
        Some(EllipsisKind::Pass) => shape.extend(bound.iter().copied()),
        Some(EllipsisKind::Flatten) => shape.push(bound.iter().product()),
    }
    for group in &side.post {
        shape.push(group_shape(group, sizes, source));
    }
    shape
}

/// Product of one named group after size resolution.
fn group_shape(group: &[Axis], sizes: &HashMap<String, usize>, source: &str) -> usize {
    group
        .iter()
        .map(|axis| {
            *sizes.get(&axis.key()).unwrap_or_else(|| {
                panic!(
                    "einops pattern '{source}': no size for axis '{}'; pass it in sizes",
                    axis.key()
                )
            })
        })
        .product()
}

/// Sizes bound by the lhs `...`, taken straight from the input shape.
fn ellipsis_sizes(input_shape: &[usize], pre_groups: usize, rank: usize) -> Vec<usize> {
    input_shape[pre_groups..pre_groups + rank].to_vec()
}

/// Fully split lhs flat keys: named keys around synthetic ellipsis keys.
fn expanded_lhs_flat(lhs: &Side, rank: usize) -> Vec<String> {
    flat_group_keys(&lhs.pre)
        .into_iter()
        .chain(ellipsis_keys(rank))
        .chain(flat_group_keys(&lhs.post))
        .collect()
}

/// Fully split lhs flat shape in the same order.
fn expanded_lhs_shape(
    lhs: &Side,
    sizes: &HashMap<String, usize>,
    ellipsis: &[usize],
    source: &str,
) -> Vec<usize> {
    flat_axis_sizes(&lhs.pre, sizes, source)
        .into_iter()
        .chain(ellipsis.iter().copied())
        .chain(flat_axis_sizes(&lhs.post, sizes, source))
        .collect()
}

/// Fully split sizes of named groups, one entry per flat axis.
fn flat_axis_sizes(
    groups: &[Vec<Axis>],
    sizes: &HashMap<String, usize>,
    source: &str,
) -> Vec<usize> {
    groups
        .iter()
        .flatten()
        .map(|axis| {
            *sizes.get(&axis.key()).unwrap_or_else(|| {
                panic!(
                    "einops pattern '{source}': no size for axis '{}'; pass it in sizes",
                    axis.key()
                )
            })
        })
        .collect()
}

/// Fully split rhs flat keys with a passthrough ellipsis.
fn expanded_rhs_flat(rhs: &Side, rank: usize) -> Vec<String> {
    flat_group_keys(&rhs.pre)
        .into_iter()
        .chain(ellipsis_keys(rank))
        .chain(flat_group_keys(&rhs.post))
        .collect()
}

/// Fully split rhs flat shape with a passthrough ellipsis.
fn expanded_rhs_shape(
    rhs: &Side,
    sizes: &HashMap<String, usize>,
    ellipsis: &[usize],
    source: &str,
) -> Vec<usize> {
    flat_axis_sizes(&rhs.pre, sizes, source)
        .into_iter()
        .chain(ellipsis.iter().copied())
        .chain(flat_axis_sizes(&rhs.post, sizes, source))
        .collect()
}

/// Merges the contiguous ellipsis run of a passthrough-ordered tensor.
fn merge_ellipsis_run(
    tensor: &Tensor,
    rhs: &Side,
    pass_shape: &[usize],
    rank: usize,
) -> Tensor {
    let pre: usize = rhs.pre.iter().flatten().count();
    let mut merged = pass_shape[..pre].to_vec();
    merged.push(pass_shape[pre..pre + rank].iter().product());
    merged.extend_from_slice(&pass_shape[pre + rank..]);
    reshape_unless(tensor, merged)
}

/// Positions of `order` inside `base`.
fn permutation(base: &[String], order: &[String], source: &str) -> Vec<usize> {
    order
        .iter()
        .map(|key| {
            base.iter().position(|other| other == key).unwrap_or_else(|| {
                panic!("einops pattern '{source}': rhs axis '{key}' not present on lhs")
            })
        })
        .collect()
}

/// Reshapes only when the shape actually changes.
fn reshape_unless(x: &Tensor, shape: Vec<usize>) -> Tensor {
    let current: Vec<usize> = x.layout().shape().iter().copied().collect();
    if current == shape { x.clone() } else { x.reshape(shape) }
}

/// Views `x` with one extra size-1 dim at `axis` without copying.
///
/// A size-1 dim addresses a single element, so stride 0 reads the same
/// element the copying reshape would expose. Training tensors keep the
/// copying reshape so the Reshape grad node stays in the graph.
fn view_insert_size1(x: &Tensor, axis: usize) -> Tensor {
    let mut shape: Vec<usize> = x.layout().shape().iter().copied().collect();
    shape.insert(axis, 1);
    let mut strides: Vec<isize> = x.layout().strides().iter().copied().collect();
    strides.insert(axis, 0);
    Tensor::new(x.storage_clone(), Layout::new(shape, strides, x.layout().offset), false, None)
}

/// Views `x` with the size-1 dim at `axis` removed without copying.
fn view_remove_size1(x: &Tensor, axis: usize) -> Tensor {
    let mut shape: Vec<usize> = x.layout().shape().iter().copied().collect();
    assert_eq!(shape[axis], 1);
    shape.remove(axis);
    let mut strides: Vec<isize> = x.layout().strides().iter().copied().collect();
    strides.remove(axis);
    Tensor::new(x.storage_clone(), Layout::new(shape, strides, x.layout().offset), false, None)
}

/// Permutes only when the order actually changes.
fn permute_unless(tensor: &Tensor, axes: Vec<usize>) -> Tensor {
    if axes.iter().enumerate().all(|(index, &axis)| index == axis) {
        tensor.clone()
    } else {
        tensor.permute(axes)
    }
}

impl Tensor {
    /// Rearranges axes by name: splits `(h d)` groups, permutes flat axes
    /// into rhs order, then merges rhs groups.
    ///
    /// The flat axis multiset is preserved exactly. Dropping axes needs
    /// [`Tensor::reduce`]. Adding axes needs [`Tensor::repeat`]. A `...` on
    /// both sides binds the same leading dims in order, so
    /// `x.rearrange("... (h d) -> ... h d", &[("h", h)])` splits the last
    /// dim at any rank. `(...)` on the rhs merges the bound dims into one.
    pub fn rearrange(&self, pattern: &str, sizes: &[(&str, usize)]) -> Tensor {
        let parsed = parse_pattern(pattern);
        let source = parsed.source.as_str();
        check_unique(&parsed.lhs, "lhs", source);
        check_unique(&parsed.rhs, "rhs", source);
        let input_shape: Vec<usize> = self.layout().shape().iter().copied().collect();
        let rank = bind_ellipsis(&parsed, &input_shape);
        let resolved = resolve_sizes(&parsed, &input_shape, sizes, rank);
        let bound = ellipsis_sizes(&input_shape, parsed.lhs.pre.len(), rank);
        let lhs_flat = expanded_lhs_flat(&parsed.lhs, rank);
        let rhs_flat = expanded_rhs_flat(&parsed.rhs, rank);
        for key in &rhs_flat {
            if !lhs_flat.contains(key) {
                panic!("einops rearrange '{source}': rhs axis '{key}' not present on lhs");
            }
        }
        if lhs_flat.len() != rhs_flat.len() {
            panic!(
                "einops rearrange '{source}': rearrange must preserve the axis multiset; use reduce or repeat"
            );
        }
        let permuted = permute_unless(
            &reshape_unless(self, expanded_lhs_shape(&parsed.lhs, &resolved, &bound, source)),
            permutation(&lhs_flat, &rhs_flat, source),
        );
        let pass_shape = expanded_rhs_shape(&parsed.rhs, &resolved, &bound, source);
        match parsed.rhs.ellipsis {
            Some(EllipsisKind::Flatten) => {
                let merged = merge_ellipsis_run(&permuted, &parsed.rhs, &pass_shape, rank);
                reshape_unless(&merged, grouped_shape(&parsed.rhs, &resolved, &bound, source))
            }
            _ => reshape_unless(
                &permuted,
                grouped_shape(&parsed.rhs, &resolved, &bound, source),
            ),
        }
    }

    /// Reduces dropped lhs axes with one multi-axis `sum`, `mean`, or `max`.
    ///
    /// All dropped axes reduce in a single call after moving them last, so a
    /// two-axis drop never sees a stale rank. A `...` on both sides keeps the
    /// batch dims while named axes drop, as in `x.reduce("... t c -> ... c",
    /// "mean", &[])`.
    pub fn reduce(&self, pattern: &str, op: &str, sizes: &[(&str, usize)]) -> Tensor {
        if !["sum", "mean", "max"].contains(&op) {
            panic!(
                "einops reduce '{pattern}': unsupported reduction '{op}'; use sum, mean, or max"
            );
        }
        let parsed = parse_pattern(pattern);
        let source = parsed.source.as_str();
        check_unique(&parsed.lhs, "lhs", source);
        check_unique(&parsed.rhs, "rhs", source);
        let input_shape: Vec<usize> = self.layout().shape().iter().copied().collect();
        let rank = bind_ellipsis(&parsed, &input_shape);
        let resolved = resolve_sizes(&parsed, &input_shape, sizes, rank);
        let bound = ellipsis_sizes(&input_shape, parsed.lhs.pre.len(), rank);
        let lhs_flat = expanded_lhs_flat(&parsed.lhs, rank);
        let rhs_flat = expanded_rhs_flat(&parsed.rhs, rank);
        for key in &rhs_flat {
            if !lhs_flat.contains(key) {
                panic!("einops pattern '{source}': rhs axis '{key}' not present on lhs");
            }
        }
        let mut dropped: Vec<String> =
            lhs_flat.iter().filter(|key| !rhs_flat.contains(key)).cloned().collect();
        dropped.dedup();
        if dropped.is_empty() {
            panic!("einops reduce '{source}': reduce pattern drops no axis; use rearrange");
        }
        let mut order = rhs_flat.clone();
        order.extend(dropped.clone());
        let permuted = permute_unless(
            &reshape_unless(self, expanded_lhs_shape(&parsed.lhs, &resolved, &bound, source)),
            permutation(&lhs_flat, &order, source),
        );
        let axes: Vec<usize> = (rhs_flat.len()..order.len()).collect();
        let reduced = match op {
            "sum" => permuted.sum(axes, false),
            "mean" => permuted.mean(axes, false),
            _ => permuted.max(axes, false),
        };
        let pass_shape = expanded_rhs_shape(&parsed.rhs, &resolved, &bound, source);
        match parsed.rhs.ellipsis {
            Some(EllipsisKind::Flatten) => {
                let merged = merge_ellipsis_run(&reduced, &parsed.rhs, &pass_shape, rank);
                reshape_unless(&merged, grouped_shape(&parsed.rhs, &resolved, &bound, source))
            }
            _ => reshape_unless(
                &reduced,
                grouped_shape(&parsed.rhs, &resolved, &bound, source),
            ),
        }
    }

    /// Tiles new rhs axes by unsqueezing size 1 dims, permuting them into
    /// rhs position, and broadcasting to full size.
    ///
    /// Every lhs axis must survive on the rhs. Dropping axes needs
    /// [`Tensor::reduce`]. A `...` on both sides tiles inside the batch, as
    /// in `x.repeat("... c -> ... c h", &[("h", 2)])`.
    pub fn repeat(&self, pattern: &str, sizes: &[(&str, usize)]) -> Tensor {
        let parsed = parse_pattern(pattern);
        let source = parsed.source.as_str();
        check_unique(&parsed.lhs, "lhs", source);
        check_unique(&parsed.rhs, "rhs", source);
        let input_shape: Vec<usize> = self.layout().shape().iter().copied().collect();
        let rank = bind_ellipsis(&parsed, &input_shape);
        let resolved = resolve_sizes(&parsed, &input_shape, sizes, rank);
        let bound = ellipsis_sizes(&input_shape, parsed.lhs.pre.len(), rank);
        let lhs_flat = expanded_lhs_flat(&parsed.lhs, rank);
        let rhs_flat = expanded_rhs_flat(&parsed.rhs, rank);
        for key in &lhs_flat {
            if !rhs_flat.contains(key) {
                panic!("einops repeat '{source}': repeat drops lhs axis '{key}'; use reduce");
            }
        }
        let mut new_axes: Vec<String> =
            rhs_flat.iter().filter(|key| !lhs_flat.contains(key)).cloned().collect();
        new_axes.dedup();
        if new_axes.is_empty() {
            panic!("einops repeat '{source}': repeat pattern adds no axis; use rearrange");
        }
        for key in &new_axes {
            if !resolved.contains_key(key) {
                panic!(
                    "einops repeat '{source}': repeat needs a size for new axis '{key}'; pass it in sizes"
                );
            }
        }
        let flat_lhs_shape = expanded_lhs_shape(&parsed.lhs, &resolved, &bound, source);
        let mut pre = flat_lhs_shape.clone();
        pre.extend(new_axes.iter().map(|_| 1));
        let unsqueezed = reshape_unless(self, flat_lhs_shape);
        let unsqueezed = reshape_unless(&unsqueezed, pre);
        let from: Vec<String> = lhs_flat.iter().cloned().chain(new_axes.iter().cloned()).collect();
        let permuted = permute_unless(&unsqueezed, permutation(&from, &rhs_flat, source));
        let pass_shape = expanded_rhs_shape(&parsed.rhs, &resolved, &bound, source);
        let tiled = permuted.broadcast(pass_shape.clone());
        match parsed.rhs.ellipsis {
            Some(EllipsisKind::Flatten) => {
                let merged = merge_ellipsis_run(&tiled, &parsed.rhs, &pass_shape, rank);
                reshape_unless(&merged, grouped_shape(&parsed.rhs, &resolved, &bound, source))
            }
            _ => reshape_unless(
                &tiled,
                grouped_shape(&parsed.rhs, &resolved, &bound, source),
            ),
        }
    }
}

/// A parsed two-input contraction equation: `lhs, rhs -> out`.
///
/// Each side holds whitespace separated labels in the same alphabet as the
/// einops axes plus one optional leading `...`. One token is one label, so
/// a compact run such as `bhts` names a single axis and never four. The
/// `...` binds zero or more leading batch dims in input order.
struct Equation {
    lhs: EquationSide,
    rhs: EquationSide,
    out: EquationSide,
    source: String,
}

/// One equation side: explicit labels plus a leading batch ellipsis.
struct EquationSide {
    ellipsis: bool,
    labels: Vec<String>,
}

/// Label roles for one single-contract batch matmul.
///
/// Validated once so the lowering reads positions instead of branching on
/// labels: batch labels live in both inputs and the output, each input keeps
/// at most one label into the output, and the contracted label lives in both
/// inputs but not the output. A side with no kept label is a batch vector:
/// it grows a size 1 matmul dim and squeezes it back after the product.
struct ContractPlan {
    batch: Vec<String>,
    keep_left: Option<String>,
    keep_right: Option<String>,
    contract: String,
}

/// Splits one equation side into a leading ellipsis plus labels.
fn parse_equation_side(kind: &str, side: &str, pattern: &str) -> EquationSide {
    let mut ellipsis = false;
    let mut labels = Vec::new();
    for (index, token) in side.split_whitespace().enumerate() {
        if token == "..." {
            if ellipsis {
                panic!(
                    "einsum '{pattern}': multiple '...' in {kind} side; at most one ellipsis per side"
                );
            }
            if index != 0 {
                panic!(
                    "einsum '{pattern}': '...' must lead the {kind} side; batch ellipsis comes first"
                );
            }
            ellipsis = true;
            continue;
        }
        if !valid_name(token) {
            panic!("einsum '{pattern}': '{token}' is not a valid axis name");
        }
        labels.push(token.to_string());
    }
    if labels.is_empty() && !ellipsis {
        panic!("einsum '{pattern}': {kind} side is empty");
    }
    let mut seen = Vec::new();
    for label in &labels {
        if seen.contains(label) {
            panic!(
                "einsum '{pattern}': duplicate label '{label}' in {kind} side; labels must be unique per side"
            );
        }
        seen.push(label.clone());
    }
    EquationSide { ellipsis, labels }
}

/// Parses `lhs, rhs -> out` into three label lists.
fn parse_equation(pattern: &str) -> Equation {
    let parts: Vec<&str> = pattern.split("->").collect();
    if parts.len() != 2 {
        panic!("einsum '{pattern}': pattern must contain exactly one '->'");
    }
    let inputs: Vec<&str> = parts[0].split(',').collect();
    if inputs.len() != 2 {
        panic!("einsum '{pattern}': pattern must contain exactly two inputs separated by ','");
    }
    Equation {
        lhs: parse_equation_side("input side 1", inputs[0], pattern),
        rhs: parse_equation_side("input side 2", inputs[1], pattern),
        out: parse_equation_side("output", parts[1], pattern),
        source: pattern.to_string(),
    }
}

/// Sorts equation labels into batch, kept, and contracted roles.
///
/// Ellipsis dims never enter the plan. They form a leading batch block that
/// the lowering carries alongside the named batch labels.
fn plan_contraction(eq: &Equation) -> ContractPlan {
    let source = eq.source.as_str();
    for label in &eq.out.labels {
        if !eq.lhs.labels.contains(label) && !eq.rhs.labels.contains(label) {
            panic!("einsum '{source}': output label '{label}' is not present in either input");
        }
    }
    for label in eq.lhs.labels.iter().chain(eq.rhs.labels.iter()) {
        let in_both = eq.lhs.labels.contains(label) && eq.rhs.labels.contains(label);
        if !eq.out.labels.contains(label) && !in_both {
            panic!(
                "einsum '{source}': input label '{label}' is missing from the output; only the contracted axis may leave the output"
            );
        }
    }
    let contracted: Vec<&String> = eq
        .lhs
        .labels
        .iter()
        .filter(|label| eq.rhs.labels.contains(label) && !eq.out.labels.contains(label))
        .collect();
    let contract = match contracted.as_slice() {
        [one] => (*one).clone(),
        [] => panic!(
            "einsum '{source}': no contracted axis; one label must appear in both inputs but not the output"
        ),
        many => panic!(
            "einsum '{source}': axes {} are all contracted; only single-axis contraction is supported",
            many.iter().map(|label| label.as_str()).collect::<Vec<_>>().join(", ")
        ),
    };
    let kept_left: Vec<&String> = eq
        .lhs
        .labels
        .iter()
        .filter(|label| eq.out.labels.contains(label) && !eq.rhs.labels.contains(label))
        .collect();
    let keep_left = match kept_left.as_slice() {
        [] => None,
        [one] => Some((*one).clone()),
        many => panic!(
            "einsum '{source}': left input keeps axes {}; each input must keep at most one",
            many.iter().map(|label| label.as_str()).collect::<Vec<_>>().join(", ")
        ),
    };
    let kept_right: Vec<&String> = eq
        .rhs
        .labels
        .iter()
        .filter(|label| eq.out.labels.contains(label) && !eq.lhs.labels.contains(label))
        .collect();
    let keep_right = match kept_right.as_slice() {
        [] => None,
        [one] => Some((*one).clone()),
        many => panic!(
            "einsum '{source}': right input keeps axes {}; each input must keep at most one",
            many.iter().map(|label| label.as_str()).collect::<Vec<_>>().join(", ")
        ),
    };
    if keep_left.is_none() && keep_right.is_none() {
        panic!(
            "einsum '{source}': neither input keeps an axis into the output; at least one input must keep exactly one"
        );
    }
    let batch: Vec<String> = eq
        .lhs
        .labels
        .iter()
        .filter(|label| eq.rhs.labels.contains(label) && eq.out.labels.contains(label))
        .cloned()
        .collect();
    ContractPlan { batch, keep_left, keep_right, contract }
}

impl Tensor {
    /// Contracts two tensors along one shared axis, written as an equation.
    ///
    /// The pattern is `left, right -> out` with whitespace separated labels.
    /// One label appears in both inputs but not the output: the contracted
    /// axis. Each input keeps exactly one label into the output. The rest are
    /// batch labels, which must match in size. Both inputs permute into
    /// `[batch, kept, contracted]` matmul form, [`Tensor::matmul`] runs, and
    /// the product permutes into output order. Gradients flow through those
    /// existing operators, so no separate backward exists. A leading `...` on
    /// a side binds extra batch dims, so attention scores read
    /// `Tensor::einsum("... q d, ... k d -> ... q k", &q, &k)` at any rank.
    ///
    /// Three shapes cover the supported uses:
    ///
    /// ```ignore
    /// let scores = Tensor::einsum("b h t d, b h s d -> b h t s", &q, &k);
    /// let out = Tensor::einsum("b h t s, b h s d -> b h t d", &attn, &v);
    /// let proj = Tensor::einsum("b t c, b c e -> b t e", &x, &w);
    /// ```
    pub fn einsum(pattern: &str, a: &Tensor, b: &Tensor) -> Tensor {
        let equation = parse_equation(pattern);
        let source = equation.source.as_str();
        let shape_a: Vec<usize> = a.layout().shape().iter().copied().collect();
        let shape_b: Vec<usize> = b.layout().shape().iter().copied().collect();
        let rank_a = bind_equation_side(&equation.lhs, &shape_a, source, "left", "first");
        let rank_b = bind_equation_side(&equation.rhs, &shape_b, source, "right", "second");
        if (equation.lhs.ellipsis || equation.rhs.ellipsis) && !equation.out.ellipsis {
            panic!(
                "einsum '{source}': an input has '...' but the output does not; batch dims are preserved in the output"
            );
        }
        if equation.lhs.ellipsis && equation.rhs.ellipsis && rank_a != rank_b {
            panic!(
                "einsum '{source}': '...' binds {rank_a} dims in the first input but {rank_b} in the second; batch rank must match"
            );
        }
        let batch_rank = rank_a.max(rank_b);
        let ellipsis_shape: Vec<usize> = if equation.lhs.ellipsis {
            shape_a[..rank_a].to_vec()
        } else if equation.rhs.ellipsis {
            shape_b[..rank_b].to_vec()
        } else {
            Vec::new()
        };
        if equation.lhs.ellipsis && equation.rhs.ellipsis {
            for (index, (&first, &second)) in
                shape_a[..rank_a].iter().zip(shape_b[..rank_b].iter()).enumerate()
            {
                if first != second {
                    panic!(
                        "einsum '{source}': batch '...' dim {index} has size {first} in the first input but {second} in the second"
                    );
                }
            }
        }
        let plan = plan_contraction(&equation);
        let size = |labels: &[String], shape: &[usize], rank: usize, label: &str| {
            shape[rank + labels.iter().position(|other| other == label).unwrap()]
        };
        let first = size(&equation.lhs.labels, &shape_a, rank_a, &plan.contract);
        let second = size(&equation.rhs.labels, &shape_b, rank_b, &plan.contract);
        if first != second {
            panic!(
                "einsum '{source}': contracted axis '{}' has size {first} in the first input but {second} in the second",
                plan.contract
            );
        }
        for label in &plan.batch {
            let first = size(&equation.lhs.labels, &shape_a, rank_a, label);
            let second = size(&equation.rhs.labels, &shape_b, rank_b, label);
            if first != second {
                panic!(
                    "einsum '{source}': batch axis '{label}' has size {first} in the first input but {second} in the second"
                );
            }
        }
        let position = |labels: &[String], rank: usize, label: &str| {
            rank + labels.iter().position(|other| other == label).unwrap()
        };
        let mut left_perm: Vec<usize> = (0..rank_a).collect();
        left_perm.extend(plan.batch.iter().map(|label| position(&equation.lhs.labels, rank_a, label)));
        if let Some(keep) = &plan.keep_left {
            left_perm.push(position(&equation.lhs.labels, rank_a, keep));
        }
        left_perm.push(position(&equation.lhs.labels, rank_a, &plan.contract));
        let mut right_perm: Vec<usize> = (0..rank_b).collect();
        right_perm
            .extend(plan.batch.iter().map(|label| position(&equation.rhs.labels, rank_b, label)));
        right_perm.push(position(&equation.rhs.labels, rank_b, &plan.contract));
        if let Some(keep) = &plan.keep_right {
            right_perm.push(position(&equation.rhs.labels, rank_b, keep));
        }
        let batch_shape: Vec<usize> = ellipsis_shape
            .iter()
            .copied()
            .chain(plan.batch.iter().map(|label| size(&equation.lhs.labels, &shape_a, rank_a, label)))
            .collect();
        let left = expand_batch(
            unsqueeze_vector(permute_unless(a, left_perm), plan.keep_left.is_none(), true),
            &batch_shape,
        );
        let right = expand_batch(
            unsqueeze_vector(permute_unless(b, right_perm), plan.keep_right.is_none(), false),
            &batch_shape,
        );
        let product = left.matmul(&right);
        let squeezed = squeeze_vector(product, plan.keep_left.is_none(), plan.keep_right.is_none());
        let mut out_perm: Vec<usize> = (0..batch_rank).collect();
        let order: Vec<String> = plan
            .batch
            .iter()
            .cloned()
            .chain(plan.keep_left.clone())
            .chain(plan.keep_right.clone())
            .collect();
        out_perm.extend(
            equation.out.labels.iter().map(|label| batch_rank + position(&order, 0, label)),
        );
        permute_unless(&squeezed, out_perm)
    }
}

/// Grows a size 1 matmul dim for a vector side: `[batch, k]` becomes
/// `[batch, 1, k]` on the left or `[batch, k, 1]` on the right.
fn unsqueeze_vector(tensor: Tensor, vector: bool, left: bool) -> Tensor {
    if !vector {
        return tensor;
    }
    let mut shape: Vec<usize> = tensor.layout().shape().iter().copied().collect();
    let axis = if left { shape.len() - 1 } else { shape.len() };
    shape.insert(axis, 1);
    if tensor.requires_grad() { tensor.reshape(shape) } else { view_insert_size1(&tensor, axis) }
}

/// Drops the size 1 dims grown for vector sides after the product.
fn squeeze_vector(tensor: Tensor, left_vector: bool, right_vector: bool) -> Tensor {
    if !left_vector && !right_vector {
        return tensor;
    }
    if tensor.requires_grad() {
        let shape: Vec<usize> = tensor.layout().shape().iter().copied().collect();
        let rank = shape.len();
        let mut squeezed = shape.clone();
        if left_vector {
            assert_eq!(squeezed[rank - 2], 1);
            squeezed.remove(rank - 2);
        }
        if right_vector {
            assert_eq!(squeezed[squeezed.len() - 1], 1);
            squeezed.pop();
        }
        return tensor.reshape(squeezed);
    }
    let mut out = tensor;
    if left_vector {
        let rank = out.layout().shape().ndim();
        assert_eq!(out.layout().shape()[rank - 2], 1);
        out = view_remove_size1(&out, rank - 2);
    }
    if right_vector {
        let rank = out.layout().shape().ndim();
        assert_eq!(out.layout().shape()[rank - 1], 1);
        out = view_remove_size1(&out, rank - 1);
    }
    out
}

/// Binds one equation side: explicit labels plus leading batch dims.
fn bind_equation_side(
    side: &EquationSide,
    shape: &[usize],
    source: &str,
    side_name: &str,
    input_name: &str,
) -> usize {
    if !side.ellipsis {
        if side.labels.len() != shape.len() {
            panic!(
                "einsum '{source}': {side_name} side has {} labels but the {input_name} input is {}-d (shape {shape:?}); one label per dim",
                side.labels.len(),
                shape.len()
            );
        }
        return 0;
    }
    if side.labels.len() > shape.len() {
        panic!(
            "einsum '{source}': {side_name} side has {} labels plus '...' but the {input_name} input is {}-d (shape {shape:?}); '...' binds zero or more dims",
            side.labels.len(),
            shape.len()
        );
    }
    shape.len() - side.labels.len()
}

/// Broadcasts matmul-form `[batch, kept, contracted]` to the full batch.
///
/// `matmul` needs equal rank with exact batch sizes, so an input without
/// `...` grows leading ones and broadcasts while a full batch stays put.
fn expand_batch(tensor: Tensor, batch_shape: &[usize]) -> Tensor {
    let shape: Vec<usize> = tensor.layout().shape().iter().copied().collect();
    let full_rank = batch_shape.len() + 2;
    if shape.len() == full_rank
        && shape[..batch_shape.len()] == batch_shape[..]
    {
        return tensor;
    }
    let mut full = batch_shape.to_vec();
    full.extend_from_slice(&shape[shape.len() - 2..]);
    if tensor.requires_grad() {
        let mut grown = vec![1; full_rank - shape.len()];
        grown.extend(shape.iter().copied());
        return reshape_unless(&tensor, grown).broadcast(full);
    }
    let mut out = tensor;
    for _ in shape.len()..full_rank {
        out = view_insert_size1(&out, 0);
    }
    out.broadcast(full)
}
