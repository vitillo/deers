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

/// One flat axis inside a pattern group.
#[derive(Debug, Clone, PartialEq)]
enum Axis {
    /// A user written name such as `b` or `head`.
    Named(String),
    /// An anonymous `_`, identified by its flat position on its own side.
    Anon(usize),
}

/// One side of a `lhs -> rhs` pattern: an ordered list of axis groups.
#[derive(Debug, Clone)]
struct Side {
    groups: Vec<Vec<Axis>>,
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

/// Parses one side of a pattern into groups.
fn parse_side(source: &str, pattern: &str) -> Side {
    let mut groups: Vec<Vec<Axis>> = Vec::new();
    let mut anon = 0;
    let mut token = String::new();
    let mut chars = source.chars().peekable();
    let flush = |token: &mut String, groups: &mut Vec<Vec<Axis>>, anon: &mut usize| {
        let word = token.trim().to_string();
        token.clear();
        if word.is_empty() {
            return;
        }
        if word == "_" {
            groups.push(vec![Axis::Anon(*anon)]);
            *anon += 1;
        } else {
            groups.push(vec![Axis::Named(check_name(&word, pattern))]);
        }
    };
    while let Some(next) = chars.next() {
        match next {
            '(' => {
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
                    panic!(
                        "einops pattern '{pattern}': '(' has no closing ')' in '{source}'"
                    );
                }
                let mut group = Vec::new();
                for word in inner.split_whitespace() {
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
                groups.push(group);
            }
            current if current.is_whitespace() => flush(&mut token, &mut groups, &mut anon),
            _ => token.push(next),
        }
    }
    flush(&mut token, &mut groups, &mut anon);
    if groups.is_empty() {
        panic!("einops pattern '{pattern}': empty pattern side '{source}'");
    }
    Side { groups }
}

/// Rejects tokens that are not plain axis names.
fn check_name(word: &str, pattern: &str) -> String {
    let valid = !word.is_empty()
        && word
            .chars()
            .all(|current| current.is_ascii_alphanumeric() || current == '_')
        && !word.chars().next().is_some_and(|current| current.is_ascii_digit());
    if !valid {
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
    let lhs_anons = lhs.groups.iter().flatten().filter(|axis| matches!(axis, Axis::Anon(_))).count();
    let rhs_anons = rhs.groups.iter().flatten().filter(|axis| matches!(axis, Axis::Anon(_))).count();
    if lhs_anons != rhs_anons {
        panic!(
            "einops pattern '{pattern}': anonymous axis count differs with {lhs_anons} on lhs vs {rhs_anons} on rhs; '_' pairs positionally"
        );
    }
    Pattern { lhs, rhs, source: pattern.to_string() }
}

/// Flat axis keys in group order.
fn flat_keys(side: &Side) -> Vec<String> {
    side.groups.iter().flatten().map(Axis::key).collect()
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
) -> HashMap<String, usize> {
    let source = pattern.source.as_str();
    if pattern.lhs.groups.len() != input_shape.len() {
        panic!(
            "einops pattern '{source}': lhs has {} groups but input is {}-d (shape {input_shape:?}); one group per input dim"
            ,
            pattern.lhs.groups.len(),
            input_shape.len()
        );
    }
    let mut sizes: HashMap<String, usize> = HashMap::new();
    for (name, size) in hints {
        if sizes.insert(name.to_string(), *size).is_some() {
            panic!("einops pattern '{source}': duplicate size for axis '{name}'");
        }
    }
    for (group, &dim) in pattern.lhs.groups.iter().zip(input_shape.iter()) {
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
        let unknown: Vec<String> = group
            .iter()
            .map(Axis::key)
            .filter(|key| !sizes.contains_key(key))
            .collect();
        if unknown.len() > 1 {
            panic!(
                "einops pattern '{source}': split of dim {dim} needs sizes for {}; pass them in sizes"
                ,
                unknown.join(", ")
            );
        }
        let mut product = 1;
        for axis in group {
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
fn grouped_shape(side: &Side, sizes: &HashMap<String, usize>, source: &str) -> Vec<usize> {
    side.groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|axis| {
                    *sizes.get(&axis.key()).unwrap_or_else(|| {
                        panic!(
                            "einops pattern '{source}': no size for axis '{}'; pass it in sizes"
                            ,
                            axis.key()
                        )
                    })
                })
                .product()
        })
        .collect()
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
    if current == shape {
        x.clone()
    } else {
        x.reshape(shape)
    }
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
    /// [`Tensor::reduce`]. Adding axes needs [`Tensor::repeat`].
    pub fn rearrange(&self, pattern: &str, sizes: &[(&str, usize)]) -> Tensor {
        let parsed = parse_pattern(pattern);
        let source = parsed.source.as_str();
        check_unique(&parsed.lhs, "lhs", source);
        check_unique(&parsed.rhs, "rhs", source);
        let input_shape: Vec<usize> = self.layout().shape().iter().copied().collect();
        let resolved = resolve_sizes(&parsed, &input_shape, sizes);
        let lhs_flat = flat_keys(&parsed.lhs);
        let rhs_flat = flat_keys(&parsed.rhs);
        for key in &rhs_flat {
            if !lhs_flat.contains(key) {
                panic!(
                    "einops rearrange '{source}': rhs axis '{key}' not present on lhs"
                );
            }
        }
        if lhs_flat.len() != rhs_flat.len()
        {
            panic!(
                "einops rearrange '{source}': rearrange must preserve the axis multiset; use reduce or repeat"
            );
        }
        let flat_lhs_shape: Vec<usize> =
            lhs_flat.iter().map(|key| resolved[key.as_str()]).collect();
        let permuted = permute_unless(
            &reshape_unless(self, flat_lhs_shape),
            permutation(&lhs_flat, &rhs_flat, source),
        );
        reshape_unless(&permuted, grouped_shape(&parsed.rhs, &resolved, source))
    }

    /// Reduces dropped lhs axes with one multi-axis `sum`, `mean`, or `max`.
    ///
    /// All dropped axes reduce in a single call after moving them last, so a
    /// two-axis drop never sees a stale rank.
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
        let resolved = resolve_sizes(&parsed, &input_shape, sizes);
        let lhs_flat = flat_keys(&parsed.lhs);
        let rhs_flat = flat_keys(&parsed.rhs);
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
            &reshape_unless(
                self,
                lhs_flat.iter().map(|key| resolved[key.as_str()]).collect(),
            ),
            permutation(&lhs_flat, &order, source),
        );
        let axes: Vec<usize> = (rhs_flat.len()..order.len()).collect();
        let reduced = match op {
            "sum" => permuted.sum(axes, false),
            "mean" => permuted.mean(axes, false),
            _ => permuted.max(axes, false),
        };
        reshape_unless(&reduced, grouped_shape(&parsed.rhs, &resolved, source))
    }

    /// Tiles new rhs axes by unsqueezing size 1 dims, permuting them into
    /// rhs position, and broadcasting to full size.
    ///
    /// Every lhs axis must survive on the rhs. Dropping axes needs
    /// [`Tensor::reduce`].
    pub fn repeat(&self, pattern: &str, sizes: &[(&str, usize)]) -> Tensor {
        let parsed = parse_pattern(pattern);
        let source = parsed.source.as_str();
        check_unique(&parsed.lhs, "lhs", source);
        check_unique(&parsed.rhs, "rhs", source);
        let input_shape: Vec<usize> = self.layout().shape().iter().copied().collect();
        let resolved = resolve_sizes(&parsed, &input_shape, sizes);
        let lhs_flat = flat_keys(&parsed.lhs);
        let rhs_flat = flat_keys(&parsed.rhs);
        for key in &lhs_flat {
            if !rhs_flat.contains(key) {
                panic!(
                    "einops repeat '{source}': repeat drops lhs axis '{key}'; use reduce"
                );
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
        let flat_lhs_shape: Vec<usize> =
            lhs_flat.iter().map(|key| resolved[key.as_str()]).collect();
        let mut pre = flat_lhs_shape.clone();
        pre.extend(new_axes.iter().map(|_| 1));
        let unsqueezed = reshape_unless(self, flat_lhs_shape);
        let unsqueezed = reshape_unless(&unsqueezed, pre);
        let from: Vec<String> =
            lhs_flat.iter().cloned().chain(new_axes.iter().cloned()).collect();
        let permuted = permute_unless(&unsqueezed, permutation(&from, &rhs_flat, source));
        let full: Vec<usize> = rhs_flat.iter().map(|key| resolved[key.as_str()]).collect();
        reshape_unless(
            &permuted.broadcast(full),
            grouped_shape(&parsed.rhs, &resolved, source),
        )
    }
}
