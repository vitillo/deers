#![allow(dead_code)]

use std::ops::{Index, IndexMut};

/// The dimensions of a tensor (e.g. `[2, 3, 4]` for a 3-D tensor).
///
/// Can be created from tuples: `Shape::from((2, 3))` or from `Vec<usize>`.
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Shape {
    shape: Vec<usize>,
}

impl Shape {
    /// Creates a shape from a vector of dimension sizes.
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }

    /// Returns the strides for a contiguous row-major layout with this shape.
    pub fn compact_strides(&self) -> Strides {
        let mut strides = vec![1isize; self.shape.len()];
        if self.shape.len() > 1 {
            for i in (0..(self.shape.len() - 1)).rev() {
                strides[i] = self.shape[i + 1] as isize * strides[i + 1];
            }
        }
        strides.into()
    }

    /// Total number of elements (product of all dimensions).
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.shape.iter()
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.shape[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.shape[index]
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Shape::new(value)
    }
}

macro_rules! impl_from_tuple {
    ($type:tt $($idx:tt $t:ty),*) => {
        impl From<($($t,)*)> for $type {
            #[allow(unused_variables)]
            fn from(value: ($($t,)*)) -> Self {
                $type::new(vec![$(value.$idx),*])
            }
        }
    };
}

impl_from_tuple!(Shape);
impl_from_tuple!(Shape 0 usize);
impl_from_tuple!(Shape 0 usize, 1 usize);
impl_from_tuple!(Shape 0 usize, 1 usize, 2 usize);

/// Per-dimension byte offsets that map logical indices to storage positions.
///
/// A stride of 0 indicates a broadcasted dimension.
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Strides(pub Vec<isize>);

impl Strides {
    pub fn new(strides: Vec<isize>) -> Self {
        Self(strides)
    }

    pub fn iter(&self) -> impl Iterator<Item = &isize> {
        self.0.iter()
    }
}

impl_from_tuple!(Strides);
impl_from_tuple!(Strides 0 isize);
impl_from_tuple!(Strides 0 isize, 1 isize);
impl_from_tuple!(Strides 0 isize, 1 isize, 2 isize);

impl<'a> IntoIterator for &'a Strides {
    type Item = &'a isize;
    type IntoIter = std::slice::Iter<'a, isize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Index<usize> for Strides {
    type Output = isize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl From<Vec<isize>> for Strides {
    fn from(value: Vec<isize>) -> Self {
        Self(value)
    }
}

/// Describes how a tensor's elements are laid out in memory.
///
/// Combines [`Shape`] (logical dimensions), [`Strides`] (memory stepping),
/// and an offset into the storage buffer. View operations like `permute` and
/// `broadcast` change the layout without copying data.
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Layout {
    pub shape: Shape,
    pub strides: Strides,
    pub offset: usize,
}

impl Layout {
    /// Creates a layout with explicit shape, strides, and offset.
    pub fn new(shape: impl Into<Shape>, strides: impl Into<Strides>, offset: usize) -> Self {
        Self {
            shape: shape.into(),
            strides: strides.into(),
            offset,
        }
    }

    /// Returns the shape of this layout.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the strides of this layout.
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns a new layout with dimensions reordered according to `axis`.
    pub fn permute(&self, axis: &Shape) -> Self {
        assert!(axis.iter().all(|x| *x < self.ndim()));

        let shape = axis.iter().map(|&i| self.shape[i]).collect();
        let strides = axis.iter().map(|&i| self.strides.0[i]).collect();
        Self {
            shape: Shape::new(shape),
            strides: Strides(strides),
            offset: self.offset,
        }
    }

    /// Returns true if this layout is contiguous row-major (no gaps, reordering, or offset).
    pub fn is_compact(&self) -> bool {
        self.offset == 0 && self.shape.compact_strides() == self.strides
    }

}

impl From<Shape> for Layout {
    fn from(shape: Shape) -> Self {
        let strides = shape.compact_strides();
        Self {
            shape,
            strides,
            offset: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_to_contiguous_strides() {
        let shape = Shape::from((2, 3, 4));
        let strides = shape.compact_strides();
        assert_eq!(strides.0, vec![12, 4, 1]);
    }

    #[test]
    fn test_shape_size() {
        let shape = Shape::from((2, 3, 4));
        assert_eq!(shape.size(), 24);
    }

    #[test]
    fn test_layout_permute() {
        let shape = Shape::from((2, 3, 4));
        let layout = Layout::from(shape);

        let layout = layout.permute(&Shape::from((1, 2, 0)));

        assert_eq!(layout.shape.shape, vec![3, 4, 2]);
        assert_eq!(layout.strides.0, vec![4, 1, 12]);
    }

    #[test]
    fn test_layout_with_offset_is_not_compact() {
        let layout = Layout::new(Shape::from((2, 3)), Strides::from(vec![3, 1]), 1);
        assert!(!layout.is_compact());
    }
}
