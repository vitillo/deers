#![allow(dead_code)]

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Shape(pub Vec<i32>);

impl Shape {
    pub fn contiguous_strides(&self) -> Strides {
        let mut strides = vec![1; self.0.len()];
        for i in (0..(self.0.len() - 1)).rev() {
            strides[i] = self.0[i + 1] * strides[i + 1];
        }
        strides.into()
    }

    pub fn size(&self) -> usize {
        self.0
            .iter()
            .copied()
            .map(|n| usize::try_from(n).unwrap())
            .product()
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<(i32,)> for Shape {
    fn from(value: (i32,)) -> Self {
        Self(vec![value.0])
    }
}

impl From<(i32, i32)> for Shape {
    fn from(value: (i32, i32)) -> Self {
        Self(vec![value.0, value.1])
    }
}

impl From<(i32, i32, i32)> for Shape {
    fn from(value: (i32, i32, i32)) -> Self {
        Self(vec![value.0, value.1, value.2])
    }
}

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Strides(pub Vec<i32>);

impl From<Vec<i32>> for Strides {
    fn from(value: Vec<i32>) -> Self {
        Self(value)
    }
}

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Layout {
    pub shape: Shape,
    pub strides: Strides,
}

impl Layout {
    pub fn size(&self) -> usize {
        self.shape.size()
    }
}

impl From<Shape> for Layout {
    fn from(shape: Shape) -> Self {
        let strides = shape.contiguous_strides();
        Self { shape, strides }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_to_contiguous_strides() {
        let shape = Shape::from((2, 3, 4));
        let strides = shape.contiguous_strides();
        assert_eq!(strides.0, vec![12, 4, 1]);
    }
}
