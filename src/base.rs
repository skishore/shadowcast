use std::cmp::{max, min};
use std::ops::{Add, Sub};

//////////////////////////////////////////////////////////////////////////////

// Point

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Point(pub i32, pub i32);

impl Point {
    pub fn dot(&self, other: Point) -> i64 {
        (self.0 as i64 * other.0 as i64) + (self.1 as i64 * other.1 as i64)
    }

    pub fn in_l2_range(&self, range: i32) -> bool {
        self.len_l2() <= range as f64 - 0.5
    }

    pub fn len_nethack(&self) -> i32 {
        let (ax, ay) = (self.0.abs() as i64, self.1.abs() as i64);
        ((46 * min(ax, ay) + 95 * max(ax, ay) + 25) / 100) as i32
    }

    pub fn len_taxicab(&self) -> i32 {
        self.0.abs() + self.1.abs()
    }

    pub fn len_l1(&self) -> i32 {
        max(self.0.abs(), self.1.abs())
    }

    pub fn len_l2(&self) -> f64 {
        (self.len_l2_squared() as f64).sqrt()
    }

    pub fn len_l2_squared(&self) -> i64 {
        let (x, y) = (self.0 as i64, self.1 as i64);
        x * x + y * y
    }

    pub fn normalize(&self, length: f64) -> Point {
        let factor = length / self.len_l2();
        let x = (self.0 as f64 * factor).round() as i32;
        let y = (self.1 as f64 * factor).round() as i32;
        Point(x, y)
    }
}

impl Add for Point {
    type Output = Point;
    fn add(self, other: Point) -> Point {
        Point(self.0 + other.0, self.1 + other.1)
    }
}

impl Sub for Point {
    type Output = Point;
    fn sub(self, other: Point) -> Point {
        Point(self.0 - other.0, self.1 - other.1)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Matrix

#[derive(Clone, Default)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub size: Point,
    pub default: T,
}

// SAFETY: Non-none index() results are always valid indices into data.
impl<T: Clone> Matrix<T> {
    pub fn new(size: Point, value: T) -> Self {
        assert!(0 <= size.0);
        assert!(0 <= size.1);
        let mut data = Vec::new();
        data.resize((size.0 * size.1) as usize, value.clone());
        Self { data, size, default: value }
    }

    pub fn get(&self, point: Point) -> T {
        let Some(x) = self.index(point) else { return self.default.clone(); };
        unsafe { self.data.get_unchecked(x).clone() }
    }

    pub fn set(&mut self, point: Point, value: T) {
        let Some(x) = self.index(point) else { return; };
        unsafe { *self.data.get_unchecked_mut(x) = value; }
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    pub fn entry_ref(&self, point: Point) -> &T {
        let Some(x) = self.index(point) else { return &self.default; };
        unsafe { self.data.get_unchecked(x) }
    }

    pub fn entry_mut(&mut self, point: Point) -> Option<&mut T> {
        let Some(x) = self.index(point) else { return None; };
        unsafe { Some(self.data.get_unchecked_mut(x)) }
    }

    #[inline(always)]
    pub fn contains(&self, point: Point) -> bool {
        let Point(px, py) = point;
        let Point(sx, sy) = self.size;
        0 <= px && px < sx && 0 <= py && py < sy
    }

    #[inline(always)]
    pub fn index(&self, point: Point) -> Option<usize> {
        if !self.contains(point) { return None; }
        Some((point.0 + point.1 * self.size.0) as usize)
    }
}
