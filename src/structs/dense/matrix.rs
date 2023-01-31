use std::{ptr::NonNull, ops::{IndexMut, Index, Mul, Add}, fmt::Display};

use crate::helpers::traits::AsIndex;
use crate::helpers::memory::{allocate, deallocate, copymemory};

pub struct Matrix<T: Sized> {
    pub rows: usize,
    pub cols: usize,
    pub data: NonNull<T>,
    t: bool
}

impl<T: Sized> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = allocate::<T>(rows, cols);
        for i in 0..(rows * cols) {
            unsafe {data.as_ptr().add(i).write(std::mem::zeroed())}
        }
        Self {
            rows,
            cols,
            data,
            t: false
        }
    }

    fn get_bounds<U: AsIndex>(&self, rows: U, cols: U) -> ((usize, usize), (usize, usize)) {
        let row_start = match rows.start() {
            Some(x) => x,
            None => 0
        };
        let row_end = match rows.end() {
            Some(x) => x,
            None => self.rows
        };
        let col_start = match cols.start() {
            Some(x) => x,
            None => 0
        };
        let col_end = match cols.end() {
            Some(x) => x,
            None => self.cols
        };
        assert!(row_end <= self.rows, "Row end index out of bounds");
        assert!(col_end <= self.cols, "Column end index out of bounds");
        assert!(row_start < row_end, "Row start index must be smaller than row end index");
        assert!(col_start < col_end, "Column start index must be smaller than column end index");
        ((row_start, row_end), (col_start, col_end))
    }

    pub fn get<U: AsIndex>(&self, rows: U, cols: U) -> Self {
        let ((row_start, row_end), (col_start, col_end)) = self.get_bounds(rows, cols);
        let mut result = Self::new(row_end - row_start, col_end - col_start);
        for i in row_start..row_end {
            unsafe {copymemory(&self.data, &mut result.data, (i * self.cols + col_start)..(i * self.cols + col_end), 
                (i - row_start) * result.cols)}
        }
        result
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < self.rows, "Row index out of bounds");
        assert!(index.1 < self.cols, "Column index out of bounds");
        unsafe {self.data.as_ptr().add(index.0 * self.cols + index.1).as_ref().unwrap()}
    }
}
impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(index.0 < self.rows, "Row index out of bounds");
        assert!(index.1 < self.cols, "Column index out of bounds");
        unsafe {self.data.as_ptr().add(index.0 * self.cols + index.1).as_mut().unwrap()}
    }
}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        deallocate::<T>(self.data, self.rows, self.cols);
    }
}

impl<T: Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{} ", self[(i, j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
} 

#[macro_export]
macro_rules! mat {
    ($rows:expr, $cols:expr; data: [$t:ty] = $($x:expr), +) => {
        {
            let mut result = Matrix::new($rows, $cols);
            let mut data: [$t; $cols * $rows] = [$($x),+];
            // assert!(data.len() == $rows * $cols, "Matrix size does not match data length");
            unsafe {crate::helpers::memory::copymemory(&std::ptr::NonNull::new(data.as_mut_ptr()).unwrap(), &mut result.data, 0..data.len(), 0)}
            result
        }
    };
}