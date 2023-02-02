use std::{ptr::NonNull, ops::{IndexMut, Index, Mul, Add, Sub}, fmt::Display};

pub use crate::helpers::traits::{AsIndex, Numerical, MatOperations};
use crate::helpers::memory::{allocate, deallocate, copymemory};
use crate::helpers::errors::Error;

pub struct Matrix<T: Numerical<T>> {
    pub rows: usize,
    pub cols: usize,
    pub data: NonNull<T>,
    scale: T,
    shift: T,
    transposed: bool,
    pointwise: bool
}

pub struct MatOpResult<T: Numerical<T>>(pub Result<Matrix<T>, Error>);

impl<T: Numerical<T>> Matrix<T> {
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

    fn get_data_mut(&self, index: usize) -> &mut T {
        assert!(index < self.rows * self.cols, "Index out of bounds");
        unsafe {self.data.as_ptr().add(index).as_mut().unwrap()}
    }

    unsafe fn get_data_mut_unchecked(&self, index: usize) -> &mut T {
        self.data.as_ptr().add(index).as_mut().unwrap()
    }
}

impl<T: Numerical<T>> Drop for Matrix<T> {
    fn drop(&mut self) {
        deallocate::<T>(self.data, self.rows, self.cols);
    }
}

impl<T: Numerical<T>> Display for Matrix<T> {
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

// ---- OPERATIONS ----
// Mul
fn matmul<T: Numerical<T>>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> MatOpResult<T> {
    if lhs.cols != rhs.rows {
        return MatOpResult(Err(Error::InvalidDimensions((0, lhs.cols), (rhs.rows, 0))));
    }
    let mut result = Matrix::new(lhs.rows, rhs.cols);
    for i in 0..lhs.rows {
        for j in 0..rhs.cols {
            for k in 0..lhs.cols {
                unsafe {*result.get_data_mut_unchecked(i * result.cols + j) += 
                    *lhs.get_data_mut_unchecked(i * lhs.cols + k) * *rhs.get_data_mut_unchecked(k * rhs.cols + j)}
            }
        }
    }
    MatOpResult(Ok(result))
}
fn pointwisemul<T: Numerical<T>>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> MatOpResult<T> {
    if lhs.rows != rhs.rows || lhs.cols != rhs.cols {
        return MatOpResult(Err(Error::InvalidDimensions((lhs.rows, lhs.cols), (rhs.rows, rhs.cols))));
    }
    let mut result = Matrix::new(lhs.rows, lhs.cols);
    for i in 0..(lhs.rows * lhs.cols) {
        unsafe {*result.get_data_mut_unchecked(i) = *lhs.get_data_mut_unchecked(i) * *rhs.get_data_mut_unchecked(i)}
    }
    MatOpResult(Ok(result))
}
impl<T: Numerical<T>> Mul<Matrix<T>> for Matrix<T> {
    type Output = MatOpResult<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.pointwise || rhs.pointwise {
            pointwisemul(&self, &rhs)
        } else {
            matmul(&self, &rhs)
        }
    }
}
impl<T: Numerical<T>> Mul<MatOpResult<T>> for Matrix<T> {
    type Output = MatOpResult<T>;

    fn mul(self, rhs: MatOpResult<T>) -> Self::Output {
        let lhs = self;
        let Ok(rhs) = rhs.0 else {return rhs};
        if lhs.pointwise || rhs.pointwise {
            pointwisemul(&lhs, &rhs)
        } else {
            matmul(&lhs, &rhs)
        }
    }
}
impl<T: Numerical<T>> Mul<Matrix<T>> for MatOpResult<T> {
    type Output = MatOpResult<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        let Ok(lhs) = self.0 else {return self};
        if lhs.pointwise || rhs.pointwise {
            pointwisemul(&lhs, &rhs)
        } else {
            matmul(&lhs, &rhs)
        }
    }
}
impl<T: Numerical<T>> Mul<MatOpResult<T>> for MatOpResult<T> {
    type Output = MatOpResult<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let Ok(lhs) = self.0 else {return self};
        let Ok(rhs) = rhs.0 else {return rhs};
        if lhs.pointwise || rhs.pointwise {
            pointwisemul(&lhs, &rhs)
        } else {
            matmul(&lhs, &rhs)
        }
    }
}
// Add
fn matadd<T: Numerical<T>>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> MatOpResult<T> {
    if lhs.rows != rhs.rows || lhs.cols != rhs.cols {
        return MatOpResult(Err(Error::InvalidDimensions((lhs.rows, lhs.cols), (rhs.rows, rhs.cols))));
    }
    let result = Matrix::new(lhs.rows, lhs.cols);
    for i in 0..(lhs.rows * lhs.cols) {
        unsafe {*result.get_data_mut_unchecked(i) = *lhs.get_data_mut_unchecked(i) + *rhs.get_data_mut_unchecked(i)}
    }
    MatOpResult(Ok(result))
}
impl<T: Numerical<T>> Add<Matrix<T>> for Matrix<T> {
    type Output = MatOpResult<T>;

    fn add(self, rhs: Self) -> Self::Output {
        matadd(&self, &rhs)
    }
}
impl<T: Numerical<T>> Add<MatOpResult<T>> for Matrix<T> {
    type Output = MatOpResult<T>;

    fn add(self, rhs: MatOpResult<T>) -> Self::Output {
        let lhs = self;
        let Ok(rhs) = rhs.0 else {return rhs};
        matadd(&lhs, &rhs)
    }
}
impl<T: Numerical<T>> Add<Matrix<T>> for MatOpResult<T> {
    type Output = MatOpResult<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        let Ok(lhs) = self.0 else {return self};
        matadd(&lhs, &rhs)
    }
}
impl<T: Numerical<T>> Add<MatOpResult<T>> for MatOpResult<T> {
    type Output = MatOpResult<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let Ok(lhs) = self.0 else {return self};
        let Ok(rhs) = rhs.0 else {return rhs};
        matadd(&lhs, &rhs)
    }
}
// Sub
fn matsub<T: Numerical<T>>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> MatOpResult<T> {
    if lhs.rows != rhs.rows || lhs.cols != rhs.cols {
        return MatOpResult(Err(Error::InvalidDimensions((lhs.rows, lhs.cols), (rhs.rows, rhs.cols))));
    }
    let mut result = Matrix::new(lhs.rows, lhs.cols);
    for i in 0..(lhs.rows * lhs.cols) {
        unsafe {*result.get_data_mut_unchecked(i) = *lhs.get_data_mut_unchecked(i) - *rhs.get_data_mut_unchecked(i)}
    }
    MatOpResult(Ok(result))
}
impl<T: Numerical<T>> Sub<Matrix<T>> for Matrix<T> {
    type Output = MatOpResult<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        matsub(&self, &rhs)
    }
}
impl<T: Numerical<T>> Sub<MatOpResult<T>> for Matrix<T> {
    type Output = MatOpResult<T>;

    fn sub(self, rhs: MatOpResult<T>) -> Self::Output {
        let lhs = self;
        let Ok(rhs) = rhs.0 else {return rhs};
        matsub(&lhs, &rhs)
    }
}
impl<T: Numerical<T>> Sub<Matrix<T>> for MatOpResult<T> {
    type Output = MatOpResult<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        let Ok(lhs) = self.0 else {return self};
        matsub(&lhs, &rhs)
    }
}
impl<T: Numerical<T>> Sub<MatOpResult<T>> for MatOpResult<T> {
    type Output = MatOpResult<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let Ok(lhs) = self.0 else {return self};
        let Ok(rhs) = rhs.0 else {return rhs};
        matsub(&lhs, &rhs)
    }
}
impl<T: Numerical<T>> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < self.rows, "Row index out of bounds");
        assert!(index.1 < self.cols, "Column index out of bounds");
        unsafe {self.data.as_ptr().add(index.0 * self.cols + index.1).as_ref().unwrap()}
    }
}
impl<T: Numerical<T>> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(index.0 < self.rows, "Row index out of bounds");
        assert!(index.1 < self.cols, "Column index out of bounds");
        unsafe {self.data.as_ptr().add(index.0 * self.cols + index.1).as_mut().unwrap()}
    }
}
impl<T: Numerical<T>> MatOperations<T> for Matrix<T> {
    fn new(rows: usize, cols: usize) -> Self {
        let data = allocate::<T>(rows, cols);
        for i in 0..(rows * cols) {
            unsafe {data.as_ptr().add(i).write(std::mem::zeroed())}
        }
        Self {
            rows,
            cols,
            data,
            scale: T::ident(),
            shift: T::zero(),
            transposed: false,
            pointwise: false
        }
    }
    fn dup(&self) -> Self {
        let mut result = Self::new(self.rows, self.cols);
        unsafe {copymemory(&self.data, &mut result.data, 0..(self.rows * self.cols), 0)};
        result
    }
    fn get<U: AsIndex>(&self, rows: U, cols: U) -> Self {
        let ((row_start, row_end), (col_start, col_end)) = self.get_bounds(rows, cols);
        let mut result = Self::new(row_end - row_start, col_end - col_start);
        for i in row_start..row_end {
            unsafe {copymemory(&self.data, &mut result.data, (i * self.cols + col_start)..(i * self.cols + col_end), 
                (i - row_start) * result.cols)}
        }
        result
    }
    fn scale(&mut self, scale: T) -> &mut Self {
        self.scale *= scale;
        for i in 0..(self.rows * self.cols) {
            unsafe {*self.get_data_mut_unchecked(i) *= scale};
        }
        self
    }
    fn shift(&mut self, shift: T) -> &mut Self {
        self.shift += shift;
        for i in 0..(self.rows * self.cols) {
            unsafe {*self.get_data_mut_unchecked(i) += shift};
        }
        self
    }
    fn p(mut self) -> Self {
        self.pointwise = true;
        self
    }
    fn t(mut self) -> Self {
        self.transposed = !self.transposed;
        for i in 0..self.rows - 1 {
            for j in i + 1..self.cols {
                unsafe {std::mem::swap(self.get_data_mut_unchecked(i * self.cols + j), 
                    self.get_data_mut_unchecked(j * self.cols + i))};
            }
        }
        std::mem::swap(&mut self.rows, &mut self.cols);
        self
    }
}

#[macro_export]
macro_rules! mat {
    ($rows:expr, $cols:expr; data: [$t:ty] = $($x:expr), +) => {
        {
            let mut result = Matrix::new($rows, $cols);
            let mut data: [$t; $cols * $rows] = [$($x),+];
            unsafe {crate::helpers::memory::copymemory(&std::ptr::NonNull::new(data.as_mut_ptr()).unwrap(), &mut result.data, 0..data.len(), 0)}
            result
        }
    };
}