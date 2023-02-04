use std::{ptr::NonNull, ops::{IndexMut, Index, Mul, Add, Sub}, fmt::Display};

pub use crate::helpers::traits::{AsIndex, Numerical, Matrix};
use crate::helpers::memory::{allocate, deallocate, copymemory};
use crate::helpers::errors::Error;

pub struct DenseData<T: Numerical<T>> {
    pub rows: usize,
    pub cols: usize,
    pub values: NonNull<T>,
    transposed: bool,
    pointwise: bool
}

pub struct Dense<T: Numerical<T>> {
    pub data: Result<DenseData<T>, Error>
}

impl<T: Numerical<T>> DenseData<T> {
    pub fn from_values(rows: usize, cols: usize, values: NonNull<T>) -> Self {
        Self {
            rows,
            cols,
            values,
            transposed: false,
            pointwise: false
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

    fn get_data_mut(&self, index: usize) -> &mut T {
        assert!(index < self.rows * self.cols, "Index out of bounds");
        unsafe {self.values.as_ptr().add(index).as_mut().unwrap()}
    }

    unsafe fn get_data_mut_unchecked(&self, index: usize) -> &mut T {
        self.values.as_ptr().add(index).as_mut().unwrap()
    }
}

impl<T: Numerical<T>> Drop for DenseData<T> {
    fn drop(&mut self) {
        deallocate::<T>(self.values, self.rows, self.cols);
    }
}

impl<T: Numerical<T>> Display for DenseData<T> {
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

impl<T: Numerical<T>> Display for Dense<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            Ok(x) => write!(f, "{}", x),
            Err(x) => write!(f, "{}", x)
        }
    }
}

// ---- OPERATIONS ----
// Mul
fn matmul<T: Numerical<T>>(lhs: &DenseData<T>, rhs: &DenseData<T>) -> Dense<T> {
    if lhs.cols != rhs.rows {
        return Dense::err(Error::InvalidDimensions((0, lhs.cols), (rhs.rows, 0)));
    }
    let mut result = Dense::new(lhs.rows, rhs.cols);
    let data = &mut result.data.as_mut().unwrap();
    for i in 0..lhs.rows {
        for j in 0..rhs.cols {
            for k in 0..lhs.cols {
                unsafe {*data.get_data_mut_unchecked(i * data.cols + j) += 
                    *lhs.get_data_mut_unchecked(i * lhs.cols + k) * *rhs.get_data_mut_unchecked(k * rhs.cols + j)}
            }
        }
    }
    result
}
fn pointwisemul<T: Numerical<T>>(lhs: &DenseData<T>, rhs: &DenseData<T>) -> Dense<T> {
    if lhs.rows != rhs.rows || lhs.cols != rhs.cols {
        return Dense::err(Error::InvalidDimensions((lhs.rows, lhs.cols), (rhs.rows, rhs.cols)));
    }
    let mut result = Dense::new(lhs.rows, lhs.cols);
    let data = &mut result.data.as_mut().unwrap();
    for i in 0..(lhs.rows * lhs.cols) {
        unsafe {*data.get_data_mut_unchecked(i) = *lhs.get_data_mut_unchecked(i) * *rhs.get_data_mut_unchecked(i)}
    }
    result
}
impl<T: Numerical<T>> Mul<Dense<T>> for Dense<T> {
    type Output = Dense<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let Ok(lhs) = self.data else {return self};
        let Ok(rhs) = rhs.data else {return rhs};
        if lhs.pointwise || rhs.pointwise {
            pointwisemul(&lhs, &rhs)
        } else {
            matmul(&lhs, &rhs)
        }
    }
}
// Add
fn matadd<T: Numerical<T>>(lhs: &DenseData<T>, rhs: &DenseData<T>) -> Dense<T> {
    if lhs.rows != rhs.rows || lhs.cols != rhs.cols {
        return Dense::err(Error::InvalidDimensions((lhs.rows, lhs.cols), (rhs.rows, rhs.cols)));
    }
    let mut result = Dense::new(lhs.rows, lhs.cols);
    let data = &mut result.data.as_mut().unwrap();
    for i in 0..(lhs.rows * lhs.cols) {
        unsafe {*data.get_data_mut_unchecked(i) = *lhs.get_data_mut_unchecked(i) + *rhs.get_data_mut_unchecked(i)}
    }
    result
}
impl<T: Numerical<T>> Add<Dense<T>> for Dense<T> {
    type Output = Dense<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let Ok(lhs) = self.data else {return self};
        let Ok(rhs) = rhs.data else {return rhs};
        matadd(&lhs, &rhs)
    }
}
// Sub
fn matsub<T: Numerical<T>>(lhs: &DenseData<T>, rhs: &DenseData<T>) -> Dense<T> {
    if lhs.rows != rhs.rows || lhs.cols != rhs.cols {
        return Dense::err(Error::InvalidDimensions((lhs.rows, lhs.cols), (rhs.rows, rhs.cols)));
    }
    let mut result = Dense::new(lhs.rows, lhs.cols);
    let data = &mut result.data.as_mut().unwrap();
    for i in 0..(lhs.rows * lhs.cols) {
        unsafe {*data.get_data_mut_unchecked(i) = *lhs.get_data_mut_unchecked(i) - *rhs.get_data_mut_unchecked(i)}
    }
    result
}
impl<T: Numerical<T>> Sub<Dense<T>> for Dense<T> {
    type Output = Dense<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let Ok(lhs) = self.data else {return self};
        let Ok(rhs) = rhs.data else {return rhs};
        matsub(&lhs, &rhs)
    }
}

impl<T: Numerical<T>> Index<(usize, usize)> for DenseData<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < self.rows, "Row index out of bounds");
        assert!(index.1 < self.cols, "Column index out of bounds");
        unsafe {self.values.as_ptr().add(index.0 * self.cols + index.1).as_ref().unwrap()}
    }
}
impl<T: Numerical<T>> IndexMut<(usize, usize)> for DenseData<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(index.0 < self.rows, "Row index out of bounds");
        assert!(index.1 < self.cols, "Column index out of bounds");
        unsafe {self.values.as_ptr().add(index.0 * self.cols + index.1).as_mut().unwrap()}
    }
}
impl<T: Numerical<T>> Index<(usize, usize)> for Dense<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data.as_ref().expect("Cannot index into an error matrix")[index]
    }
}
impl<T: Numerical<T>> IndexMut<(usize, usize)> for Dense<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data.as_mut().expect("Cannot index into an error matrix")[index]
    }
}
impl<T: Numerical<T>> Matrix<T, Error> for Dense<T> {
    fn new(rows: usize, cols: usize) -> Self {
        let data = allocate::<T>(rows, cols);
        for i in 0..(rows * cols) {
            unsafe {data.as_ptr().add(i).write(std::mem::zeroed())}
        }
        Self { data: Ok(DenseData::from_values(rows, cols, data)) }
    }
    fn err(error: Error) -> Self {
        Self { data: Err(error) }
    }
    fn is_ok(&self) -> bool {
        self.data.is_ok()
    }
    fn is_err(&self) -> bool {
        self.data.is_err()
    }
    fn dup(&self) -> Self {
        let orig_data = match &self.data {
            Ok(data) => data,
            Err(err) => return Self::err(err.clone())
        };
        let mut result = Self::new(orig_data.rows, orig_data.cols);
        let result_data = &mut result.data.as_mut().unwrap();
        unsafe {copymemory(&orig_data.values, &mut result_data.values, 0..(orig_data.rows * orig_data.cols), 0)};
        result
    }
    fn get<U: AsIndex>(&self, rows: U, cols: U) -> Self {
        let orig_data = match &self.data {
            Ok(data) => data,
            Err(err) => return Self::err(err.clone())
        };
        let ((row_start, row_end), (col_start, col_end)) = orig_data.get_bounds(rows, cols);
        let mut result = Self::new(row_end - row_start, col_end - col_start);
        let result_data = &mut result.data.as_mut().unwrap();
        for i in row_start..row_end {
            unsafe {copymemory(&orig_data.values, &mut result_data.values, (i * orig_data.cols + col_start)..(i * orig_data.cols + col_end), 
                (i - row_start) * result_data.cols)}
        }
        result
    }
    fn scale(mut self, scale: T) -> Self {
        let Ok(data) = &mut self.data else {return self};
        for i in 0..(data.rows * data.cols) {
            unsafe {*data.get_data_mut_unchecked(i) *= scale};
        }
        self
    }
    fn shift(mut self, shift: T) -> Self {
        let Ok(data) = &mut self.data else {return self};
        for i in 0..(data.rows * data.cols) {
            unsafe {*data.get_data_mut_unchecked(i) += shift};
        }
        self
    }
    fn p(mut self) -> Self {
        let Ok(data) = &mut self.data else {return self};
        data.pointwise = true;
        self
    }
    fn t(mut self) -> Self {
        let Ok(data) = &mut self.data else {return self};
        data.transposed = !data.transposed;
        for i in 0..data.rows - 1 {
            for j in i + 1..data.cols {
                unsafe {std::mem::swap(data.get_data_mut_unchecked(i * data.cols + j), 
                    data.get_data_mut_unchecked(j * data.cols + i))};
            }
        }
        std::mem::swap(&mut data.rows, &mut data.cols);
        self
    }
}

#[macro_export]
macro_rules! mat {
    ($rows:expr, $cols:expr; [$t:ty] = $($x:expr), +) => {
        {
            let mut result = crate::structs::dense::matrix::Dense::new($rows, $cols);
            let result_data = &mut result.data.as_mut().unwrap();
            let mut data: [$t; $cols * $rows] = [$($x),+];
            unsafe {crate::helpers::memory::copymemory(&std::ptr::NonNull::new(data.as_mut_ptr()).unwrap(), &mut result_data.values, 0..data.len(), 0)}
            result
        }
    };
}