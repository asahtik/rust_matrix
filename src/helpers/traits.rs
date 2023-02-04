use std::{ops::{Range, RangeFull, RangeInclusive, RangeFrom, RangeTo, RangeToInclusive, Mul, MulAssign, Add, Sub, Div, Index, IndexMut, AddAssign}, fmt::Display};

pub trait AsIndex {
    fn start(&self) -> Option<usize>;
    fn end(&self) -> Option<usize>;
}
impl AsIndex for usize {
    fn start(&self) -> Option<usize> {
        Some(*self)
    }
    fn end(&self) -> Option<usize> {
        Some(*self + 1)
    }
}
impl AsIndex for Range<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }
    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}
impl AsIndex for RangeFull {
    fn start(&self) -> Option<usize> {
        None
    }
    fn end(&self) -> Option<usize> {
        None
    }
}
impl AsIndex for RangeInclusive<usize> {
    fn start(&self) -> Option<usize> {
        Some(*self.start())
    }
    fn end(&self) -> Option<usize> {
        Some(*self.end() + 1)
    }
}
impl AsIndex for RangeFrom<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }
    fn end(&self) -> Option<usize> {
        None
    }
}
impl AsIndex for RangeTo<usize> {
    fn start(&self) -> Option<usize> {
        None
    }
    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}
impl AsIndex for RangeToInclusive<usize> {
    fn start(&self) -> Option<usize> {
        None
    }
    fn end(&self) -> Option<usize> {
        Some(self.end + 1)
    }
}

pub trait Numerical<T>: Mul<Output = T> + MulAssign + Add<Output = T> + AddAssign + Sub<Output = T> + Copy + Sized + Display {
    fn ident() -> Self;
    fn zero() -> Self;
}

impl<T: From<u8> + Mul<Output = T> + MulAssign + Add<Output = T> + AddAssign + Sub<Output = T> + Copy + Sized + Display> Numerical<T> for T {
    fn ident() -> Self {
        1.into()
    }
    fn zero() -> Self {
        0.into()
    }
}

pub trait Matrix<T: Numerical<T>, E>: Mul + Add + Sub + Index<(usize, usize)> + IndexMut<(usize, usize)> + Sized {
    fn new(rows: usize, cols: usize) -> Self;
    fn err(error: E) -> Self;
    
    fn is_ok(&self) -> bool;
    fn is_err(&self) -> bool;
    fn dup(&self) -> Self;
    
    fn get<U: AsIndex>(&self, rows: U, cols: U) -> Self;

    fn t(self) -> Self;
    fn p(self) -> Self;
    fn scale(self, scale: T) -> Self;
    fn shift(self, scale: T) -> Self;
}