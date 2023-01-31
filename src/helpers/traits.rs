use std::ops::{Range, RangeFull, RangeInclusive, RangeFrom, RangeTo, RangeToInclusive};

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