use std::{ptr::NonNull, alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error}, ops::Range};

pub fn allocate<T>(rows: usize, cols: usize) -> NonNull<T> {
    assert!(rows * cols > 0, "Minimal dimension is 1x1");
    let layout = Layout::array::<T>(rows * cols).expect("Capacity overflow");
    assert!(!(usize::BITS < 64 && layout.size() > isize::MAX as usize), "Allocation too large");
    
    let ptr = unsafe {alloc_zeroed(layout)};

    match NonNull::new(ptr as *mut T) {
        Some(data) => data,
        None => handle_alloc_error(layout),
    }
}

pub fn deallocate<T>(data: NonNull<T>, rows: usize, cols: usize) {
    let layout = Layout::array::<T>(rows * cols).expect("Capacity overflow");
    unsafe {dealloc(data.as_ptr() as *mut u8, layout)}
}

///
/// # Safety
/// 
/// Source size must be larger or equal to the range size and the destination size must be larger or equal to the range end.
/// 
pub unsafe fn copymemory<T>(src: &NonNull<T>, dst: &mut NonNull<T>, range: Range<usize>, offset: usize) {
    let start_ptr = src.as_ptr().add(range.start);
    std::ptr::copy_nonoverlapping(start_ptr, dst.as_ptr().add(offset), range.end - range.start);
}