use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid dimensions")]
    InvalidDimensions((usize, usize), (usize, usize)),
}