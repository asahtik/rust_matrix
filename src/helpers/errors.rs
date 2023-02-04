use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("Invalid dimensions")]
    InvalidDimensions((usize, usize), (usize, usize)),
}