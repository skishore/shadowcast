#![feature(int_roundings)]
#![feature(let_chains)]
#![feature(test)]

mod base;
mod shadowcast;

pub use base::{Matrix, Point};
pub use shadowcast::{INITIAL_VISIBILITY, VISIBILITY_LOSSES};
pub use shadowcast::{Vision, VisionArgs};
