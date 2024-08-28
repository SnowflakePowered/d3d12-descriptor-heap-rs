#![cfg(target_os = "windows")]
#![forbid(missing_docs)]

//! A simple-to-use descriptor heap allocator for Direct3D 12.
//!
//! Heaps are lock-free and thread-safe. On creation, heaps can be
//! partitioned into heaps of equal size if needed.
mod descriptor_heap;

pub use descriptor_heap::*;
