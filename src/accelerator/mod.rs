//! # Description
//! - this module is the accelerator module
//! - the main sub module is system, all the components are in system
//! - read system.rs for more details
//! 
//! # Components
//! - system: the main sub module, all the components are in system
//! - agg_buffer and other buffers: provide data for aggregator and mlp
//! - aggregator and mlp, the module for calculating the result
//! - mem_interface: the interface between system and memory(ramulator)
//! 


pub mod input_buffer;
pub mod sparsify_buffer;
pub mod agg_buffer;
pub mod sliding_window;
pub mod window_id;
pub mod output_buffer;
pub mod mlp_buffer;

pub mod mem_interface;
// pub use system::System;