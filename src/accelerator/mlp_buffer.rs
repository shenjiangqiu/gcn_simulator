use std::rc::Rc;

use pipe_sim::DoubleBuffer;

use super::sliding_window::OutputWindow;

pub type MlpBuffer = DoubleBuffer<Rc<OutputWindow>>;

