use pipe_sim::DoubleBuffer;

use super::sliding_window::InputWindow;
pub type InputBuffer<'a> = DoubleBuffer<InputWindow<'a>>;

#[cfg(test)]
mod test {
    use pipe_sim::Buffer;
    use ramulator_wrapper::RamulatorWrapper;

    use super::*;
    use crate::{
        accelerator::{
            mem_interface::{MemInterface, MemInterfaceReq},
            sliding_window::{OutputWindowIterator, WindowIterSettings},
        },
        settings::RunningMode,
        Graph,
    };
    #[test]
    fn input_buffer_test() {
        let graph = Graph::new("graphs/test.graph").unwrap();
        let window_iter_setting = WindowIterSettings {
            agg_buffer_size: 60,
            gcn_hidden_size: vec![2, 2],
            input_buffer_size: 60,
            is_final_layer: false,
            layer: 0,
            running_mode: RunningMode::Dense,
        };
        let mut output_window_iter = OutputWindowIterator::new(&graph, None, window_iter_setting);
        let window = output_window_iter.next().unwrap().next().unwrap();

        let mut input_buffer = InputBuffer::new();
        input_buffer.push_input(window);
        let output = input_buffer.pop_output().unwrap();
        println!("output: {:?}", output);
    }
    #[test]
    fn connect_test() {
        let graph = Graph::new("graphs/test.graph").unwrap();
        let window_iter_setting = WindowIterSettings {
            agg_buffer_size: 60,
            gcn_hidden_size: vec![2, 2],
            input_buffer_size: 60,
            is_final_layer: false,
            layer: 0,
            running_mode: RunningMode::Dense,
        };
        let mut output_window_iter = OutputWindowIterator::new(&graph, None, window_iter_setting);
        let window = output_window_iter.next().unwrap().next().unwrap();
        let mem_interface_req = MemInterfaceReq {
            window,
            addr_vec: vec![0, 64, 128],
        };
        let input_buffer = InputBuffer::new();
        let mut mem = RamulatorWrapper::new("HBM-config.cfg", "output_connect_input.txt");

        let mem_interface = MemInterface::new(&mut mem, 2, false);
        let mut system = mem_interface.output_connect_input(input_buffer, |_, _| 1);
        system.push_input(mem_interface_req);
        let mut cycle = 0;
        while !system.output_avaliable() {
            system.cycle();
            cycle += 1;
        }
        let output = system.pop_output().unwrap();
        println!("output: {:?}", output);
        println!("cycle: {}", cycle);
    }
}
