use std::collections::{HashSet, VecDeque};

use log::debug;
use pipe_sim::*;
use ramulator_wrapper::RamulatorWrapper;

use super::sliding_window::InputWindow;

#[derive(Debug)]
enum TempMemReqState {
    // 1. waiting for send,2. waiting for recv
    Working(Vec<u64>, HashSet<u64>),
    Done,
}

use TempMemReqState::*;

pub struct MemInterfaceReq<'a> {
    pub addr_vec: Vec<u64>,
    pub window: InputWindow<'a>,
}

#[derive(Debug)]
pub struct TempMemReq<'a> {
    id: InputWindow<'a>,
    is_write: bool,
    state: TempMemReqState,
}

pub struct MemInterface<'a> {
    // public interface with size 1
    input_queue: VecDeque<MemInterfaceReq<'a>>,
    output_queue: VecDeque<InputWindow<'a>>,

    // internal buffer, size configurable
    mem: &'a mut RamulatorWrapper,
    // that is the current working queue
    send_queue: VecDeque<TempMemReq<'a>>,
    current_inflight_addr: HashSet<u64>,
    // usaully the size of send_queue is 2
    send_size: usize,

    is_write: bool,
}

impl<'a> MemInterface<'a> {
    pub fn new(mem: &'a mut RamulatorWrapper, send_size: usize, is_write: bool) -> Self {
        MemInterface {
            input_queue: VecDeque::new(),
            output_queue: VecDeque::new(),
            mem,
            send_queue: VecDeque::new(),
            current_inflight_addr: HashSet::new(),
            send_size,
            is_write,
        }
    }
}
impl<'a> Buffer for MemInterface<'a> {
    type Output = InputWindow<'a>;

    type Input = MemInterfaceReq<'a>;

    type InputInfo = ();

    type OutputInfo = InputWindow<'a>;

    fn input_avaliable(&self) -> bool {
        self.input_queue.is_empty()
    }

    fn output_avaliable(&self) -> bool {
        !self.output_queue.is_empty()
    }

    fn pop_output(&mut self) -> Option<Self::Output> {
        self.output_queue.pop_front()
    }

    fn get_output(&self) -> Option<&Self::Output> {
        self.output_queue.front()
    }

    fn push_input(&mut self, input: Self::Input) {
        self.input_queue.push_back(input);
    }

    fn get_input_info(&self) -> Option<&Self::InputInfo> {
        None
    }

    fn get_output_info(&self) -> Option<&Self::OutputInfo> {
        self.output_queue.front()
    }

    fn cycle(&mut self) {
        // put req to send_queue
        if self.send_queue.len() < self.send_size && !self.input_queue.is_empty() {
            let input = self.input_queue.pop_front().unwrap();
            // build the addr_vec
            let addr_vec = input.addr_vec;

            let req = TempMemReq {
                id: input.window,
                is_write: self.is_write,
                state: Working(addr_vec, HashSet::new()),
            };
            self.send_queue.push_back(req);
        }

        // check finished req in send queue
        if self.output_queue.is_empty() {
            match self.send_queue.front() {
                Some(TempMemReq { state, .. }) if matches!(state, Done) => {
                    let value = self.send_queue.pop_front().unwrap().id;
                    self.output_queue.push_back(value)
                }
                _ => {}
            }
        }

        if let Some(TempMemReq {
            id,
            is_write,
            state: Working(remaining, receiving),
        }) = self.send_queue.front_mut()
        {
            match is_write {
                true => {
                    while let Some(addr) = remaining.pop() {
                        if self.mem.available(addr, *is_write) {
                            debug!("addr: {} ready to send!", addr);
                            // fix bug here, should merge the same addr
                            self.mem.send(addr, *is_write);
                        } else {
                            debug!("addr: {} not ready to send!", addr);
                            remaining.push(addr);
                            break;
                        }
                    }
                }
                false => {
                    while let Some(addr) = remaining.pop() {
                        if addr % 64 != 0 {
                            panic!("addr should be 64 aligned");
                        }
                        debug!("trying to send addr: {} of req: {:?}", addr, id);

                        if self.current_inflight_addr.contains(&addr) {
                            debug!("addr: {} is already in current_waiting_mem_request", addr);
                            // the request is already in flight
                            // do not send it to memory
                            receiving.insert(addr);
                        } else if self.mem.available(addr, *is_write) {
                            debug!("addr: {} ready to send!", addr);
                            // fix bug here, should merge the same addr

                            self.current_inflight_addr.insert(addr);
                            receiving.insert(addr);

                            self.mem.send(addr, *is_write);
                        } else {
                            remaining.push(addr);
                            break;
                        }
                    }
                }
            }
        }
        // change the state if possible

        while self.mem.ret_available() {
            let addr = self.mem.pop();
            debug!("receive: addr: {:?}", addr);
            debug_assert!(self.current_inflight_addr.contains(&addr));
            self.current_inflight_addr.remove(&addr);
            for i in self.send_queue.iter_mut() {
                if let Working(remaining, receiving) = &mut i.state {
                    if receiving.contains(&addr) {
                        receiving.remove(&addr);
                        if receiving.is_empty() && remaining.is_empty() {
                            i.state = Done;
                        }
                    }
                }
            }
        }
        // for any other cases
        if let Some(req) = self.send_queue.front_mut() {
            if let Working(remaining, receiving) = &mut req.state {
                if remaining.is_empty() && receiving.is_empty() {
                    req.state = Done;
                }
            }
        }
        self.mem.cycle();
    }
}

#[cfg(test)]
mod test {
    use crate::{
        accelerator::sliding_window::{OutputWindowIterator, WindowIterSettings},
        settings::RunningMode,
        Graph,
    };

    use super::*;
    #[test]
    fn test_mem_interface() {
        let mut mem = RamulatorWrapper::new("HBM-config.cfg", "output_mem_interface.txt");
        let mut mem_interface = MemInterface::new(&mut mem, 2, false);
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

        mem_interface.push_input(mem_interface_req);
        let mut cycle=0;
        while !mem_interface.output_avaliable() {
            mem_interface.cycle();
            cycle+=1;
        }
        let output = mem_interface.pop_output().unwrap();
        println!("output: {:?}", output);
        println!("cycle: {}", cycle);
    }
}
