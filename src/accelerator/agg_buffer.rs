use std::{collections::VecDeque, mem::swap, rc::Rc};

use pipe_sim::Buffer;
use std::{
    collections::{btree_set::Range, HashSet},
    vec,
};

use crate::NodeFeatures;

use super::sliding_window::{InputWindow, OutputWindow};

#[derive(enum_as_inner::EnumAsInner, Debug)]
enum Status {
    //remaining cycle, is the last one
    Working(u64, bool),
    Done,
}
use Status::*;
pub struct SparseAggBuffer<'a> {
    // the input interface, size:1
    pub input_queue: VecDeque<InputWindow<'a>>,
    // current working queue, size:1
    pub data_queue: VecDeque<Rc<OutputWindow>>,
    // output interface, size:1
    pub output_queue: VecDeque<(Rc<OutputWindow>, Vec<Vec<usize>>)>,
    pub temp_agg_result: Vec<Vec<usize>>,
    status: Status,
    sparse_cores: usize,
    node_features: &'a [NodeFeatures],
    is_first_row: bool,
    pub total_cycle: &'a mut u64,
    // output_window_size: usize,
}

impl<'a> SparseAggBuffer<'a> {
    ///
    /// # Arguments
    /// * `tasks` - The list of edges to be aggregated
    /// each line is a set of edges that need to be aggregated to taget node, the taget node is returned
    /// echo line is like:
    /// 1 2 3 4
    /// 5 6 7 8
    /// 9 10 11 12
    ///
    /// which mean node 1,2,3,4 will be aggregated to the first node,
    /// node 5,6,7,8 will be aggregated to the second node,
    /// and node 9,10,11,12 will be aggregated to the third node
    /// * node_features - the node features is sparse format, each line is a node, each column is a feature index
    ///
    /// # Return
    /// (the cycles to calculate each node, the node features of result nodes)
    ///
    /// # Example
    /// ```ignore
    /// use gcn_agg::accelerator::aggregator::Aggregator;
    /// let node_features = vec![
    ///  vec![0,4,9],
    ///  vec![1,5,10],
    ///  vec![2],
    /// ];
    /// let tasks = vec![
    /// vec![0,1],
    /// vec![1,2],
    /// ];
    /// let num_sparse_cores = 2;
    /// let num_sparse_width = 2;
    /// let aggregator = Aggregator::new(2,2,2,2);
    /// ```
    ///
    ///
    pub fn get_add_sparse_cycle(&mut self, tasks: Vec<Range<usize>>, current_layer: usize) -> u64 {
        // each task's cycles
        let mut cycle_vec = Vec::new();
        let node_features = self.node_features.get(current_layer).unwrap();
        for (task, output_vec) in tasks.into_iter().zip(self.temp_agg_result.iter_mut()) {
            let mut cycles = 0;
            // type 1, simplely add the features one by one
            let mut temp_set: HashSet<usize> = output_vec.iter().cloned().collect();

            for &i in task {
                cycles += temp_set.len() + node_features.get_features(i).len();
                for &j in node_features.get_features(i) {
                    temp_set.insert(j);
                }
            }
            output_vec.clear();
            output_vec.append(&mut temp_set.into_iter().collect());

            cycle_vec.push(cycles);
        }

        // each cores current cycles, always push task to the core with the least cycles
        let mut core_cycles = vec![0; self.sparse_cores];
        cycle_vec.into_iter().for_each(|i| {
            core_cycles.sort_unstable();
            core_cycles[0] += i;
        });
        core_cycles.sort_unstable();
        let cycles = *core_cycles.last().unwrap_or(&0);

        cycles as u64
    }

    pub fn new(
        // output_window_size: usize,
        sparse_cores: usize,
        node_features: &'a [NodeFeatures],
        total_cycle: &'a mut u64,
    ) -> Self {
        let input_queue = VecDeque::with_capacity(1);
        let data_queue = VecDeque::with_capacity(1);
        let output_queue = VecDeque::with_capacity(1);
        let temp_agg_result = vec![];
        let status = Working(0, false);
        SparseAggBuffer {
            input_queue,
            data_queue,
            output_queue,
            temp_agg_result,
            status,
            sparse_cores,
            node_features,
            is_first_row: true,
            total_cycle,
            // output_window_size,
        }
    }
}

impl<'a> Buffer for SparseAggBuffer<'a> {
    type Output = Rc<OutputWindow>;
    type Input = InputWindow<'a>;

    type InputInfo = Vec<Vec<usize>>;
    // output window and temp agg result
    type OutputInfo = (Rc<OutputWindow>, Vec<Vec<usize>>);

    fn input_avaliable(&self) -> bool {
        self.input_queue.is_empty()
    }

    fn output_avaliable(&self) -> bool {
        !self.output_queue.is_empty()
    }

    fn pop_output(&mut self) -> Option<Self::Output> {
        self.output_queue.pop_front().map(|x| x.0)
    }

    fn get_output(&self) -> Option<&Self::Output> {
        self.output_queue.front().map(|x| &x.0)
    }

    fn push_input(&mut self, input: Self::Input) {
        self.input_queue.push_back(input);
    }

    fn get_input_info(&self) -> Option<&Self::InputInfo> {
        Some(&self.temp_agg_result)
    }

    fn get_output_info(&self) -> Option<&Self::OutputInfo> {
        self.output_queue.front()
    }

    fn cycle(&mut self) {
        match self.status {
            Working(cycles, is_last) => {
                if cycles == 0 {
                    if is_last {
                        // the last one finished, reset the current temp_agg_result

                        self.status = Done;
                    } else if let Some(window) = self.input_queue.pop_front() {
                        // it's not the last, try to insert a new one
                        // first calculate the cycle
                        let tasks = window.get_tasks().clone();
                        if self.is_first_row {
                            debug_assert!(self.temp_agg_result.is_empty());
                            self.temp_agg_result.resize(tasks.len(), vec![]);
                            self.is_first_row = false;
                        }
                        let current_layer = window.get_task_id().layer_id;
                        let cycles = self.get_add_sparse_cycle(tasks, current_layer);
                        *self.total_cycle += cycles;
                        let is_last = window.is_last_row;
                        self.data_queue
                            .push_back(window.get_output_window().clone());
                        self.status = Working(cycles, is_last);
                    }
                } else {
                    let cycles = cycles - 1;
                    self.status = Working(cycles, is_last);
                }
            }
            Done => {
                // try to move the finished result to output queue
                let mut new_result = vec![];
                swap(&mut self.temp_agg_result, &mut new_result);
                let output_window = self.data_queue.pop_front().unwrap();
                self.output_queue.push_back((output_window, new_result));
                self.status = Working(0, false);
                self.is_first_row = true;
            }
        };
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
    fn test() {
        let graph = Graph::new("graphs/test.graph").unwrap();
        let window_iter_setting = WindowIterSettings {
            agg_buffer_size: 60,
            gcn_hidden_size: vec![2, 2],
            input_buffer_size: 60,
            is_final_layer: false,
            layer: 0,
            running_mode: RunningMode::Sparse,
        };
        let node_features = NodeFeatures::new("nodefeatures/test_1.feat").unwrap();
        let node_features = [node_features];

        let mut output_window_iter =
            OutputWindowIterator::new(&graph, Some(&node_features[0]), window_iter_setting);
        let mut input_window_iter = output_window_iter.next().unwrap();
        let window = input_window_iter.next().unwrap();
        let mut total_cycle=0;
        let mut agg_buffer = SparseAggBuffer::new(2, &node_features,&mut total_cycle);
        println!("input: {:?}", window);
        agg_buffer.push_input(window);
        let mut cycles = 0;
        while !agg_buffer.output_avaliable() {
            agg_buffer.cycle();
            cycles += 1;
            if agg_buffer.input_avaliable() {
                if let Some(window) = input_window_iter.next() {
                    println!("input: {:?}", window);
                    agg_buffer.push_input(window);
                }
            }
        }

        let output = agg_buffer.pop_output().unwrap();
        println!("output: {:?}", output);
        println!("cycles: {}", cycles);
    }
}
