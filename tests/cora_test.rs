use chrono::Local;
use clap::{Command, CommandFactory, Parser};
use clap_complete::{generate, Generator};
use gcn_agg::{
    accelerator::{
        agg_buffer::SparseAggBuffer,
        input_buffer::InputBuffer,
        mem_interface::{MemInterface, MemInterfaceReq},
        mlp_buffer::MlpBuffer,
        output_buffer::OutputBuffer,
        sliding_window::{InputWindow, OutputWindow, OutputWindowIterator, WindowIterSettings},
        sparsify_buffer::SparsifyBuffer,
    },
    cmd_args::Args,
    settings::{AcceleratorSettings, RunningMode, Settings},
    utils, GcnAggResult, GcnStatistics, Graph, NodeFeatures,
};
use itertools::Itertools;
use pipe_sim::{AggBuffer, Buffer, DoubleBuffer};
use ramulator_wrapper::RamulatorWrapper;
use std::{error::Error, io, rc::Rc};

fn print_completions<G: Generator>(gen: G, cmd: &mut Command) {
    generate(gen, cmd, cmd.get_name().to_string(), &mut io::stdout());
}
fn get_addr_vec_from_window<'a>(
    window: InputWindow<'a>,
    running_mode: &RunningMode,
    node_features: &NodeFeatures,
    layer_id: usize,
) -> MemInterfaceReq<'a> {
    let mut addr_vec = vec![];
    match running_mode {
        RunningMode::Sparse | RunningMode::Mixed => {
            let start_addrs = &node_features.start_addrs;
            let mut start_addr = start_addrs[window.start_input_index];
            let end_addr = start_addrs[window.end_input_index];
            // round start_addr to the nearest 64
            start_addr = start_addr / 64 * 64;
            while start_addr < end_addr {
                addr_vec.push(start_addr);
                start_addr += 64;
            }
        }
        RunningMode::Dense => {
            // dense
            let base_addr: u64 = layer_id as u64 * 0x100000000;
            let mut start_addr = base_addr
                + window.start_input_index as u64
                    * window.get_output_window().get_input_dim() as u64
                    * 4;
            let end_addr = base_addr
                + window.end_input_index as u64
                    * window.get_output_window().get_input_dim() as u64
                    * 4;
            while start_addr < end_addr {
                addr_vec.push(start_addr);
                start_addr += 64;
            }
        }
    }
    MemInterfaceReq { addr_vec, window }
}

fn run_with_system<'a, T: Buffer<Input = MemInterfaceReq<'a>, Output = Rc<OutputWindow>>>(
    mut system: T,
    settings: Settings,
    node_features_vec: &'a [NodeFeatures],
    graph: &'a Graph,
) -> Result<u64, Box<dyn Error>> {
    let mut cycles = 0;
    let acc_settings = settings.accelerator_settings;
    let AcceleratorSettings {
        input_buffer_size,
        agg_buffer_size,
        gcn_hidden_size,
        aggregator_settings: _,
        mlp_settings: _,
        sparsifier_settings: _,
        // output_buffer_size,
        running_mode,
        mem_config_name: _,
        gcn_layers,
    } = acc_settings;
    for (layer_id, node_features) in node_features_vec.iter().enumerate() {
        let window_iter_settings = WindowIterSettings {
            agg_buffer_size,
            input_buffer_size,
            gcn_hidden_size: gcn_hidden_size.clone(),
            running_mode: running_mode.clone(),
            layer: 0,
            is_final_layer: layer_id == gcn_layers - 1,
        };
        let output_window_iter =
            OutputWindowIterator::new(graph, Some(node_features), window_iter_settings);
        for input_window_iter in output_window_iter {
            for window in input_window_iter {
                // first build addr vec
                let mem_req =
                    get_addr_vec_from_window(window, &running_mode, node_features, layer_id);
                system.push_input(mem_req);

                loop {
                    system.cycle();
                    cycles += 1;
                    if system.input_avaliable() {
                        break;
                    }
                    if let Some(out) = system.pop_output() {
                        // should already jump out
                        debug_assert!(!out.is_final_window);
                    }
                }
            }
            // finished this input iter, need to change to the next input iter
            // need to empty the temp agg result, but it's done automatically
        }
        // finished this layer, need to change to the next layer,
        // before change to the next layer, need to make sure that the last output window is finished
        loop {
            system.cycle();
            cycles += 1;
            if let Some(out) = system.pop_output() {
                if out.is_final_window {
                    break;
                }
            }
        }
    }

    Ok(cycles)
}
#[test]
fn test_cora() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_log();
    let start_time = std::time::Instant::now();
    let current_time: String = Local::now().format("%Y-%m-%d-%H-%M-%S%.6f").to_string();
    let mut stat = GcnStatistics::new();
    let mut results = GcnAggResult::new();

    let config_names = vec![
        "configs/default.toml".into(),
        "configs/user_configs/cora.toml".into(),
    ];

    let settings = Settings::new(config_names)?;
    results.settings = Some(settings.clone());

    println!("{}", serde_json::to_string_pretty(&settings)?);
    // create the folder for output
    std::fs::create_dir_all("output")?;

    let graph_name = &settings.graph_path;
    let features_name = &settings.features_paths;

    let graph = Graph::new(graph_name.as_str())?;

    let running_mode = &settings.accelerator_settings.running_mode;

    let node_features_vec: Vec<_> = match running_mode {
        RunningMode::Sparse | RunningMode::Mixed => features_name
            .iter()
            .map(|x| NodeFeatures::new(x.as_str()))
            .try_collect()?,
        RunningMode::Dense => {
            //dense mode, no features
            vec![]
        }
    };

    let stats_name = format!("output/{}_mem_stat.txt", current_time);

    let mut ramulator =
        RamulatorWrapper::new(&settings.accelerator_settings.mem_config_name, &stats_name);
    let mem_interface = MemInterface::new(&mut ramulator, 16, false);
    let input_buffer = InputBuffer::new();
    let output_buffer = OutputBuffer::new();
    let mut sparse_agg_cycle = 0;
    let sparse_agg_buffer = SparseAggBuffer::new(
        settings
            .accelerator_settings
            .aggregator_settings
            .sparse_cores,
        &node_features_vec,
        &mut sparse_agg_cycle,
    );

    let dense_agg_buffer = AggBuffer::new(
        |x: InputWindow| x.output_window,
        |old, new| {
            *old = new.clone();
        },
        |x| x.last_row_completed,
    );
    let translation_buffer = DoubleBuffer::new();
    let mlp_buffer = MlpBuffer::new();
    let sparsify_buffer = SparsifyBuffer::new();
    let mlp_sparse_cores = settings.accelerator_settings.mlp_settings.mlp_sparse_cores;
    let mut mlp_sparse_cycles = 0;
    let calculate_mlp_cycle_sparse =
        |x: Option<&(Rc<OutputWindow>, Vec<Vec<usize>>)>, _y: Option<&Rc<OutputWindow>>| {
            let x = x.unwrap();
            let output_results = &x.1;
            let output_window = &x.0;
            let total_add = output_results.iter().fold(0, |acc, x| acc + x.len());
            let mut total_cycle = total_add * output_window.get_output_dim() / mlp_sparse_cores;
            mlp_sparse_cycles += total_cycle as u64;
            total_cycle *= 2;
            total_cycle as u64
        };
    let sparsifier_cols = settings
        .accelerator_settings
        .sparsifier_settings
        .sparsifier_cols;
    let sparsifier_width = settings
        .accelerator_settings
        .sparsifier_settings
        .sparsifier_width;
    let mut sparsify_cycle = 0u64;
    let calculate_sparsify_cycle = |x: Option<&Rc<OutputWindow>>, _y: Option<&Rc<OutputWindow>>| {
        let mut total_cycles = 0;
        // calculate the number of cycles needed to finish the mlp
        let output_window = x.unwrap();
        let num_nodes = output_window.get_output_len();
        let output_node_dim = output_window.output_node_dim;
        let input_node_dim = output_node_dim;
        let steps = (sparsifier_width + num_nodes - 1) / sparsifier_width;
        let elements_steps = (output_node_dim + sparsifier_cols - 1) / sparsifier_cols;
        for _i in 0..steps {
            for _j in 0..elements_steps - 1 {
                total_cycles += sparsifier_width + sparsifier_cols + input_node_dim;
                total_cycles += sparsifier_width * sparsifier_width / 4 / 32;
            }
        }
        sparsify_cycle += total_cycles as u64;
        total_cycles as u64
    };
    let dense_width = settings
        .accelerator_settings
        .aggregator_settings
        .dense_width;
    let dense_cores = settings
        .accelerator_settings
        .aggregator_settings
        .dense_cores;
    let mut dense_agg_cycle = 0;
    let calulate_agg_dense = |x: Option<&InputWindow>, _y: Option<&InputWindow>| {
        // dense aggregation
        let mut cycles = 0;
        let task = x.unwrap();
        let num_add = task
            .get_tasks()
            .iter()
            .fold(0, |acc, x| acc + x.clone().count());
        cycles += (num_add * task.get_output_window().get_input_dim() / (dense_width * dense_cores))
            as u64;
        // extra cycle for load data
        cycles *= 2;
        dense_agg_cycle += cycles;
        cycles as u64
    };
    let systolic_rows = settings.accelerator_settings.mlp_settings.systolic_rows;
    let systolic_cols = settings.accelerator_settings.mlp_settings.systolic_cols;
    let mut dense_mlp_cycle = 0u64;
    let calculate_mlp_dense = |x: Option<&Rc<OutputWindow>>, _y: Option<&Rc<OutputWindow>>| {
        let mut total_cycles = 0;
        // calculate the number of cycles needed to finish the mlp
        let output_window = x.unwrap();
        let num_nodes = output_window.get_output_len();
        let output_node_dim = output_window.output_node_dim;
        let input_node_dim = output_window.input_node_dim;
        let steps = (systolic_rows + num_nodes - 1) / systolic_rows;
        let elements_steps = (output_node_dim + systolic_cols - 1) / systolic_cols;
        for _i in 0..steps {
            for _j in 0..elements_steps - 1 {
                total_cycles += systolic_rows + systolic_cols + input_node_dim;
                total_cycles += systolic_rows * systolic_rows / 4 / 32;
            }
        }
        dense_mlp_cycle += total_cycles as u64;
        total_cycles as u64
    };
    let mut translation_cycle = 0;
    let calculate_translation_cycle = |x: Option<&InputWindow>, _y: Option<&InputWindow>| {
        let window = x.unwrap();
        let mut cycle = 0;
        let tasks = window.get_tasks();
        for task in tasks.iter() {
            cycle += task.clone().count() / 16;
        }
        translation_cycle += cycle as u64;
        cycle as u64
    };
    let cycle = match running_mode {
        RunningMode::Sparse => {
            let system = mem_interface
                // .output_connect_input(translation_buffer, |x, y| 1)
                .connect(input_buffer)
                .connect(sparse_agg_buffer)
                .connect_with_fn(mlp_buffer, calculate_mlp_cycle_sparse)
                .connect_with_fn(sparsify_buffer, calculate_sparsify_cycle)
                .connect(output_buffer);
            let cycles = run_with_system(system, settings, &node_features_vec, &graph)?;
            cycles
        }
        RunningMode::Dense => {
            let system = mem_interface
                // .output_connect_input(translation_buffer, |x, y| 1)
                .connect(input_buffer)
                .connect_with_fn(dense_agg_buffer, calulate_agg_dense)
                .connect_with_fn(mlp_buffer, calculate_mlp_dense)
                // .connect_with_fn(sparsify_buffer, |_,_| 1)
                .connect(output_buffer);
            run_with_system(system, settings, &node_features_vec, &graph)?
        }
        RunningMode::Mixed => {
            let system = mem_interface
                .connect(input_buffer)
                .connect_with_fn(translation_buffer, calculate_translation_cycle)
                .connect(sparse_agg_buffer)
                .connect_with_fn(mlp_buffer, calculate_mlp_cycle_sparse)
                .connect_with_fn(sparsify_buffer, calculate_sparsify_cycle)
                .connect(output_buffer);
            run_with_system(system, settings, &node_features_vec, &graph)?
        }
    };

    // record the simulation time
    let simulation_time = start_time.elapsed().as_secs();
    // record the result
    let seconds = simulation_time % 60;
    let minutes = (simulation_time / 60) % 60;
    let hours = (simulation_time / 60) / 60;
    let time_str = format!("{}:{}:{}", hours, minutes, seconds);
    stat.simulation_time = time_str;
    stat.cycle = cycle;
    stat.dense_agg_cycle = dense_agg_cycle;
    stat.dense_mlp_cycle = dense_mlp_cycle;
    stat.sparsify_cycle = sparsify_cycle;
    stat.translation_cycle = translation_cycle;
    stat.sparse_mlp_cycle = mlp_sparse_cycles;
    stat.sparse_agg_cycle = sparse_agg_cycle;
    results.stats = Some(stat);
    let output_path = format!("output/{}.json", current_time);

    println!("{}", serde_json::to_string_pretty(&results)?);
    // write json of results to output_path
    std::fs::write(output_path, serde_json::to_string_pretty(&results)?)?;
    Ok(())
    // run the system
}
