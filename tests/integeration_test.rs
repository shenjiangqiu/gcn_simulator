// mod common;
// use std::vec;

// use chrono::Local;
// use gcn_agg::{
//     accelerator::System,
//     gcn_result::GcnAggResult,
//     graph::Graph,
//     node_features::NodeFeatures,
//     settings::{RunningMode, Settings}, utils,
// };
// use itertools::Itertools;

// #[test]
// fn test_system_sparse() -> Result<(), Box<dyn std::error::Error>> {
//     std::fs::create_dir_all("output")?;

//     utils::init_log();
//     let current_time: String = Local::now().format("%Y-%m-%d-%H-%M-%S%.6f").to_string();

//     let start_time = std::time::Instant::now();
//     let mut results = GcnAggResult::default();

//     let settings = Settings::new(vec!["configs/default.toml".into()]).unwrap();
//     results.settings = Some(settings.clone());
//     // create the folder for output
//     std::fs::create_dir_all("output")?;

//     let graph_name = &settings.graph_path;
//     let features_name = &settings.features_paths;

//     let graph = Graph::new(graph_name.as_str())?;
//     let node_features: Vec<_> = match settings.accelerator_settings.running_mode {
//         RunningMode::Sparse | RunningMode::Mixed => features_name
//             .iter()
//             .map(|x| NodeFeatures::new(x.as_str()))
//             .try_collect()?,
//         RunningMode::Dense => {
//             vec![]
//         }
//     };

//     let mem_stat_path = format!("output/{}_mem_stat.txt", current_time);
//     let mut system = System::new(
//         &graph,
//         &node_features,
//         settings.accelerator_settings,
//         &mem_stat_path,
//     );

//     // run the system
//     let mut stat = system.run()?;

//     // record the simulation time
//     let simulation_time = start_time.elapsed().as_secs();
//     // record the result
//     let seconds = simulation_time % 60;
//     let minutes = (simulation_time / 60) % 60;
//     let hours = (simulation_time / 60) / 60;
//     let time_str = format!("{}:{}:{}", hours, minutes, seconds);
//     stat.simulation_time = time_str;

//     results.stats = Some(stat);
//     let output_path = format!("output/{}.json", current_time);

//     println!("{}", serde_json::to_string_pretty(&results)?);
//     // write json of results to output_path
//     std::fs::write(output_path, serde_json::to_string_pretty(&results)?)?;
//     Ok(())
// }
// #[test]
// fn test_system_dense() -> Result<(), Box<dyn std::error::Error>> {
//     std::fs::create_dir_all("output")?;
//     let current_time: String = Local::now().format("%Y-%m-%d-%H-%M-%S%.6f").to_string();

//     utils::init_log();

//     let start_time = std::time::Instant::now();
//     let mut results = GcnAggResult::default();

//     let settings = Settings::new(vec![
//         "configs/default.toml".into(),
//         "configs/optional_configs/dense.toml".into(),
//     ])
//     .unwrap();
//     results.settings = Some(settings.clone());
//     // create the folder for output
//     std::fs::create_dir_all("output")?;

//     let graph_name = &settings.graph_path;
//     let features_name = &settings.features_paths;

//     let graph = Graph::new(graph_name.as_str())?;

//     let node_features: Vec<_> = match settings.accelerator_settings.running_mode {
//         RunningMode::Sparse | RunningMode::Mixed => features_name
//             .iter()
//             .map(|x| NodeFeatures::new(x.as_str()))
//             .try_collect()?,
//         RunningMode::Dense => {
//             vec![]
//         }
//     };
//     assert!(node_features.is_empty());
//     let mem_stat_path = format!("output/{}_mem_stat.txt", current_time);
//     let mut system = System::new(
//         &graph,
//         &node_features,
//         settings.accelerator_settings,
//         &mem_stat_path,
//     );

//     // run the system
    
//     let mut stat = system.run()?;

//     // record the simulation time
//     let simulation_time = start_time.elapsed().as_secs();
//     // record the result
//     let seconds = simulation_time % 60;
//     let minutes = (simulation_time / 60) % 60;
//     let hours = (simulation_time / 60) / 60;
//     let time_str = format!("{}:{}:{}", hours, minutes, seconds);
//     stat.simulation_time = time_str;

//     results.stats = Some(stat);
//     let output_path = format!("output/{}.json", current_time);

//     println!("{}", serde_json::to_string_pretty(&results)?);
//     // write json of results to output_path
//     std::fs::write(output_path, serde_json::to_string_pretty(&results)?)?;
//     Ok(())
// }


// #[test]
// fn test_system_mixed() -> Result<(), Box<dyn std::error::Error>> {
//     std::fs::create_dir_all("output")?;
//     let current_time: String = Local::now().format("%Y-%m-%d-%H-%M-%S%.6f").to_string();

//     utils::init_log();

//     let start_time = std::time::Instant::now();
//     let mut results = GcnAggResult::default();

//     let settings = Settings::new(vec![
//         "configs/default.toml".into(),
//         "configs/optional_configs/mixed.toml".into(),
//     ])
//     .unwrap();
//     results.settings = Some(settings.clone());
//     // create the folder for output
//     std::fs::create_dir_all("output")?;

//     let graph_name = &settings.graph_path;
//     let features_name = &settings.features_paths;

//     let graph = Graph::new(graph_name.as_str())?;

//     let node_features: Vec<_> = match settings.accelerator_settings.running_mode {
//         RunningMode::Sparse | RunningMode::Mixed => features_name
//             .iter()
//             .map(|x| NodeFeatures::new(x.as_str()))
//             .try_collect()?,
//         RunningMode::Dense => {
//             vec![]
//         }
//     };
//     assert!(!node_features.is_empty());
//     let mem_stat_path = format!("output/{}_mem_stat.txt", current_time);
//     let mut system = System::new(
//         &graph,
//         &node_features,
//         settings.accelerator_settings,
//         &mem_stat_path,
//     );

//     // run the system
//     let mut stat = system.run()?;

//     // record the simulation time
//     let simulation_time = start_time.elapsed().as_secs();
//     // record the result
//     let seconds = simulation_time % 60;
//     let minutes = (simulation_time / 60) % 60;
//     let hours = (simulation_time / 60) / 60;
//     let time_str = format!("{}:{}:{}", hours, minutes, seconds);
//     stat.simulation_time = time_str;

//     results.stats = Some(stat);
//     let output_path = format!("output/{}.json", current_time);

//     println!("{}", serde_json::to_string_pretty(&results)?);
//     // write json of results to output_path
//     std::fs::write(output_path, serde_json::to_string_pretty(&results)?)?;
//     Ok(())
// }
