#![feature(slice_take)]

use clap::Parser;

pub mod game;
pub mod alpha_zero_model;
pub mod config;
pub mod neural_network;
pub mod mcts;
pub mod coach;
pub mod arena;
pub mod utils;
pub mod canonical_board;


#[derive(Parser, Debug, Clone, Default)]
#[clap(name = "battlesnake alphazero", version = "1.0", author = "Canarit")]
pub struct Args {
    #[arg(long, default_value_t = 1000)]
    pub num_iterations: i32,

    #[arg(long, default_value_t = 50)]
    pub num_episodes: i32,

    #[arg(long, default_value_t = 200000)]
    pub max_queue_size: usize,

    #[arg(long, default_value_t = 15)]
    pub temp_threshold: i32,

    #[arg(long, default_value_t = 0.55)]
    pub update_threshold: f32,

    #[arg(long, default_value_t = 50)]
    pub num_mcts_sims: i32,

    #[arg(long, default_value_t = 40)]
    pub arena_compare: i32,

    #[arg(long, default_value_t = 1.0)]
    pub c_puct: f32,

    #[arg(long, default_value_t = String::from("./temp2/"))]
    pub checkpoint: String,

    #[arg(long, default_value_t = false)]
    pub load_model: bool,

    #[arg(long, default_value_t = String::from("./temp2/best.safetensors"))]
    pub model_path: String,


    #[arg(long, default_value_t = false)]
    pub load_checkpoint: bool,


    #[arg(long, default_value_t = String::from("./temp2/checkpoint_4_.safetensors.examples"))]
    pub check_point_path: String,

    #[arg(long, default_value_t = 10)]
    pub num_iters_for_train_examples_history: usize,
}


