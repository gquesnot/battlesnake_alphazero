
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
pub mod examples_handler;


#[derive(Parser, Debug, Clone, Default)]
#[clap(name = "battlesnake alphazero", version = "2.0", author = "Canarit")]
pub struct Args {
    #[arg(long, default_value_t = 300)]
    pub num_iterations: i32,

    #[arg(long, default_value_t = 0.002_f64)]
    pub learning_rate: f64,

    #[arg(long, default_value_t = 64_usize)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 10)]
    pub num_epochs: i32,

    #[arg(long, default_value_t = 25)]
    pub num_episodes: i32,

    #[arg(long, default_value_t = 200000_usize)]
    pub max_queue_size: usize,

    #[arg(long, default_value_t = 15)]
    pub temp_threshold: i32,

    #[arg(long, default_value_t = 0.55_f32)]
    pub update_threshold: f32,

    #[arg(long, default_value_t = 25)]
    pub num_mcts_sims: i32,

    #[arg(long, default_value_t = 40)]
    pub arena_compare: i32,

    #[arg(long, default_value_t = 1.0_f32)]
    pub c_puct: f32,

    #[arg(long, default_value_t = String::from("./temp3/"))]
    pub checkpoint: String,

    #[arg(long, default_value_t = false)]
    pub load_model: bool,

    #[arg(long, default_value_t = String::from("./temp/best.safetensors"))]
    pub model_path: String,

    #[arg(long, default_value_t = false)]
    pub load_examples: bool,

    #[arg(long, default_value_t = String::from("./temp/"))]
    pub examples_dir: String,

    #[arg(long, default_value_t = 20_usize)]
    pub num_iters_for_train_examples_history: usize,
}


