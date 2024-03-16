
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
pub mod normal_mcts;


#[derive(Parser, Debug, Clone, Default)]
#[clap(name = "battlesnake alphazero", version = "2.0", author = "Canarit")]
pub struct Args {
    #[arg(long, default_value_t = 20)]
    pub num_iterations: i32,

    #[arg(long, default_value_t = 0.001_f64)]
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

    #[arg(long, default_value_t = 400)]
    pub num_mcts_sims: i32,

    #[arg(long, default_value_t = 30)]
    pub arena_compare: i32,

    #[arg(long, default_value_t = 4.0_f32)]
    pub c_puct: f32,

    #[arg(long, default_value_t = String::from("./temp"))]
    pub save_dir: String,


    #[arg(long, default_value_t = false)]
    pub load_model: bool,


    #[arg(long, default_value_t = false)]
    pub load_examples: bool,


    #[arg(long, default_value_t = 10_usize)]
    pub num_iters_for_train_examples_history: usize,

    #[arg(long)]
    pub  vs_model_path: Option<String>,

    #[arg(long)]
    pub vs_normal_mcts: Option<usize>,

    #[arg(long, default_value_t = 512i64)]
    pub num_channels: i64,

    // 100 - 85 => 2.25food/20round
    // 100 - 80 => 3food/20round
    // 100 - 75 => 3.75food/20round
    // 100 - 70 => 4.5food/20round
    #[arg(long, default_value_t = 75)]
    pub min_health_threshold: u8,

}


