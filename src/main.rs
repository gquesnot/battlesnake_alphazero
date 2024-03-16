use std::path::PathBuf;

use clap::Parser;

use battlesnake_alphazero::alpha_zero_model::AlphaZeroModel;
use battlesnake_alphazero::arena::Arena;
use battlesnake_alphazero::Args;
use battlesnake_alphazero::coach::Coach;
use battlesnake_alphazero::mcts::MCTS;

pub fn print_board(board: &[[f32; 11]; 11]) {
    for row in board.iter() {
        print!("|");
        for cell in row.iter() {
            if *cell == 0.0_f32 {
                print!(" |");
            } else if *cell == -1.0f32 {
                print!("1|");
            } else if *cell == -2.0f32 || *cell == 2.0f32 {
                print!("X|");
            } else if *cell == 1.0f32 {
                print!("0|");
            } else if *cell == 0.5f32 {
                print!("*|");
            }
        }
        println!();
    }
    println!();
}



fn main() {
    let args = Args::parse();
    let mut model = AlphaZeroModel::new(args.num_channels);
    let save_dir = PathBuf::from(&args.save_dir);
    if !save_dir.exists() {
        std::fs::create_dir_all(&save_dir).unwrap();
    }

    if args.load_model {
        let path = PathBuf::from(&args.save_dir).join("best.safetensors");
        if path.exists() {
            println!("load model from {}", path.display());
            model.load_checkpoint(&path).unwrap();
        } else {
            println!("No model found at {}", path.display());
        }
    } else {
        println!("Not loading a checkpoint.");
    }
    if let Some(vs_model_path) = &args.vs_model_path {
        let path = PathBuf::from(&vs_model_path);
        let mut other_model = AlphaZeroModel::new(args.num_channels);
        if path.exists() {
            println!("load vs model from {}", path.display());
            other_model.load_checkpoint(&path).unwrap();
        } else {
            println!("No model found at {}", path.display());
        }
        let model_mcts = MCTS::new(&model, args.c_puct, args.num_mcts_sims);
        let other_model_mcts = MCTS::new(&other_model, args.c_puct, args.num_mcts_sims);
        let mut arena = Arena::new(model_mcts, Some(other_model_mcts), args.min_health_threshold);
        let (model_wins, other_model_wins, draws) = arena.play_games(args.arena_compare);
        println!("Model Wins: {}, Other Model Wins: {}, Draws: {}", model_wins, other_model_wins, draws);
    }else if let Some(vs_normal_mcts) = &args.vs_normal_mcts{
        let model_mcts = MCTS::new(&model, args.c_puct, args.num_mcts_sims);
        let mut arena = Arena::new(model_mcts, None, args.min_health_threshold);
        let (model_wins, other_model_wins, draws) = arena.play_games_vs_normal_mcts(args.arena_compare, *vs_normal_mcts);
        println!("Model Wins: {}, MCTS({}) Wins: {}, Draws: {}", model_wins, *vs_normal_mcts,other_model_wins, draws);
    }
    else{
        let mut coach = Coach::new(model, &args);
        println!("Starting the learning process");
        coach.learn().unwrap();
    }
}
