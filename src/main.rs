use std::path::PathBuf;
use clap::Parser;

use battlesnake_alphazero::alpha_zero_model::AlphaZeroModel;
use battlesnake_alphazero::Args;
use battlesnake_alphazero::coach::Coach;

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
    let mut model = AlphaZeroModel::default();

    if args.load_model {
        println!("load model from {}", args.model_path.clone());
        model.load_checkpoint(&PathBuf::from(&args.model_path)).unwrap();
    } else {
        println!("Not loading a checkpoint.");
    }
    let mut coach = Coach::new(model, &args);
    if args.load_checkpoint {
        println!("load trainExamples from file");
        coach.load_train_examples().unwrap();
    }

    println!("Starting the learning process");
    coach.learn().unwrap();
}
