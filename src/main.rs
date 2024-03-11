use clap::Parser;

use battlesnake_alphazero::alpha_zero_model::AlphaZeroModel;
use battlesnake_alphazero::Args;
use battlesnake_alphazero::coach::Coach;

fn main() {
    let args = Args::parse();

    let mut model = AlphaZeroModel::default();
    if args.load_model {
        println!("load model from {}/{}", args.load_folder.clone(), args.load_file.clone());
        model.load_checkpoint(args.load_folder.clone(), args.load_file.clone()).unwrap();
    } else {
        println!("Not loading a checkpoint.");
    }
    let mut coach = Coach::new(model, &args);
    if args.load_model {
        println!("load trainExamples from file");
        coach.load_train_examples().unwrap();
    }

    println!("Starting the learning process");
    coach.learn().unwrap();
}
