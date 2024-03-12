use std::fs;
use std::fs::OpenOptions;
use std::path::Path;

use indicatif::ProgressStyle;
use itertools::Itertools;
use rand::seq::SliceRandom;

use crate::alpha_zero_model::AlphaZeroModel;
use crate::arena::Arena;
use crate::Args;
use crate::game::{Board, BoardInit, CanCanonical, Sample};
use crate::mcts::MCTS;
use crate::utils::{BoundedDeque, choose_index_based_on_probability};

pub struct Coach {
    model: AlphaZeroModel,
    p_model: AlphaZeroModel,
    mcts: MCTS,
    examples_history: Vec<Vec<Sample>>,
    args: Args,
    skip_first_self_play: bool,
}

impl Coach {
    pub fn new(model: AlphaZeroModel, args: &Args) -> Self {
        Self {
            model: model.clone(),
            p_model: model.clone(),
            mcts: MCTS::new(&model, args.clone()),
            examples_history: Vec::new(),
            args: args.clone(),
            skip_first_self_play: false,
        }
    }

    pub fn execute_episode(&mut self) -> Vec<Sample> {
        let mut train_examples: Vec<([[f32; 11]; 11], [f32; 4], i32)> = Vec::new();
        let board = Board::init_random_board();
        let mut current_player = 1;
        let mut canonical_board = board.as_canonical(current_player);
        let mut episode_step = 0;
        loop {
            episode_step += 1;
            let temp = if episode_step < self.args.temp_threshold { 1.0 } else { 0.0 };
            let pi = self.mcts.get_action_prob(&canonical_board, temp);

            train_examples.append(&mut canonical_board.get_mirroring_and_rotation(&pi));
            // chose using the action probabilities of pi
            let action = choose_index_based_on_probability(&pi);
            (canonical_board, current_player) = canonical_board.get_next_state(action);
            let value = canonical_board.get_game_ended(current_player);
            if value != 0.0 {
                return train_examples.into_iter().map(|(s, p, player)| {
                    let player_v: f64 = if current_player != player { 1.0 } else { 0.0 };
                    let b_pow: f64 = -1.0;
                    let player_value = value * b_pow.powf(player_v) as f32;
                    (s, p, player_value)
                }).collect();
            }
        }
    }


    pub fn learn(&mut self) -> std::io::Result<()> {
        for iteration in 1..self.args.num_iterations {
            // self play
            if !self.skip_first_self_play || iteration > 1 {
                // create a dequeue with max size of num_examples_history

                let mut train_examples: BoundedDeque<Sample> = BoundedDeque::new(self.args.max_queue_size);
                let pb = indicatif::ProgressBar::new(self.args.num_episodes as u64);
                pb.set_style(ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"));
                for _ in 0..self.args.num_episodes {
                    pb.inc(1);
                    self.mcts = MCTS::new(&self.model, self.args.clone());
                    train_examples.append(self.execute_episode());
                }
                self.examples_history.push(train_examples.deque.into_iter().collect_vec());
                pb.finish();
            }
            if self.examples_history.len() > self.args.num_iters_for_train_examples_history {
                println!("Removing the oldest examples. len: {}", self.examples_history.len());
                self.examples_history.remove(0);
            }
            self.save_train_examples(iteration - 1).unwrap_or_else(|e| {
                println!("Failed to save examples: {}", e);
            });

            let mut train_examples = self.examples_history.clone().into_iter().flatten().collect::<Vec<Sample>>();
            train_examples.shuffle(&mut rand::thread_rng());


            self.model.save_checkpoint(self.args.checkpoint.clone(), "temp.pth.tar".to_string())?;
            self.p_model.load_checkpoint(self.args.checkpoint.clone(), "temp.pth.tar".to_string())?;
            let p_mcts = MCTS::new(&self.p_model, self.args.clone());

            self.model.train(train_examples);

            let mcts = MCTS::new(&self.model, self.args.clone());

            let mut arena = Arena::new(mcts, p_mcts);
            let (n_wins, p_wins, draws) = arena.play_games(self.args.arena_compare);
            println!("NEW/PREV WINS : {} / {} ; DRAWS : {}", n_wins, p_wins, draws);

            if p_wins + n_wins == 0 || (n_wins as f32 / (p_wins + n_wins) as f32) < self.args.update_threshold {
                println!("REJECTING NEW MODEL");
                self.model.load_checkpoint(self.args.checkpoint.clone(), "temp.pt".to_string())?;
            } else {
                println!("ACCEPTING NEW MODEL");
                self.model.save_checkpoint(self.args.checkpoint.clone(), self.get_checkpoint_file(iteration))?;
                self.model.save_checkpoint(self.args.checkpoint.clone(), "best.pt".to_string())?;
            }
        }
        Ok(())
    }

    pub fn get_checkpoint_file(&self, iteration: i32) -> String {
        format!("checkpoint_{}_.pt", iteration).to_string()
    }

    pub fn load_train_examples(&mut self) -> std::io::Result<()> {
        let examples_file_name = self.args.load_file.to_string() + ".examples";
        let example_file = Path::new(&self.args.load_folder).join(&examples_file_name);
        if !example_file.exists() {
            panic!("File not found: {}", example_file.to_str().unwrap());
        } else {
            let  file = fs::File::open(example_file)?;
            self.examples_history = bincode::deserialize_from(&file).unwrap_or_else(|e| {
                println!("Failed to load examples: {}", e);
                Vec::new()
            });
        }
        self.skip_first_self_play = true;
        Ok(())
    }

    pub fn save_train_examples(&self, iteration: i32) -> std::io::Result<()> {
        if !Path::new(&self.args.checkpoint).exists() {
            fs::create_dir_all(&self.args.checkpoint)?;
        }
        let check_point_file_name = self.get_checkpoint_file(iteration) + ".examples";
        let check_point_file = Path::new(&check_point_file_name);
        let filename = Path::new(&self.args.checkpoint).join(check_point_file);
        let mut prev_examples_history: Vec<Vec<Sample>> = Vec::new();
        if filename.exists() {
            let file = fs::File::open(filename.clone())?;
            prev_examples_history = bincode::deserialize_from(&file).unwrap_or_else(|e| {
                println!("Failed to load examples: {}", e);
                Vec::new()
            });
        }
        prev_examples_history.append(&mut self.examples_history.clone());
        let file = OpenOptions::new().write(true).create(true).open(filename)?;
        bincode::serialize_into(&file, &prev_examples_history).unwrap_or_else(|e| {
            println!("Failed to save examples: {}", e);
        });
        Ok(())
    }
}