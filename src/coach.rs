use std::collections::HashMap;
use std::path::PathBuf;

use indicatif::ProgressStyle;
use itertools::Itertools;
use rand::seq::SliceRandom;

use crate::alpha_zero_model::AlphaZeroModel;
use crate::arena::Arena;
use crate::Args;
use crate::examples_handler::ExamplesHandler;
use crate::game::{Board, BoardInit, CanCanonical, Sample};
use crate::mcts::MCTS;
use crate::utils::{choose_index_based_on_probability};

pub struct Coach {
    model: AlphaZeroModel,
    p_model: AlphaZeroModel,
    mcts: MCTS,
    args: Args,
    skip_first_self_play: bool,
    pub examples_handler: ExamplesHandler,
}

impl Coach {
    pub fn new(model: AlphaZeroModel, args: &Args) -> Self {
        let mut examples_handler = ExamplesHandler::new(args.save_dir.clone(), args.num_iters_for_train_examples_history);
        if args.load_examples {
            examples_handler.load_examples();
        }

        Self {
            model: model.clone(),
            p_model: model.clone(),
            mcts: MCTS::new(&model, args.c_puct, args.num_mcts_sims),
            args: args.clone(),
            skip_first_self_play: args.load_examples,
            examples_handler,
        }
    }

    pub fn execute_episode(&mut self) -> HashMap<[u8;121],  Vec<Sample>> {
        let mut train_examples: HashMap<[u8;121],  Vec<Sample>> = HashMap::new();
        let board = Board::init_random_board();
        let mut current_player = 1;
        let mut canonical_board = board.as_canonical(current_player, self.args.min_health_threshold);
        let mut episode_step = 0;
        loop {
            episode_step += 1;
            let temp = if episode_step < self.args.temp_threshold { 1.0 } else { 0.0 };
            let pi = self.mcts.get_action_prob(&canonical_board, temp);

            let canonical_board_hash = canonical_board.to_hashmap_bytes();
            train_examples.entry(canonical_board_hash).or_insert_with(|| canonical_board.get_mirroring_and_rotation(&pi));

            // chose using the action probabilities of pi
            let action = choose_index_based_on_probability(&pi);
            (canonical_board, current_player) = canonical_board.get_next_state(action,false);
            let value = canonical_board.get_game_ended(current_player);
            if value != 0.0 {
                train_examples.iter_mut().for_each(|(_, data)| {
                    data.iter_mut().for_each(|(_, _, player)| {
                        let player_v: f32 = if current_player as f32 != *player { 1.0 } else { 0.0 };
                        let b_pow: f32 = -1.0;
                        let player_value = value * b_pow.powf(player_v);
                        *player = player_value;
                    });
                });
                return train_examples;
            }
        }
    }


    pub fn learn(&mut self) -> std::io::Result<()> {
        for iteration in 1..self.args.num_iterations {
            // self play
            if !self.skip_first_self_play || iteration > 1 {
                // create a dequeue with max size of num_examples_history

                let mut train_examples: HashMap<[u8;121],  Vec<Sample>> = HashMap::new();
                let pb = indicatif::ProgressBar::new(self.args.num_episodes as u64);
                pb.set_style(ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"));
                let mut sum_episodes_length = 0f32;

                for _ in 0..self.args.num_episodes {
                    self.mcts = MCTS::new(&self.model, self.args.c_puct, self.args.num_mcts_sims);

                    let temp_examples = self.execute_episode();
                    sum_episodes_length += temp_examples.len() as f32  ;
                    for (k, v) in temp_examples.into_iter(){
                        train_examples.entry(k).or_insert(v);
                    }
                    pb.inc(1);
                }
                self.mcts = MCTS::new(&self.model, self.args.c_puct, self.args.num_mcts_sims);
                pb.finish();
                println!("AVG EP LENGTH : {:.2}", sum_episodes_length / self.args.num_episodes as f32);
                self.examples_handler.save_example(train_examples.into_values().flatten().collect_vec());
            }


            let mut train_examples = self.examples_handler.examples.clone().into_iter().flatten().collect::<Vec<Sample>>();
            train_examples.shuffle(&mut rand::thread_rng());

            self.model.save_checkpoint(&PathBuf::from(&self.args.save_dir).join("temp.pth.tar"))?;
            self.p_model.load_checkpoint(&PathBuf::from(&self.args.save_dir).join("temp.pth.tar"))?;


            self.model.train(train_examples, self.args.learning_rate, self.args.num_epochs, self.args.batch_size);

            let mcts = MCTS::new(&self.model,  self.args.c_puct, self.args.num_mcts_sims / 2);
            let p_mcts = MCTS::new(&self.p_model,  self.args.c_puct, self.args.num_mcts_sims / 2);

            let mut arena = Arena::new(mcts, Some(p_mcts), self.args.min_health_threshold);
            let (n_wins, p_wins, draws) = arena.play_games(self.args.arena_compare);
            println!("NEW/PREV WINS : {} / {} ; DRAWS : {}", n_wins, p_wins, draws);

            if p_wins + n_wins == 0 || (n_wins as f32 / (p_wins + n_wins) as f32) < self.args.update_threshold {
                println!("REJECTING NEW MODEL");
                self.model.load_checkpoint(&PathBuf::from(&self.args.save_dir).join("temp.pth.tar"))?;
            } else {
                println!("ACCEPTING NEW MODEL");
                self.model.save_checkpoint(&PathBuf::from(&self.args.save_dir).join(self.get_checkpoint_file(iteration)))?;
                self.model.save_checkpoint(&PathBuf::from(&self.args.save_dir).join("best.pth.tar"))?;
            }
        }
        Ok(())
    }

    pub fn get_checkpoint_file(&self, iteration: i32) -> String {
        format!("checkpoint_{}.pth.tar", iteration).to_string()
    }
}