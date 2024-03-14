use battlesnake_game_types::types::{Move, SnakeId};
use indicatif::ProgressStyle;
use itertools::Itertools;

use crate::game::{Board, BoardInit, CanCanonical};
use crate::mcts::MCTS;
use crate::normal_mcts::{ mcts_parallel, MCTSNode};

pub struct Arena {
    n_player: MCTS,
    p_player: Option<MCTS>,
}

impl Arena {
    pub fn new(n_player: MCTS, p_player: Option<MCTS>) -> Arena {
        Arena {
            n_player,
            p_player,
        }
    }


    pub fn play_game(&mut self) -> f32 {
        if let Some(ref mut p_player) = &mut self.p_player{
            let board = Board::init_random_board();
            let mut current_player = 1;
            let mut canonical_board = board.as_canonical(current_player);
            loop {
                let value = canonical_board.get_game_ended(1);
                if value != 0.0 {
                    return value;
                }
                let actions = if current_player == 1 {
                    self.n_player.get_action_prob(&canonical_board, 0.0)
                } else {
                    p_player.get_action_prob(&canonical_board, 0.0)
                };
                let mut best_action_index = actions.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
                let valid_moves = canonical_board.get_valid_moves();
                if !valid_moves[best_action_index] {
                    best_action_index = 0;
                }
                (canonical_board, current_player) = canonical_board.get_next_state(best_action_index);
            }
        }
        0.0

    }

    pub fn play_games(&mut self, num: i32) -> (i32, i32, i32) {
        let mut n_wins = 0;
        let mut p_wins = 0;
        let mut draws = 0;

        let pb = indicatif::ProgressBar::new(num as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        for _ in 0..num {
            let game_result = self.play_game();
            if game_result == 1.0 {
                n_wins += 1;
            } else if game_result == -1.0 {
                p_wins += 1;
            } else {
                draws += 1;
            }
            pb.inc(1);
            pb.set_message(format!("New wins: {} Past wins: {} Draws: {}", n_wins, p_wins, draws));
        }
        pb.finish();
        (n_wins, p_wins, draws)
    }


    pub fn play_games_vs_normal_mcts(&mut self, num: i32, num_mcts_iterations:usize) -> (i32, i32, i32) {
        let mut n_wins = 0;
        let mut p_wins = 0;
        let mut draws = 0;

        let pb = indicatif::ProgressBar::new(num as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        for _ in 0..num {
            let game_result = self.play_game_vs_normal_mcts(num_mcts_iterations);
            if game_result == 1.0 {
                n_wins += 1;
            } else if game_result == -1.0 {
                p_wins += 1;
            } else {
                draws += 1;
            }
            pb.inc(1);
            pb.set_message(format!("New wins: {} Past wins: {} Draws: {}", n_wins, p_wins, draws));
        }
        pb.finish();
        (n_wins, p_wins, draws)
    }


    pub fn play_game_vs_normal_mcts(&mut self, num_mcts_iterations:usize) -> f32 {
        let board = Board::init_random_board();
        let mut current_player = 1;
        let mut canonical_board = board.as_canonical(current_player);
        println!("{}",canonical_board.board);

        let mut iter = 0;
        let mut temp_moves = vec![];
        loop {
            let value = canonical_board.get_game_ended(1);
            if value != 0.0 {
                return value;
            }
            let actions = if current_player == 1 {
                self.n_player.get_action_prob(&canonical_board, 0.0)
            } else {
                let other_snake_id= SnakeId(1);
                let best_node = mcts_parallel(
                        MCTSNode::new(canonical_board.board, None, None, 0),
                        &other_snake_id,
                        num_mcts_iterations,
                    12
                    );
                let best_move  = best_node.get_best_move_made(&other_snake_id);
                let mut actions = [0.0; 4];
                actions[best_move.as_index()] = 1.0;
                actions
            };
            let mut best_action_index = actions.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
            let valid_moves = canonical_board.get_valid_moves();
            if !valid_moves[best_action_index] {
                best_action_index = 0;
            }
            temp_moves.push(best_action_index);
            (canonical_board, current_player) = canonical_board.get_next_state(best_action_index);
            if iter%2 == 1{
                println!("Moves: {:?}",temp_moves.iter().map(|x| Move::from_index(*x)).collect_vec());
                println!("{}",canonical_board.board);
                temp_moves = vec![];
            }
            iter+=1;
        }
    }
}