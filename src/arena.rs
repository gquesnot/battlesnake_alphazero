use indicatif::ProgressStyle;

use crate::game::{Board, BoardInit, CanCanonical};
use crate::mcts::MCTS;

pub struct Arena {
    player1: MCTS,
    player2: MCTS,
}

impl Arena {
    pub fn new(player1: MCTS, player2: MCTS) -> Arena {
        Arena {
            player1,
            player2,
        }
    }


    pub fn play_game(&mut self) -> f32 {
        let board = Board::init_random_board();
        let mut current_player = 1;
        let mut canonical_board = board.as_canonical(current_player);
        while canonical_board.get_game_ended(1) == 0.0 {
            let actions = if current_player == 1 {
                self.player1.get_action_prob(&canonical_board, 0.0)
            } else {
                self.player2.get_action_prob(&canonical_board, 0.0)
            };
            let mut best_action_index = actions.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
            let valid_moves = canonical_board.get_valid_moves();
            if !valid_moves[best_action_index] {
                best_action_index = 0;
            }
            (canonical_board, current_player) = canonical_board.get_next_state(best_action_index);
        }
        canonical_board.get_game_ended(1)
    }

    pub fn play_games(&mut self, num: i32) -> (i32, i32, i32) {
        let mut p1_wins = 0;
        let mut p2_wins = 0;
        let mut draws = 0;

        let pb = indicatif::ProgressBar::new(num as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("##-"));
        for _ in 0..num {
            pb.inc(1);
            let game_result = self.play_game();
            if game_result == 1.0 {
                p1_wins += 1;
            } else if game_result == -1.0 {
                p2_wins += 1;
            } else {
                draws += 1;
            }
            pb.set_message(format!("P1 wins: {} P2 wins: {} Draws: {}", p1_wins, p2_wins, draws));
        }
        (p1_wins, p2_wins, draws)
    }
}