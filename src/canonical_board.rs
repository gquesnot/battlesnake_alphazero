use battlesnake_game_types::types::{FoodGettableGame, HeadGettableGame, HealthGettableGame, Move, ReasonableMovesGame, SnakeBodyGettableGame, SnakeId, VictorDeterminableGame};
use battlesnake_game_types::wire_representation::Position;
use itertools::Itertools;
use ndarray::Array2;
use tch::Tensor;

use crate::config::BOARD_SIZE;
use crate::game;
use crate::game::{Board, MoveBattleSnake};

pub type AllBoardInfo = (Option<Position>, Option<Vec<Position>>, Option<Position>, Option<Vec<Position>>, Vec<Position>);

pub const  Y_MAPPING:[usize;11] = [0, BOARD_SIZE as usize, (2*BOARD_SIZE) as usize, (3*BOARD_SIZE) as usize, (4*BOARD_SIZE) as usize, (5*BOARD_SIZE) as usize, (6*BOARD_SIZE) as usize, (7*BOARD_SIZE) as usize, (8*BOARD_SIZE) as usize, (9*BOARD_SIZE) as usize, (10*BOARD_SIZE) as usize];


pub fn rotate_board(board: &[[f32; 11]; 11], rotation: usize) -> [[f32; 11]; 11] {
    let mut new_board = [[0.0; 11]; 11];
    let n = board.len(); // Assuming the board is always square

    for i in 0..n {
        for j in 0..n {
            match rotation {
                0 => new_board[i][j] = board[i][j], // No rotation
                1 => new_board[j][n - i - 1] = board[i][j], // 90 degrees rotation
                2 => new_board[n - i - 1][n - j - 1] = board[i][j], // 180 degrees rotation
                3 => new_board[n - j - 1][i] = board[i][j], // 270 degrees rotation
                _ => (), // Should not happen
            }
        }
    }
    new_board
}

pub fn flip_board_horizontal(board: &[[f32; 11]; 11]) -> [[f32; 11]; 11] {
    let mut new_board = [[0.0; 11]; 11];
    let n = board.len(); // Assuming the board is square

    for i in 0..n {
        for j in 0..n {
            new_board[i][j] = board[i][n - j - 1];
        }
    }
    new_board
}


pub fn rotate_policy(pi: &[f32; 4], rotation: usize, flip_horizontal: bool) -> [f32; 4] {
    // Adjust the policy vector based on rotation and flip
    // You need to map the directions accordingly

    let mut rotation_mapping = match rotation {
        0 => [0, 1, 2, 3], // No rotation
        1 => [2, 3, 1, 0], // 90 degrees - Up becomes Left, Right becomes Up, etc.
        2 => [1, 0, 3, 2], // 180 degrees - Up becomes Down, Left becomes Right, etc.
        3 => [3, 2, 0, 1], // 270 degrees - Up becomes Right, Down becomes Left, etc.
        _ => [0, 1, 2, 3], // Should not happen
    };
    if flip_horizontal {
        rotation_mapping.swap(2, 3);
    }
    let mut new_pis = [0.0; 4];
    for i in 0..4 {
        new_pis[i] = pi[rotation_mapping[i]];
    }
    new_pis
}


#[derive(Clone, Debug, Copy)]
pub struct CanonicalBoard {
    pub board: Board,
    pub first_player: i32,
    pub prev_action: Option<Move>,
    pub min_health_threshold: u8,
}

impl CanonicalBoard {
    pub fn new(board: Board, first_player: i32, prev_action: Option<Move>, health_kill_threshold: u8) -> Self {
        CanonicalBoard {
            board,
            first_player,
            prev_action,
            min_health_threshold: health_kill_threshold,
        }
    }


    pub fn to_tensor(&self) -> Tensor {
        let (board_size_x, board_size_y) = (11, 11); // Assuming a fixed board size, adjust if necessary
        let mut array_board: Array2<f32> = Array2::zeros((board_size_x, board_size_y));
        let (self_head, self_body, other_head, other_body, foods) = self.get_info_for_repr();

        if let Some(self_head) = self_head {
            array_board[(self_head.x as usize, self_head.y as usize)] = 1.0;
        }
        if let Some(self_body) = self_body {
            for body in self_body {
                array_board[(body.x as usize, body.y as usize)] = 2.0;
            }
        }
        if let Some(other_head) = other_head {
            array_board[(other_head.x as usize, other_head.y as usize)] = -1.0;
        }
        if let Some(other_body) = other_body {
            for body in other_body {
                array_board[(body.x as usize, body.y as usize)] = -2.0;
            }
        }
        for food in foods {
            array_board[(food.x as usize, food.y as usize)] = 0.5;
        }

        Tensor::try_from(array_board).expect("Failed to convert ndarray to Tensor")
    }


    pub fn reset_and_clone_as_current_player(&self) -> CanonicalBoard {
        let mut new_board = *self;
        if new_board.prev_action.is_some() {
            new_board.prev_action = None;
            new_board.first_player = -new_board.first_player;
        }
        new_board
    }


    pub fn get_mirroring_and_rotation(&self, pi: &[f32; 4]) -> Vec<([[f32; 11]; 11], [f32; 4], f32)> {
        let mut symmetries: Vec<([[f32; 11]; 11], [f32; 4], f32)> = Vec::new();

        let rotations = [0, 1, 2, 3]; // Represents 0, 90, 180, and 270 degrees
        let flips_horizontal = [false, true]; // Represents no flip and horizontal flip
        let current_player = self.get_current_player();

        let array_board = self.to_array_board();
        for &rotation in &rotations {
            for &flip_horizontal in &flips_horizontal {
                let mut new_board = rotate_board(&array_board, rotation);
                if flip_horizontal {
                    new_board = flip_board_horizontal(&new_board);
                }
                let new_pi = rotate_policy(pi, rotation, flip_horizontal);
                symmetries.push((new_board, new_pi, current_player as f32));
            }
        }
        symmetries
    }


    pub fn to_array_board(&self) -> [[f32; 11]; 11] {
        let mut result = [[0.0; 11]; 11];
        let board_size_i32 = (BOARD_SIZE - 1) as i32;
        let (self_head, self_body, other_head, other_body, foods) = self.get_info_for_repr();
        if let Some(self_head) = self_head {
            result[(board_size_i32 - self_head.y) as usize][self_head.x as usize] = 1.0;
            if let Some(self_body) = self_body {
                for body in self_body {
                    if body != self_head {
                        result[(board_size_i32 - body.y) as usize][body.x as usize] = 2.0;
                    }
                }
            }
        }

        if let Some(other_head) = other_head {
            result[(board_size_i32 - other_head.y) as usize][other_head.x as usize] = -1.0;
            if let Some(other_body) = other_body {
                for body in other_body {
                    if body != other_head {
                        result[(board_size_i32 - body.y) as usize][body.x as usize] = -2.0;
                    }
                }
            }
        }

        for food in foods {
            result[(board_size_i32 - food.y) as usize][food.x as usize] = 0.5;
        }
        result
    }


    pub fn to_hashmap_bytes(&self) -> [u8; 121] {
        let mut result = [0u8; 121];
        let (self_head, self_body, other_head, other_body, foods) = self.get_info_for_repr();

        if let Some(self_head) = self_head {
            result[Y_MAPPING[self_head.y as usize] + self_head.x as usize] = 1;
            if let Some(self_body) = self_body {
                for body in self_body {
                    if body != self_head {
                        result[Y_MAPPING[body.y as usize] + body.x as usize] = 2;
                    }
                }
            }
        }

        if let Some(other_head) = other_head {
            result[Y_MAPPING[other_head.y as usize] + other_head.x as usize] = 3;
            if let Some(other_body) = other_body {
                for body in other_body {
                    if body != other_head {
                        result[Y_MAPPING[body.y as usize] + body.x as usize] = 4;
                    }
                }
            }
        }

        for food in foods {
            result[Y_MAPPING[food.y as usize] + food.x as usize] = 5;
        }
        result
    }


    pub fn to_hashmap_string(&self) -> String {
        let (self_head, self_body, other_head, other_body, foods) = self.get_info_for_repr();
        let board_size = BOARD_SIZE as usize;
        let mut array_string = vec!['a'; board_size * board_size];  // Pre-fill the string with '0'

        // Inline function to reduce code duplication
        let mut set_position = |x: usize, y: usize, ch: char| {
            array_string[x + (board_size - 1 - y) * board_size] = ch;
        };

        if let Some(self_head) = self_head {
            set_position(self_head.x as usize, self_head.y as usize, 'b');
            if let Some(self_body) = self_body {
                for body in self_body {
                    if body == self_head {
                        continue;
                    }
                    set_position(body.x as usize, body.y as usize, 'c');
                }
            }
        }

        if let Some(other_head) = other_head {
            set_position(other_head.x as usize, other_head.y as usize, 'd');
            if let Some(other_body) = other_body {
                for body in other_body {
                    if body == other_head {
                        continue;
                    }
                    set_position(body.x as usize, body.y as usize, 'e');
                }
            }
        }
        foods.iter().for_each(|food| set_position(food.x as usize, food.y as usize, 'f'));
        let mut compressed_string = String::new();
        let mut last_char = array_string[0];
        let mut count = 1;
        for &current_char in &array_string[1..] {
            if current_char == last_char {
                count += 1;
            } else {
                compressed_string.push_str(&format!("{}{}", last_char, count));
                last_char = current_char;
                count = 1;
            }
        }

        compressed_string.push_str(&format!("{}{}", last_char, count));
        compressed_string
    }


    pub fn get_current_player(&self) -> i32 {
        if self.prev_action.is_none() {
            self.first_player
        } else {
            -self.first_player
        }
    }

    pub fn get_current_snake(&self) -> SnakeId {
        if self.get_current_player() == 1 {
            SnakeId(0)
        } else {
            SnakeId(1)
        }
    }

    pub fn get_opponent_snake(&self) -> SnakeId {
        if self.get_current_player() == 1 {
            SnakeId(1)
        } else {
            SnakeId(0)
        }
    }


    pub fn get_game_ended(&self, player_id: i32) -> f32 {
        let mut is_over = self.board.is_over();
        let mut winner = self.board.get_winner();
        if !is_over {
            let snake_0_is_dead = self.board.get_health(&SnakeId(0)) <= self.min_health_threshold;
            let snake_1_is_dead = self.board.get_health(&SnakeId(1)) <= self.min_health_threshold;
            if snake_0_is_dead || snake_1_is_dead {
                is_over = true;
                winner = if snake_0_is_dead && snake_1_is_dead { None } else if snake_0_is_dead { Some(SnakeId(1)) } else { Some(SnakeId(0)) };
            }
        }
        if is_over {
            match winner {
                None => {
                    1e-4
                }
                Some(winner) => {
                    if winner == game::player_to_snake(player_id)
                    {
                        1.0
                    } else {
                        -1.0
                    }
                }
            }
        } else {
            0.0
        }
    }

    pub fn get_snake_head_and_body(&self, snake_id: &SnakeId) -> (Option<Position>, Option<Vec<Position>>) {
        if self.board.is_alive(snake_id) {
            let head = self.board.get_head_as_position(snake_id);
            let body = self.board.get_snake_body_iter(snake_id).map(|cell_index| cell_index.into_position(BOARD_SIZE as u8)).collect_vec();
            (Some(head), Some(body))
        } else {
            (None, None)
        }
    }


    pub fn get_info_for_repr(&self) -> AllBoardInfo {
        let snake_id = self.get_current_snake();
        let (self_head, self_body) = self.get_snake_head_and_body(&snake_id);
        let opponent_snake_id = self.get_opponent_snake();
        let (other_head, other_body) = self.get_snake_head_and_body(&opponent_snake_id);
        let foods = self.board.get_all_food_as_positions();
        (self_head, self_body, other_head, other_body, foods)
    }

    pub fn get_next_state(&self, action: usize, in_mcts: bool) -> (CanonicalBoard, i32) {
        let action = Move::from_index(action);
        let next_board = self.play_action(action, in_mcts);
        let next_player = next_board.get_current_player();
        (next_board, next_player)
    }


    pub fn get_valid_moves(&self) -> [bool; 4] {
        let snake_id = self.get_current_snake();
        let mut valid_moves = [false; 4];
        for (id, mvs) in self.board.reasonable_moves_for_each_snake() {
            if id != snake_id { continue; }
            for mv in mvs {
                valid_moves[mv.as_index()] = true;
            }
        }
        valid_moves
    }


    pub fn play_action(&self, action: Move, in_mcts: bool) -> CanonicalBoard {
        if let Some(prev_action) = self.prev_action {
            let actions = if self.first_player == 1 {
                [prev_action, action]
            } else {
                [action, prev_action]
            };
            let next_board = self.board.simulate_moves(&actions, in_mcts);
            CanonicalBoard::new(next_board, self.first_player, None, self.min_health_threshold)
        } else {
            let mut new_state = *self;
            new_state.prev_action = Some(action);
            new_state
        }
    }
}
