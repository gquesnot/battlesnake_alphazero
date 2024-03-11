use std::collections::VecDeque;

use battlesnake_game_types::compact_representation::StandardCellBoard4Snakes11x11;
use battlesnake_game_types::types::{build_snake_id_map, Move, ReasonableMovesGame, SimulableGame, SimulatorInstruments, SnakeId, StandardFoodPlaceableGame};
use battlesnake_game_types::wire_representation::{BattleSnake, NestedGame, Position, Ruleset};
use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::Rng;

use crate::canonical_board::CanonicalBoard;

pub type Board = StandardCellBoard4Snakes11x11;

pub type Sample = ([[f32; 11]; 11], [f32; 4], f32);

pub fn player_to_snake(player: i32) -> SnakeId {
    if player == 1 {
        SnakeId(0)
    } else {
        SnakeId(1)
    }
}

#[derive(Debug)]
pub struct Instruments {}

impl SimulatorInstruments for Instruments {
    fn observe_simulation(&self, _: std::time::Duration) {}
}


pub trait BoardInit {
    fn init_random_board() -> Board;
    fn init_start_of_game_board() -> Board;
}

pub trait CanCanonical {
    fn as_canonical(&self, player: i32) -> CanonicalBoard;
}

pub trait MoveBattleSnake {
    fn get_available_moves(&self) -> Vec<[Move; 2]>;

    fn simulate_moves(&self, moves: &[Move; 2]) -> Board;
}


impl MoveBattleSnake for Board {
    fn get_available_moves(&self) -> Vec<[Move; 2]> {
        let reasonable_moves = self.reasonable_moves_for_each_snake();
        reasonable_moves.into_iter()
            .map(|(_, moves)| moves)
            .multi_cartesian_product()
            .map(|moves| [moves[0], moves[1]])
            .collect()
    }

    fn simulate_moves(&self, moves: &[Move; 2]) -> Board
    {
        let new_state = *self;
        let formatted_moves = moves.iter().enumerate().map(|(idx, &mv)| (SnakeId(idx as u8), [mv])).collect_vec();
        let mut simulated_moves = new_state.simulate_with_moves(&Instruments {}, formatted_moves);
        let mut next_state = simulated_moves.next().unwrap().1;
        let mut rng = rand::thread_rng();
        // rng 25% of the time
        if rng.gen_range(0..4) == 0 {
            next_state.place_food(&mut rng);
        }
        next_state
    }
}

pub fn get_random_snake_body(snake_head: Position, rng: &mut rand::rngs::ThreadRng) -> VecDeque<Position> {
    let all_move = Move::all();
    let mut snake_body = VecDeque::new();
    snake_body.push_back(snake_head);
    let mut current_position = snake_head;
    for _ in 0..3 {
        let mut new_position = current_position;
        while snake_body.contains(&new_position)
            || new_position.x > 10
            || new_position.y > 10
            || new_position.x < 0
            || new_position.y < 0 {
            let random_move = all_move.choose(rng).unwrap();
            new_position = current_position.add_vec(random_move.to_vector());
        }
        snake_body.push_back(new_position);
        current_position = new_position;
    }
    snake_body
}


impl BoardInit for Board {
    fn init_random_board() -> Board {
        let mut rng = rand::thread_rng();
        let player_1_head = Position { x: rng.gen_range(0..4), y: rng.gen_range(0..4) };
        let player_1_body = get_random_snake_body(player_1_head, &mut rng);

        let player_2_head = Position { x: rng.gen_range(7..11), y: rng.gen_range(7..11) };
        let player_2_body = get_random_snake_body(player_2_head, &mut rng);

        // Simplify food placement logic
        let mut foods = Vec::new();
        while foods.len() < 2 {
            let food = Position { x: rng.gen_range(0..11), y: rng.gen_range(0..11) };
            if !(player_1_body.contains(&food) || player_2_body.contains(&food) || foods.contains(&food)) {
                foods.push(food);
            }
        }
        let player_1 = BattleSnake {
            id: "gs_YkwKKSmYwqFFgDk9BycMvWf8".to_string(),
            name: "player0".to_string(),
            health: 99,
            body: player_1_body,
            head: player_1_head,
            actual_length: Some(2),
            shout: None,
        };

        let game = battlesnake_game_types::wire_representation::Game {
            turn: 1,
            board: battlesnake_game_types::wire_representation::Board {
                height: 11,
                width: 11,
                food: foods,
                snakes: vec![
                    player_1.clone(),
                    BattleSnake {
                        id: "gs_vbvwfwk6jBc4jmCrKCbdJh3G".to_string(),
                        name: "player1".to_string(),
                        health: 99,
                        body: player_2_body,
                        head: player_2_head,
                        actual_length: Some(2),
                        shout: None,
                    },
                ],
                hazards: Vec::new(),
            },
            you: player_1.clone(),
            game: NestedGame {
                id: "506514ef-249f-48b8-827b-7bf8d17ac7ad".to_string(),
                ruleset: Ruleset {
                    name: "standard".to_string(),
                    version: "v1.2.3".to_string(),
                    settings: None,
                },
                timeout: 600,
                map: None,
                source: None,
            },
        };
        let snake_id_mapping = build_snake_id_map(&game);
        game.as_cell_board(&snake_id_mapping).unwrap()
    }

    fn init_start_of_game_board() -> Board {
        let file = std::fs::File::open("fixtures/start_of_game.json").unwrap();
        let game: battlesnake_game_types::wire_representation::Game = serde_json::from_reader(file).unwrap();
        let snake_id_mapping = build_snake_id_map(&game);
        game.as_cell_board(&snake_id_mapping).unwrap()
    }
}

impl CanCanonical for Board {
    fn as_canonical(&self, player: i32) -> CanonicalBoard {
        CanonicalBoard::new(*self, player, None)
    }
}





