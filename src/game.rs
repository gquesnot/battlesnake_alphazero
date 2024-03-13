use std::collections::VecDeque;

use battlesnake_game_types::compact_representation::StandardCellBoard4Snakes11x11;
use battlesnake_game_types::types::{build_snake_id_map, Move, ReasonableMovesGame, SimulableGame, SimulatorInstruments, SnakeId, StandardFoodPlaceableGame};
use battlesnake_game_types::wire_representation::{BattleSnake, NestedGame, Position, Ruleset};
use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::Rng;
use rand::rngs::ThreadRng;

use crate::canonical_board::CanonicalBoard;
use crate::config::BOARD_SIZE;

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



pub fn generate_foods(snake_1_head:&Position, snake_2_head:&Position) -> Vec<Position>{
    let mut foods = vec![];
    let center_coord: (i32, i32) = ((BOARD_SIZE as i32 - 1) / 2, (BOARD_SIZE as i32 - 1) / 2);
    let mut rng = rand::thread_rng();
    foods.push( place_food_for_snake(&foods, snake_1_head, center_coord, &mut rng));
    foods.push(place_food_for_snake(&foods, snake_2_head, center_coord, &mut rng));
    if !foods.contains(&center_coord){
        foods.push(center_coord);
    }
    foods.iter().map(|(x, y)| Position{x: *x, y: *y}).collect()
}

pub fn place_food_for_snake(foods:&Vec<(i32,i32)>, snake_head:&Position, center_coord:(i32,i32), rng:&mut ThreadRng)->(i32,i32){

    let possible_player_food = vec![
        (snake_head.x - 1, snake_head.y - 1),
        (snake_head.x - 1, snake_head.y + 1),
        (snake_head.x + 1, snake_head.y - 1),
        (snake_head.x + 1, snake_head.y + 1),
    ];
    let mut available_food = vec![];
    for p in possible_player_food {
        if center_coord == p || (p.0 < 0 || p.0 > 10) || (p.1 < 0 || p.1 > 10) {
            continue;
        }
        if foods.contains(&p){
            continue;
        }
        if (p.0 < snake_head.x  && snake_head.x < center_coord.0 )
            || (center_coord.0 < snake_head.x && snake_head.x < p.0)
            || (p.1 < snake_head.y  && snake_head.y < center_coord.1 )
            || (center_coord.1 < snake_head.y && snake_head.y < p.1){
            if !((p.0 == 0 || p.0 == 10 ) && (p.1 == 0 || p.1 == 10)){
                available_food.push(p);
            }
        }
    }
    available_food.shuffle(rng);
    available_food[0]
}



impl BoardInit for Board {
    fn init_random_board() -> Board {
        let mut rng = rand::thread_rng();
        let (mn, md, mx): (i32, i32, i32) = (1, (BOARD_SIZE as i32 - 1) / 2, BOARD_SIZE as i32 - 2);
        let mut corners = vec![
            Position { x: mn, y: mn },
            Position { x: mn, y: mx },
            Position { x: mx, y: mn },
            Position { x: mx, y: mx },
        ];
        let mut cardinal_points = vec![
            Position { x: mn, y: md },
            Position { x: md, y: mn },
            Position { x: md, y: mx },
            Position { x: mx, y: md },
        ];
        corners.shuffle(&mut rng);
        cardinal_points.shuffle(&mut rng);
        let mut start_points: Vec<Position> = Vec::new();
        if rng.gen_range(0..2) == 0 {
            start_points.append(&mut corners);
            start_points.append(&mut cardinal_points);
        } else {
            start_points.append(&mut cardinal_points);
            start_points.append(&mut corners);
        }

        let player_1_head = start_points[0];
        let player_2_head = start_points[1];

        let player_1_body = VecDeque::from([player_1_head, player_1_head, player_1_head]);
        let  player_2_body = VecDeque::from([player_2_head, player_2_head, player_2_head]);
        let foods = generate_foods(&player_1_head, &player_2_head);




        let player_1 = BattleSnake {
            id: "gs_YkwKKSmYwqFFgDk9BycMvWf8".to_string(),
            name: "player0".to_string(),
            health: 99,
            body: player_1_body,
            head: player_1_head,
            actual_length: Some(3),
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
                        actual_length: Some(3),
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





