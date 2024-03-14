use criterion::{black_box, Criterion, criterion_group, criterion_main};

use battlesnake_alphazero::alpha_zero_model::AlphaZeroModel;
use battlesnake_alphazero::canonical_board::{CanonicalBoard, flip_board_horizontal, rotate_board, rotate_policy};
use battlesnake_alphazero::game::{Board, BoardInit, CanCanonical};
use battlesnake_alphazero::mcts::MCTS;

pub fn get_canonical_board(min_health_threshold:u8) -> CanonicalBoard {
    Board::init_start_of_game_board().as_canonical(1i32, min_health_threshold)
}


pub fn bench_generate_random_board(c: &mut Criterion) {
    c.bench_function("bench_generate_random_board", |b| b.iter(|| {
        Board::init_random_board();
    }));
}

pub fn bench_canonical_board_get_next_state(c: &mut Criterion) {
    let canonical_board = get_canonical_board(80);
    c.bench_function("bench_canonical_board_get_next_state", |b| b.iter(|| {
        canonical_board.get_next_state(black_box(0), false);
    }));
}


pub fn bench_canonical_board_to_tensor(c: &mut Criterion) {
    let canonical_board = get_canonical_board(80);
    c.bench_function("bench_canonical_board_to_tensor", |b| b.iter(|| {
        let _ = canonical_board.to_tensor();
    }));
}


pub fn bench_canonical_board_to_array_board(c: &mut Criterion) {
    let canonical_board = get_canonical_board(80);
    c.bench_function("bench_canonical_board_to_array_board", |b| b.iter(|| {
        canonical_board.to_array_board();
    }));
}

pub fn bench_canonical_board_to_hashmap_string(c: &mut Criterion) {
    let canonical_board = get_canonical_board(80);
    c.bench_function("bench_canonical_board_to_hashmap_string", |b| b.iter(|| {
        canonical_board.to_hashmap_string();
    }));
}


pub fn bench_canonical_board_mirroring_and_rotation(c: &mut Criterion) {
    let canonical_board = get_canonical_board(80);
    let pi = [0.25, 0.25, 0.25, 0.25];
    c.bench_function("bench_canonical_board_mirroring_and_rotation", |b| b.iter(|| {
        canonical_board.get_mirroring_and_rotation(black_box(&pi));
    }));
}


pub fn bench_rotate_array_board(c: &mut Criterion) {
    let canonical_board = get_canonical_board(80);
    let array_board = canonical_board.to_array_board();
    c.bench_function("bench_rotate_array_board", |b| b.iter(|| {
        rotate_board(black_box(&array_board), black_box(1));
    }));
}

pub fn bench_flip_horizontal_array_board(c: &mut Criterion) {
    let canonical_board = get_canonical_board(80);
    let array_board = canonical_board.to_array_board();
    c.bench_function("bench_flip_horizontal_array_board", |b| b.iter(|| {
        flip_board_horizontal(black_box(&array_board));
    }));
}


pub fn bench_rotate_policy(c: &mut Criterion) {
    let pi: [f32; 4] = [0.25, 0.25, 0.25, 0.25];
    c.bench_function("bench_rotate_policy", |b| b.iter(|| {
        rotate_policy(black_box(&pi), black_box(1), black_box(true));
    }));
}


pub fn bench_mcts(c: &mut Criterion) {
    let canonical_board = get_canonical_board(80);
    let model = AlphaZeroModel::new(128);
    c.bench_function("bench_mcts", |b| b.iter(|| {
        let mut mcts = MCTS::new(&model, 4.0, 1000);
        mcts.get_action_prob(black_box(&canonical_board), black_box(0.0));
    }));
}


pub fn bench_alphazero_get_action_probs(c: &mut Criterion) {
    let canonical_board = get_canonical_board(80);
    let model = AlphaZeroModel::default();
    c.bench_function("bench_alphazero_get_action_probs", |b| b.iter(|| {
        model.clone().predict(black_box(&canonical_board));
    }));
}










criterion_group!(
    benches,
    // bench_generate_random_board,
    // bench_canonical_board_to_tensor,
    //  bench_canonical_board_get_next_state,
    //  bench_canonical_board_to_array_board,
      //bench_canonical_board_to_hashmap_string,
    //  bench_canonical_board_mirroring_and_rotation,
     //bench_rotate_array_board,
    //bench_flip_horizontal_array_board,
     //bench_rotate_policy,
    bench_mcts,
    //bench_alphazero_get_action_probs,
);
criterion_main!(benches);