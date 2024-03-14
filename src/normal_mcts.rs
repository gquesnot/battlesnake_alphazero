use battlesnake_game_types::compact_representation::StandardCellBoard4Snakes11x11;
use battlesnake_game_types::types::{Move, SnakeId, VictorDeterminableGame};
use rand::prelude::IteratorRandom;
use rayon::iter::IntoParallelIterator;
use crate::game::MoveBattleSnake;
use rayon::iter::ParallelIterator;

pub fn run_simulation(state: StandardCellBoard4Snakes11x11) -> Option<SnakeId> {
    let mut current_state = state;
    let mut rng = rand::thread_rng();
    while !current_state.is_over() {
        let chosen_moves = current_state.get_available_moves()
            .into_iter()
            .choose(&mut rng).unwrap();
        current_state = current_state.simulate_moves(&chosen_moves);
    }
    current_state.get_winner()
}
pub fn backpropagate(tree: &mut [MCTSNode], mut node_index: usize, winner: Option<SnakeId>, player_id: &SnakeId) {
    while let Some(node) = tree.get_mut(node_index) {
        node.visits += 1;
        if let Some(winner) = winner {
            if winner == *player_id {
                node.wins += 1;
            }
        }
        match node.parent {
            Some(parent) => {
                if parent == node_index { // Check for self-reference
                    break;
                }
                node_index = parent;
            }
            None => break,
        };
    }
}

pub fn best_child(node: &MCTSNode, tree: &[MCTSNode], is_exploration: bool) -> usize {
    let mut best_score = f64::MIN;
    let mut best_child = 0;

    for &child_index in &node.children {
        let child = &tree[child_index];
        let child_visits = if child.visits == 0 { 1 } else { child.visits } as f64;
        let win_rate = child.wins as f64 / child_visits;
        let exploration_value = if is_exploration {
            (2.0 * (node.visits as f64).ln() / child_visits).sqrt()
        } else {
            0.0
        };
        let score = win_rate + exploration_value;

        if score > best_score {
            best_score = score;
            best_child = child_index;
        }
    }

    best_child
}

pub fn mcts(root: MCTSNode, player_id: &SnakeId, iterations: usize) -> Vec<MCTSNode> {
    let mut tree = vec![root.clone()];
    for _ in 0..iterations {
        let mut node_index = 0; // Start from the root
        while !tree[node_index].children.is_empty() && !tree[node_index].state.is_over() {
            node_index = best_child(&tree[node_index], &tree, true);
        }

        // Expansion
        if !tree[node_index].state.is_over() {
            let tree_len = tree.len();
            tree[node_index]
                .state
                .get_available_moves()
                .into_iter()
                .enumerate()
                .for_each(|(child_index, node_moves)| {
                    tree[node_index].children.push(tree_len + child_index); // Use the precomputed index here
                    let node_moves_2 = node_moves.into_iter().enumerate().map(|(idx, mv)| (SnakeId(idx as u8), mv)).collect();
                    tree.push(MCTSNode::new(tree[node_index].state.simulate_moves(&node_moves), Some(node_index), Some(node_moves_2), tree[node_index].deep + 1));
                });
        }

        if !tree[node_index].children.is_empty() {
            node_index = *tree[node_index].children.first().unwrap();
        }

        // Simulation
        let winner = run_simulation(tree[node_index].state);

        // Backpropagation
        backpropagate(&mut tree, node_index, winner, player_id);
    }

    // Return the most visited move from the root
    //let (deep, index) = get_deepest_level_and_index(&tree);
    //println!("Deep: {}, Index: {}, nodes: {}", deep, index, tree.len());
    tree
}

pub fn best_node_from_mcts(tree:&[MCTSNode]) -> MCTSNode {
    let best_move_index = best_child(&tree[0], tree, false);
    tree[best_move_index].clone()
}



pub fn mcts_parallel(root: MCTSNode, player_id: &SnakeId, iterations: usize, num_threads: usize) -> MCTSNode {
    let total_iterations = iterations * num_threads;
    let per_thread_iterations = total_iterations / num_threads;

    // Using Rayon to parallelize the MCTS execution across multiple threads
    let trees: Vec<Vec<MCTSNode>> = (0..num_threads)
        .into_par_iter()
        .map(|_| {
            let local_root = root.clone();
            let local_player_id = *player_id;
            let tree = mcts(local_root, &local_player_id, per_thread_iterations);
            let root = tree[0].clone();
            let relevant_indexes:Vec<_> = vec![0].into_iter().chain(root.children).collect();
            relevant_indexes.iter().map(|&child_index| tree[child_index].clone()).collect()
        })
        .collect();
    let mut result_tree = trees[0].clone();
    let children_index = result_tree[0].children.clone();
    for child_index in children_index{
        let child_node = &mut result_tree[child_index];
        for other_tree in trees.iter().take(num_threads).skip(1) {
            for j in 0..other_tree[0].children.len() {
                let other_child_index = other_tree[0].children[j];
                let other_child_node = &other_tree[other_child_index];
                if child_node.moves_made == other_child_node.moves_made {
                    child_node.visits += other_child_node.visits;
                    child_node.wins += other_child_node.wins;
                }
            }

        }
    }
    best_node_from_mcts(&result_tree)
}



#[derive(Debug, Clone)]
pub struct MCTSNode {
    pub state: StandardCellBoard4Snakes11x11,
    pub parent: Option<usize>,

    pub children: Vec<usize>,
    pub moves_made: Option<Vec<(SnakeId, Move)>>,
    pub wins: usize,
    pub visits: usize,
    pub deep: usize,
}


impl MCTSNode {
    pub fn new(state: StandardCellBoard4Snakes11x11, parent: Option<usize>, moves_made: Option<Vec<(SnakeId, Move)>>, deep: usize) -> MCTSNode {
        MCTSNode {
            state,
            parent,
            moves_made,
            children: Vec::new(),
            wins: 0,
            visits: 0,
            deep,
        }
    }

    pub fn get_best_move_made(&self, snake_id:&SnakeId) -> Move {
        self.moves_made.as_ref().unwrap().iter().find(|(id, _)| id == snake_id).unwrap().1
    }
}
