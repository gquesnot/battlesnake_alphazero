use std::collections::HashMap;
use rand::distributions::Distribution;

use rand::seq::SliceRandom;
use rand_distr::Dirichlet;

use crate::alpha_zero_model::AlphaZeroModel;
use crate::Args;
use crate::canonical_board::CanonicalBoard;
use crate::config::{ACTION_SIZE, EPS};

#[derive(Clone)]
pub struct MCTS {
    // Game should be a trait that your specific game implements
    nnet: AlphaZeroModel,
    // NeuralNet should be a trait for neural network implementations
    qsa: HashMap<(String, usize), f32>,
    // stores Q values for (state, action)
    nsa: HashMap<(String, usize), usize>,
    // stores visit counts for (state, action)
    ns: HashMap<String, usize>,
    // stores visit counts for states
    ps: HashMap<String, [f32; 4]>,
    // initial policy from the neural network
    es: HashMap<String, f32>,
    // game termination statuses
    vs: HashMap<String, [bool; 4]>,
    // valid moves
    c_puct: f32,
    num_mcts_sims: i32,
}

impl MCTS {
    pub fn new(nnet: &AlphaZeroModel, args: Args) -> Self {
        MCTS {
            nnet: nnet.clone(),
            qsa: HashMap::new(),
            nsa: HashMap::new(),
            ns: HashMap::new(),
            ps: HashMap::new(),
            es: HashMap::new(),
            vs: HashMap::new(),
            c_puct:args.c_puct,
            num_mcts_sims: args.num_mcts_sims,
        }
    }

    pub fn get_action_prob(&mut self, state: &CanonicalBoard, temp: f32) -> [f32; 4] {
        let current_state = state.reset_and_clone_as_current_player();
        for _ in 0..self.num_mcts_sims {
            self.search(current_state,0);
        }
        let s = current_state.to_hashmap_string();
        let mut counts: [usize; 4] = [0; 4];
        for (a, count) in counts.iter_mut().enumerate().take(ACTION_SIZE as usize) {
            let key = (s.clone(), a);
            if self.nsa.contains_key(&key) {
                *count = self.nsa[&key];
            }
        }
        if temp == 0.0 {
            let max = counts.iter().max().unwrap();
            let best_actions: Vec<usize> = counts.iter().enumerate().filter(|(_, &count)| count == *max).map(|(i, _)| i).collect();
            let best_action = best_actions.choose(&mut rand::thread_rng()).unwrap();
            let mut probabilities = [0.0; 4];
            probabilities[*best_action] = 1.0;
            probabilities
        } else {
            let mut counts_float: [f32; 4] = [0.0; 4];
            let temp2 = 1.0 / temp;
            for (i, count) in counts_float.iter_mut().enumerate() {
                *count = (counts[i] as f32).powf(temp2);
            }
            let sum: f32 = counts_float.iter().sum();
            let mut probabilities = [0.0; 4];
            for (i, count) in counts_float.iter().enumerate() {
                probabilities[i] = *count / sum;
            }
            probabilities
        }
    }

    fn search(&mut self, state: CanonicalBoard, deep:u8) -> f32 {
        let main_player = state.first_player;
        let s = state.to_hashmap_string();
        if !self.es.contains_key(&s) {
            self.es.insert(s.clone(), state.get_game_ended(main_player));
        }
        if self.es[&s] != 0.0 {
            // terminal node
            return -self.es[&s];
        }

        if !self.ps.contains_key(&s) {
            // leaf node
            let (mut p, v) = self.nnet.predict(&state);
            if deep == 0{
                let alpha = 0.3; // Alpha parameter for Dirichlet distribution
                let noise_ratio = 0.25; // How much noise to mix in with the original policy

                let dirichlet = Dirichlet::new_with_size(alpha, p.len()).unwrap();
                let noise = dirichlet.sample(&mut rand::thread_rng());

                for i in 0..p.len() {
                    p[i] = p[i] * (1.0 - noise_ratio) + noise[i] * noise_ratio;
                }
            }
            let valid_moves = state.get_valid_moves();
            p.iter_mut().enumerate().for_each(|(i, pi)| {
                if !valid_moves[i] {
                    *pi = 0.0;
                }
            });
            let sum: f32 = p.iter().sum();
            if sum > 0.0 {
                p.iter_mut().for_each(|pi| *pi /= sum); // renormalize
            } else {
                // if all valid moves were masked make all valid moves equally probable

                // NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                // If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                p.iter_mut().enumerate().for_each(|(idx, pi)| {
                    if valid_moves[idx] {
                        *pi = 1.0;
                    }
                });
                let sum: f32 = p.iter().sum();
                p.iter_mut().for_each(|pi| *pi /= sum); // renormalize
            }
            self.ps.insert(s.clone(), p);
            self.vs.insert(s.clone(), valid_moves);
            self.ns.insert(s.clone(), 0);
            return -v;
        }

        let valid_moves = self.vs[&s];
        let mut cur_best = -f32::INFINITY;
        let mut best_act = -1;

        for a in 0..ACTION_SIZE {
            let key = (s.clone(), a as usize);
            if valid_moves[a as usize] {
                let u: f32 = if self.nsa.contains_key(&key) {
                    self.qsa[&key] + self.c_puct * self.ps[&s][a as usize] * (self.ns[&s] as f32).sqrt() / (1.0 + self.nsa[&key] as f32)
                } else {
                    self.c_puct * self.ps[&s][a as usize] * (self.ns[&s] as f32 + EPS).sqrt() // Q = 0 ?
                };
                if u > cur_best {
                    cur_best = u;
                    best_act = a as i32;
                }
            }
        }
        let a = best_act as usize;
        let key = (s.clone(), a);
        let (next_s, _) = state.get_next_state(a);
        let v = self.search(next_s,deep+1);
        if self.nsa.contains_key(&key) {
            self.qsa.insert((s.clone(), a), (self.nsa[&key] as f32 * self.qsa[&key] + v) / (self.nsa[&key] + 1) as f32);
            self.nsa.insert((s.clone(), a), self.nsa[&key] + 1);
        } else {
            self.qsa.insert((s.clone(), a), v);
            self.nsa.insert((s.clone(), a), 1);
        }
        self.ns.insert(s.clone(), self.ns[&s] + 1);
        -v
    }
}