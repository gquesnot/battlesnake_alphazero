use std::fs::OpenOptions;

use indicatif::ProgressStyle;
use itertools::Itertools;
use rand::prelude::SliceRandom;
use tch::{Device, nn};
use tch::nn::{Adam, OptimizerConfig};

use crate::canonical_board::CanonicalBoard;
use crate::config::{BATCH_SIZE, BOARD_SIZE, EPOCHS, LEARNING_RATE, MINI_BATCH};
use crate::game::Sample;
use crate::neural_network::NeuralNetwork;
use crate::utils::AverageMeter;

pub fn get_base_device() -> Device {
    Device::cuda_if_available()
}


pub fn get_base_var_store() -> nn::VarStore {
    nn::VarStore::new(get_base_device())
}


pub struct AlphaZeroModel {
    vs: nn::VarStore,
    nnet: NeuralNetwork,
}


impl AlphaZeroModel {
    pub fn new() -> Self {
        let vs = get_base_var_store();
        let nnet = NeuralNetwork::new(&vs.root());
        Self {
            vs,
            nnet,
        }
    }

    pub fn train(&self, samples: Vec<Sample>) {
        let mut optimizer = Adam::default().build(&self.vs, LEARNING_RATE).unwrap();
        let mut pi_losses = AverageMeter::default();
        let mut v_losses = AverageMeter::default();
        let pb = indicatif::ProgressBar::new(EPOCHS as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg} ({eta})")
            .unwrap()
            .progress_chars("##-"));
        let total_iterations = 2048;
        for _ in 0..EPOCHS {
            for i in 0..MINI_BATCH {
                let batch = samples.choose_multiple(&mut rand::thread_rng(), BATCH_SIZE).collect_vec();
                let boards_tensors: Vec<tch::Tensor> = batch.iter().map(|(s, _, _)| {
                    tch::Tensor::from_slice(&s.iter().flat_map(|&row| row.into_iter()).collect::<Vec<f32>>())
                        .view([11, 11])
                }).collect();
                let mut boards = tch::Tensor::stack(&boards_tensors, 0);

                let target_pis = batch.iter().flat_map(|(_, p, _)| p.iter().cloned()).collect::<Vec<f32>>();
                let mut target_pis = tch::Tensor::from_slice(&target_pis).view((-1, 4));
                // Flatten the target_pis to create a single-dimensional tensor
                let target_vs = batch.iter().map(|(_, _, v)| *v).collect::<Vec<_>>();
                let mut target_vs = tch::Tensor::from_slice(&target_vs);
                if get_base_device().is_cuda() {
                    boards = boards.contiguous().to_device(get_base_device());
                    target_pis = target_pis.contiguous().to_device(get_base_device());
                    target_vs = target_vs.contiguous().to_device(get_base_device());
                }

                //println!("Boards: {:?}", boards.size());
                //println!("Target Pis: {:?}", target_pis.size());
                //println!("Target Vs: {:?}", target_vs.size());


                let (out_pi, out_v) = self.nnet.forward(&boards, true);
                //println!("Out Pi: {:?}", out_pi.size());
                //println!("Out V: {:?}", out_v.size());
                let l_pi = self.loss_pi(&target_pis, &out_pi);
                let l_v = self.loss_v(&target_vs, &out_v);
                let total_loss = l_pi.copy() + l_v.copy();

                let f32_l_pi = f32::try_from(l_pi).unwrap();
                let f32_l_v = f32::try_from(l_v).unwrap();
                let b_size = usize::try_from(boards.size()[0]).unwrap();

                pi_losses.update(f32_l_pi, b_size);
                v_losses.update(f32_l_v, b_size);
                pb.set_message(format!("{}/{} pi_loss: {} v_loss: {}", i, total_iterations, pi_losses, v_losses));

                optimizer.zero_grad();
                total_loss.backward();
                optimizer.step();
            }
            pb.inc(1);
        }
        pb.finish();
    }


    pub fn predict(&self, board: &CanonicalBoard) -> ([f32; 4], f32) {
        let mut tensor_board = board.to_tensor();
        if get_base_device().is_cuda() {
            tensor_board = tensor_board.contiguous().to_device(get_base_device());
        }
        tensor_board = tensor_board.view([1, BOARD_SIZE, BOARD_SIZE]);
        let (pi, v) = self.nnet.forward(&tensor_board, false);
        let pi = pi.exp().to(get_base_device());
        let v = v.to(get_base_device());
        let policy = Vec::<f32>::try_from(pi.view(-1))
            .map_err(|e| format!("Failed to convert policy to Vec<f32>: {}", e))
            .expect("Expected a Vec<f32> value after squeeze_dim");
        let value = f32::try_from(v.view(-1))
            .map_err(|e| format!("Failed to convert value to f32: {}", e))
            .expect("Expected a single f32 value after squeeze_dim");
        let mut actions: [f32; 4] = [0.0; 4];
        actions.copy_from_slice(&policy);
        (actions, value)
    }


    pub fn loss_pi(&self, targets: &tch::Tensor, outputs: &tch::Tensor) -> tch::Tensor {
        let loss = targets * outputs;
        -loss.sum(tch::Kind::Float) / tch::Tensor::from(targets.size()[0] as f32)
    }

    pub fn loss_v(&self, targets: &tch::Tensor, outputs: &tch::Tensor) -> tch::Tensor {
        let loss = (targets - outputs.view(-1)).pow(&tch::Tensor::from(2.0));
        loss.sum(tch::Kind::Float) / tch::Tensor::from(targets.size()[0] as f32)
    }


    pub fn save_checkpoint(&self, folder: String, filename: String) -> Result<(), std::io::Error> {
        let path = std::path::Path::new(&folder).join(filename);
        if !path.exists() {
            std::fs::create_dir_all(&folder)?;
        }
        let file = OpenOptions::new().write(true).create(true).open(path)?;
        self.vs.save_to_stream(file).unwrap_or_else(|e| {
            println!("Failed to save checkpoint: {}", e);
        });
        Ok(())
    }

    pub fn load_checkpoint(&mut self, folder: String, filename: String) -> Result<(), std::io::Error> {
        let path = std::path::Path::new(&folder).join(filename);
        self.vs.load(path).unwrap_or_else(|e| {
            println!("Failed to load checkpoint: {}", e);
        });
        Ok(())
    }
}

impl Clone for AlphaZeroModel {
    fn clone(&self) -> Self {
        let mut vs = get_base_var_store();
        vs.copy(&self.vs).unwrap();
        let nnet = NeuralNetwork::new(&vs.root());
        Self {
            vs,
            nnet,
        }
    }
}

impl Default for AlphaZeroModel {
    fn default() -> Self {
        Self::new()
    }
}