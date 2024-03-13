use std::path::PathBuf;

use indicatif::ProgressStyle;
use itertools::{Itertools, multiunzip};
use ndarray::{arr1, arr2, arr3};
use tch::{Device, nn, no_grad, Tensor};
use tch::nn::{Adam, OptimizerConfig};

use crate::canonical_board::CanonicalBoard;
use crate::game::Sample;
use crate::neural_network::NeuralNetwork;
use crate::utils::AverageMeter;

pub fn get_base_device() -> Device {
    Device::cuda_if_available()
}


pub fn get_base_var_store() -> nn::VarStore {
    nn::VarStore::new(get_base_device())
}

pub type SampleZipped = (Vec<[[f32; 11]; 11]>, Vec<[f32; 4]>, Vec<f32>);


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

    pub fn train(&self, samples: Vec<Sample>, learning_rate: f64, epochs: i32, batch_size: usize) {
        let mut optimizer = Adam::default().build(&self.vs, learning_rate).unwrap();
        let mut pi_losses = AverageMeter::default();
        let mut v_losses = AverageMeter::default();
        let pb = indicatif::ProgressBar::new(epochs as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg} ({eta})")
            .unwrap()
            .progress_chars("##-"));

        let batch_count = samples.len() / batch_size;


        for _ in 0..epochs {
            for i in 0..batch_count {
                let ids = rand::seq::index::sample(&mut rand::thread_rng(), samples.len(), batch_size).into_vec();
                let (boards, pi, value): SampleZipped = multiunzip(ids.iter().map(|&i| samples[i]).collect_vec());

                let mut s = Tensor::try_from(arr3(&boards)).unwrap();
                let mut target_pis = Tensor::try_from(arr2(&pi)).unwrap();
                let mut target_vs = Tensor::try_from(arr1(&value)).unwrap();
                if get_base_device().is_cuda() {
                    s = s.to_device(get_base_device());
                    target_pis = target_pis.to_device(get_base_device());
                    target_vs = target_vs.to_device(get_base_device());
                }

                let (out_pi, out_v) = self.nnet.forward(&s, true);
                //println!("Out Pi: {:?}", out_pi.size());
                //println!("Out V: {:?}", out_v.size());
                let l_pi = self.loss_pi(&target_pis, &out_pi);
                let l_v = self.loss_v(&target_vs, &out_v);
                let total_loss = l_pi.copy() + l_v.copy();

                let f32_l_pi = f32::try_from(l_pi).unwrap();
                let f32_l_v = f32::try_from(l_v).unwrap();
                let b_size = usize::try_from(s.size()[0]).unwrap();

                pi_losses.update(f32_l_pi, b_size);
                v_losses.update(f32_l_v, b_size);
                if i % 100 == 0 {
                    pb.set_message(format!("{}/{} pi_loss: {} v_loss: {}", i, batch_count, pi_losses, v_losses));
                }

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
        let (pi, v) = no_grad(|| {
            self.nnet.forward(&tensor_board, false)
        });
        let pi: Vec<f32> = pi.exp().to_device(Device::Cpu).view(-1).try_into().unwrap();
        let value: f32 = v.to_device(Device::Cpu).try_into().unwrap();
        let mut actions: [f32; 4] = [0.0; 4];
        actions.copy_from_slice(&pi);
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


    pub fn save_checkpoint(&self, model_path: &PathBuf) -> Result<(), std::io::Error> {
        if !model_path.exists() {
            std::fs::create_dir_all(model_path.parent().unwrap())?;
        }
        self.vs.save(model_path).unwrap_or_else(|e| {
            println!("Failed to save checkpoint: {}", e);
        });
        Ok(())
    }

    pub fn load_checkpoint(&mut self, model_path: &PathBuf) -> Result<(), std::io::Error> {
        self.vs.load(model_path).unwrap_or_else(|e| {
            println!("Failed to load checkpoint: {}", e);
        });
        Ok(())
    }
}

impl Clone for AlphaZeroModel {
    fn clone(&self) -> Self {
        let mut vs = get_base_var_store();
        vs.copy(&self.vs).unwrap();
        vs.trainable_variables();
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