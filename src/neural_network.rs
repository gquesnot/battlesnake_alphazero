use tch::nn;
use tch::nn::{ConvConfig, Module, ModuleT};

use crate::config::{ACTION_SIZE, BOARD_SIZE, DROPOUT};

pub struct NeuralNetwork {
    seq: nn::SequentialT,
    fc_v: nn::Linear,
    fc_pi: nn::Linear,
}

impl NeuralNetwork {
    pub(crate) fn new(vs: &nn::Path, num_channels: i64) -> NeuralNetwork {
        let stride = ConvConfig { stride: 1, ..Default::default() };
        let stride_padding = ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let seq = nn::seq_t()
            .add_fn(move |xs| xs.view([-1, 1, BOARD_SIZE, BOARD_SIZE]))
            .add(nn::conv2d(vs / "conv1", 1, num_channels, 3, stride_padding))
            .add(nn::batch_norm2d(vs / "bn1", num_channels, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs / "conv2", num_channels, num_channels, 3, stride_padding))
            .add(nn::batch_norm2d(vs / "bn2", num_channels, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs / "conv3", num_channels, num_channels, 3, stride))
            .add(nn::batch_norm2d(vs / "bn3", num_channels, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs / "conv4", num_channels, num_channels, 3, stride))
            .add(nn::batch_norm2d(vs / "bn4", num_channels, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn(move |xs| xs.view([-1, num_channels * (BOARD_SIZE - 4) * (BOARD_SIZE - 4)]))
            .add(nn::linear(vs / "fc1", num_channels * (BOARD_SIZE - 4) * (BOARD_SIZE - 4), 1024, Default::default()))
            .add(nn::batch_norm1d(vs / "fc1_bn", 1024, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn_t(|xs, train| xs.dropout(DROPOUT, train))
            .add(nn::linear(vs / "fc2", 1024, 512, Default::default()))
            .add(nn::batch_norm1d(vs / "fc2_bn", 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn_t(|xs, train| xs.dropout(DROPOUT, train));
        let fc_v = nn::linear(vs / "fc_v", 512, 1, Default::default());
        let fc_pi = nn::linear(vs / "fc_pi", 512, ACTION_SIZE, Default::default());


        NeuralNetwork {
            seq,
            fc_v,
            fc_pi,
        }
    }

    pub fn forward(&self, input: &tch::Tensor, is_training: bool) -> (tch::Tensor, tch::Tensor) {
        let x = self.seq.forward_t(input, is_training);
        let v = self.fc_v.forward(&x).tanh();
        let pi = self.fc_pi.forward(&x).log_softmax(1, tch::Kind::Float);
        (pi, v)
    }
}
