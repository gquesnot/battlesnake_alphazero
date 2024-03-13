use tch::nn;
use tch::nn::{BatchNorm, Conv2D, ConvConfig};

use crate::config::{ACTION_SIZE, BOARD_SIZE, DROPOUT, NUM_CHANNELS};

pub struct NeuralNetwork {
    conv1: Conv2D,
    conv2: Conv2D,
    conv3: Conv2D,
    conv4: Conv2D,

    bn1: BatchNorm,
    bn2: BatchNorm,
    bn3: BatchNorm,
    bn4: BatchNorm,

    fc1: nn::Linear,
    fc1_bn: BatchNorm,

    fc2: nn::Linear,
    fc2_bn: BatchNorm,

    fc_v: nn::Linear,
    fc_pi: nn::Linear,
}

impl NeuralNetwork {
    pub(crate) fn new(vs: &nn::Path) -> NeuralNetwork {
        let conv1 = nn::conv2d(vs, 1, NUM_CHANNELS, 3, ConvConfig { stride: 1, padding: 1, ..Default::default() });
        let conv2 = nn::conv2d(vs, NUM_CHANNELS, NUM_CHANNELS, 3, ConvConfig { stride: 1, padding: 1, ..Default::default() });
        let conv3 = nn::conv2d(vs, NUM_CHANNELS, NUM_CHANNELS, 3, ConvConfig { stride: 1, ..Default::default() });
        let conv4 = nn::conv2d(vs, NUM_CHANNELS, NUM_CHANNELS, 3, ConvConfig { stride: 1, ..Default::default() });

        let bn1 = nn::batch_norm2d(vs, NUM_CHANNELS, Default::default());
        let bn2 = nn::batch_norm2d(vs, NUM_CHANNELS, Default::default());
        let bn3 = nn::batch_norm2d(vs, NUM_CHANNELS, Default::default());
        let bn4 = nn::batch_norm2d(vs, NUM_CHANNELS, Default::default());

        let fc1 = nn::linear(vs, NUM_CHANNELS * (BOARD_SIZE - 4) * (BOARD_SIZE - 4), 1024, Default::default());
        let fc1_bn = nn::batch_norm1d(vs, 1024, Default::default());

        let fc2 = nn::linear(vs, 1024, 512, Default::default());
        let fc2_bn = nn::batch_norm1d(vs, 512, Default::default());

        let fc_v = nn::linear(vs, 512, 1, Default::default());
        let fc_pi = nn::linear(vs, 512, ACTION_SIZE, Default::default());

        NeuralNetwork {
            conv1,
            conv2,
            conv3,
            conv4,
            bn1,
            bn2,
            bn3,
            bn4,
            fc1,
            fc1_bn,
            fc2,
            fc2_bn,
            fc_v,
            fc_pi,
        }
    }

    pub fn forward(&self, input: &tch::Tensor, is_training: bool) -> (tch::Tensor, tch::Tensor) {
        let x = input.view([-1, 1, BOARD_SIZE, BOARD_SIZE])
            .apply(&self.conv1)
            .apply_t(&self.bn1, false)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, false)
            .relu()
            .apply(&self.conv3)
            .apply_t(&self.bn3, false)
            .relu()
            .apply(&self.conv4)
            .apply_t(&self.bn4, false)
            .relu()
            .view([-1, NUM_CHANNELS * (BOARD_SIZE - 4) * (BOARD_SIZE - 4)])
            .apply(&self.fc1)
            .apply_t(&self.fc1_bn,false)
            .relu()
            .dropout(DROPOUT, is_training)
            .apply(&self.fc2)
            .apply_t(&self.fc2_bn,false)
            .relu()
            .dropout(DROPOUT, is_training);
        let v = x.apply(&self.fc_v).tanh(); // Value function output
        let pi = x.apply(&self.fc_pi).log_softmax(1, tch::Kind::Float); // Policy function output
        (pi, v)
    }
}
