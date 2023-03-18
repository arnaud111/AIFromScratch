use crate::math::vector::{*};
use serde::{Deserialize, Serialize};
use crate::math::activations::leaky_relu::{leaky_relu, leaky_relu_derivative};
use crate::math::activations::relu::{relu, relu_derivative};
use crate::math::activations::sigmoid::{sigmoid, sigmoid_derivative};
use crate::math::activations::softmax::{softmax, softmax_derivative};
use crate::math::activations::tanh::{tanh, tanh_derivative};

#[derive(Serialize, Deserialize, Clone)]
pub enum ActivationEnum {
    Sigmoid,
    Relu,
    Tanh,
    LeakyRelu,
    Softmax
}

impl ActivationEnum {

    pub fn compute(&self, vec: &Vector) -> Vector {
        return match self {
            ActivationEnum::Sigmoid => vec.apply(sigmoid),
            ActivationEnum::Relu => vec.apply(relu),
            ActivationEnum::Tanh => vec.apply(tanh),
            ActivationEnum::LeakyRelu => vec.apply(leaky_relu),
            ActivationEnum::Softmax => vec.apply_to_vec(softmax),
        }
    }

    pub fn derived(&self, vec: &Vector) -> Vector {
        return match self {
            ActivationEnum::Sigmoid => vec.apply(sigmoid_derivative),
            ActivationEnum::Relu => vec.apply(relu_derivative),
            ActivationEnum::Tanh => vec.apply(tanh_derivative),
            ActivationEnum::LeakyRelu => vec.apply(leaky_relu_derivative),
            ActivationEnum::Softmax => vec.apply_to_vec(softmax_derivative),
        }
    }
}