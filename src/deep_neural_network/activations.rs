use crate::math::vector::{*};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub enum ActivationEnum {
    Sigmoid,
    Relu,
    Tanh,
    LeakyRelu,
    Switch
}

impl ActivationEnum {

    pub fn compute(&self, vec: Vector) -> Vector {
        return match self {
            ActivationEnum::Sigmoid => vec.sigmoid(),
            ActivationEnum::Relu => vec.relu(),
            ActivationEnum::Tanh => vec.tanh(),
            ActivationEnum::LeakyRelu => vec.leaky_relu(),
            ActivationEnum::Switch => vec.switch()
        }
    }

    pub fn derived(&self, vec: &Vector) -> Vector {
        return match self {
            ActivationEnum::Sigmoid => vec.sigmoid_derivative(),
            ActivationEnum::Relu => vec.sigmoid_derivative(),
            ActivationEnum::Tanh => vec.sigmoid_derivative(),
            ActivationEnum::LeakyRelu => vec.sigmoid_derivative(),
            ActivationEnum::Switch => vec.sigmoid_derivative()
        }
    }
}