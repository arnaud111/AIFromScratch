use crate::math::vector::{*};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub w: Vector,
    pub b: Vector,
    activation: ActivationEnum
}

impl Layer {

    pub fn new(input: &u16, neurons_count: &u16, activation: &ActivationEnum) -> Layer {
        let mut w_layer: Vec<Vec<f64>> = Vec::new();
        let mut b_layer: Vec<Vec<f64>> = Vec::new();

        for _ in 0..*neurons_count {
            w_layer.push(create_random_vector(input));
            b_layer.push(create_random_vector(&1))
        }

        Layer {
            w: Vector::new(w_layer),
            b: Vector::new(b_layer),
            activation: (*activation).clone()
        }
    }
}