use crate::math::vector::{*};
use crate::deep_neural_network::activations::ActivationEnum;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub w: Vector,
    pub b: Vector,
    pub activation: ActivationEnum
}

impl Layer {

    pub fn new(input: &u16, neurons_count: &u16, activation: &ActivationEnum) -> Layer {
        let mut w_layer: Vec<Vec<f64>> = Vec::new();
        let mut b_layer: Vec<Vec<f64>> = Vec::new();

        for _ in 0..*neurons_count {
            w_layer.push(Layer::create_random_vec(input));
            b_layer.push(Layer::create_random_vec(&1))
        }

        Layer {
            w: Vector::new(w_layer),
            b: Vector::new(b_layer),
            activation: (*activation).clone()
        }
    }

    fn create_random_vec(size: &u16) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let distribution = Normal::new(0.0, 1.0).unwrap();
        let mut vector: Vec<f64> = Vec::new();

        for _ in 0..*size {
            let num = distribution.sample(&mut rng);
            vector.push(num);
        }

        vector
    }
}