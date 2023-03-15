use crate::math::vector::{create_random_vector, generate_number};

pub struct Neuron {
    pub w: Vec<f64>,
    pub b: f64,
}

impl Neuron {
    pub fn new(x: u16) -> Neuron {
        Neuron {
            w: create_random_vector(x),
            b: generate_number()
        }
    }

    pub fn get_probability(&self, x: &Vec<f64>) -> f64 {
        let mut z = 0.0;
        for i in 0..x.len() {
            z += x[i] * self.w[i];
        }
        z += self.b;
        1.0 / (1.0 + (-z).exp())
    }

    pub fn update(&mut self, dw: &[f64], db: f64, learning_rate: f64) {
        for i in 0..self.w.len() {
            self.w[i] -= learning_rate * dw[i];
        }
        self.b -= learning_rate * db;
    }
}

