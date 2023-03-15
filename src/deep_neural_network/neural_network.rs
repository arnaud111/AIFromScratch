use crate::create_random_vector;
use crate::math::vector::generate_number;

pub struct Neuron {
    pub w: Vec<f64>,
    pub b: f64,
}

impl Neuron {
    pub fn new(x: &[f64]) -> Neuron {
        Neuron {
            w: create_random_vector(x.len()),
            b: generate_number()
        }
    }

    pub fn get_probability(&self, x: &[f64]) -> f64 {
        let mut y = 0.0;
        for i in 0..x.len() {
            y += x[i] * self.w[i];
        }
        y += self.b;
        1.0 / (1.0 + (-y).exp())
    }

    pub fn predict(&self, x: &[f64]) -> bool {
        self.get_probability(x) > 0.5
    }

    pub fn accuracy(&self, x: &Vec<&[f64]>, y: &Vec<bool>) -> f64 {

        let mut correct = 0;
        for i in 0..x.len() {
            if self.predict(&x[i]) == y[i] {
                correct += 1;
            }
        }
        correct as f64 / x.len() as f64
    }

    fn gradient(&self, x: &[f64], y: bool) -> (Vec<f64>, f64) {
        let p = self.get_probability(x);
        let mut dw = vec![0.0; x.len()];
        for i in 0..x.len() {
            dw[i] = (p - y as i64 as f64) * x[i];
        }
        let db = p - y as i64 as f64;
        (dw, db)
    }

    fn update(&mut self, dw: &[f64], db: f64, learning_rate: f64) {
        for i in 0..self.w.len() {
            self.w[i] -= learning_rate * dw[i];
        }
        self.b -= learning_rate * db;
    }

    pub fn train(&mut self, x: &Vec<&[f64]>, y: &Vec<bool>, learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            for i in 0..x.len() {
                let (dw, db) = self.gradient(x[i], y[i]);
                self.update(&dw, db, learning_rate);
                println!("a: {}", self.accuracy(&x, &y));
            }
        }
    }
}

