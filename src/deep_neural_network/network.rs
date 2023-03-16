use crate::{create_random_vector, generate_number, dot, transpose};

pub struct Network {
    w: Vec<Vec<Vec<f64>>>,
    b: Vec<Vec<f64>>
}

impl Network {

    pub fn new() -> Network {
        Network {
            w: Vec::new(),
            b: Vec::new()
        }
    }

    pub fn init_layers(&mut self, layers_size: Vec<u16>, input: &Vec<f64>) {
        self.add_layer(input.len() as u16, layers_size[0]);
        for i in 1..layers_size.len() {
            self.add_layer(layers_size[i - 1], layers_size[i]);
        }
    }

    fn forward_propagation(&self, input: &Vec<f64>) -> Vec<Vec<f64>> {
        let mut output: Vec<f64> = Vec::new();
        let mut input = input.clone();
        let mut activations: Vec<Vec<f64>> = Vec::new();
        activations.push(input.clone());

        for i in 0..self.w.len() {
            for j in 0..self.w[i].len() {
                let mut sum = 0.0;
                for k in 0..self.w[i][j].len() {
                    sum += self.w[i][j][k] * input[k];
                }
                sum += self.b[i][j];
                output.push(sum);
            }
            input = output.clone();
            activations.push(input.clone());
            output.clear();
        }

        activations
    }

    fn back_propagation(&self, activations: Vec<Vec<f64>>, real_output: f64) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {

        let mut dz = activations[activations.len() - 1][0] - real_output;
        let mut dw = Vec::new();
        let mut db = Vec::new();

        for i in (0..self.w.len()).rev() {
            let mut dw_layer = Vec::new();
            let mut db_layer = Vec::new();

            for j in 0..activations[i].len() {
                let mut dw_neuron = Vec::new();
                for k in 0..activations[i].len() {
                    let dw = dz * activations[i][j];
                    dw_neuron.push(dw);
                }
                dw_layer.push(dw_neuron);
            }

            dw.push(dw_layer);
            db.push(db_layer);
        }

        println!("dw: {:?}", dw);
        println!("db: {:?}", db);

        (dw, db)
    }

    fn update(&mut self, dw: Vec<Vec<Vec<f64>>>, db: Vec<Vec<f64>>, learning_rate: f64) {
        for i in 0..self.w.len() {
            for j in 0..self.w[i].len() {
                for k in 0..self.w[i][j].len() {
                    self.w[i][j][k] -= dw[i][j][k] * learning_rate;
                }
                self.b[i][j] -= db[i][j] * learning_rate;
            }
        }
    }

    fn sigmoid_derivative(&self, x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn add_layer(&mut self, input: u16, neurons_count: u16) {
        let mut w_layer: Vec<Vec<f64>> = Vec::new();
        let mut b_layer: Vec<f64> = Vec::new();

        for _ in 0..neurons_count {
            w_layer.push(create_random_vector(input));
            b_layer.push(generate_number())
        }

        self.w.push(w_layer);
        self.b.push(b_layer);
    }

    pub fn train(&mut self, input: &Vec<Vec<f64>>, real_output: &Vec<bool>, epochs: u16, learning_rate: f64) {
        for i in 0..epochs {
            println!("Epoch {} / {}", i, epochs);
            self.train_epoch(input, real_output, learning_rate);
        }
    }

    fn train_epoch(&mut self, input: &Vec<Vec<f64>>, real_output: &Vec<bool>, learning_rate: f64) {
        for i in 0..input.len() {
            let activations = self.forward_propagation(&input[i]);
            let (dw, db) = self.back_propagation(activations, real_output[i] as i64 as f64);
            self.update(dw, db, learning_rate);
        }
    }

    pub fn predict(&self, input: &Vec<f64>) -> bool {
        let activations = self.forward_propagation(input);
        activations[activations.len() - 1][0] > 0.5
    }

    pub fn accuracy(&self, input: &Vec<Vec<f64>>, real_output: &Vec<bool>) -> f64 {
        let mut correct = 0;
        for i in 0..input.len() {
            if self.predict(&input[i]) == real_output[i] {
                correct += 1;
            }
        }
        correct as f64 / input.len() as f64
    }

    pub fn display_layers(&self) {
        for i in 0..self.w.len() {
            println!("Layer {}", i);
            for j in 0..self.w[i].len() {
                println!(" - Neuron {}", j);
                println!("    - Weights: {:?}", self.w[i][j]);
                println!("    - Bias: {}", self.b[i][j]);
            }
        }
    }
}
