use std::fmt::format;
use crate::math::vector::{*};
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{Read, Write};
use crate::ActivationEnum;
use crate::deep_neural_network::layer::Layer;
use crate::math::activations::log::log;

#[derive(Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Layer>
}

impl Network {

    pub fn new() -> Network {
        Network {
            layers: Vec::new()
        }
    }

    pub fn init_layers(&mut self, layers: Vec<(u16, ActivationEnum)>, input: u16) {
        self.layers.push(Layer::new(&input, &layers[0].0, &layers[0].1));
        for i in 1..layers.len() {
            self.layers.push(Layer::new(&layers[i - 1].0, &layers[i].0, &layers[i].1));
        }
    }

    fn forward_propagation(&self, input: &Vector) -> Vec<Vector> {
        let mut a: Vec<Vector> = Vec::new();
        a.push((*input).clone());

        for i in 0..self.layers.len() {
            let z = self.layers[i].w.dot(&a[i]).add(&self.layers[i].b);
            a.push(self.layers[i].activation.compute(&z));
        }

        a
    }

    fn back_propagation(&self, y: &Vector, activations: &Vec<Vector>) -> (Vec<Vector>, Vec<Vector>) {
        let mut dw: Vec<Vector> = Vec::new();
        let mut db: Vec<Vector> = Vec::new();
        let m = y.shape.1 as f64;

        //let mut dz = activations[activations.len() - 1].apply(log).add(&y.number_sub(1.0).transpose().dot(&y.number_sub(1.0).apply(log)));
        let mut dz = activations[activations.len() - 1].sub(&y);

        for i in (0..self.layers.len()).rev() {
            let dw_layer = dz.dot(&activations[i].transpose()).div_by_number(m);
            let db_layer = dz.sum().div_by_number(m);

            if i > 0 {
                let da = self.layers[i - 1].activation.derived(&activations[i]);
                dz = self.layers[i].w.transpose().dot(&dz).multiply_one_by_one(&da);
            }

            dw.push(dw_layer);
            db.push(db_layer);
        }

        dw.reverse();
        db.reverse();

        (dw, db)
    }

    fn update(&mut self, dw: Vec<Vector>, db: Vec<Vector>, learning_rate: f64) {
        for i in 0..self.layers.len() {
            self.layers[i].w = self.layers[i].w.sub(&dw[i].mul_by_number(learning_rate));
            self.layers[i].b = self.layers[i].b.sub(&db[i].mul_by_number(learning_rate));
        }
    }

    fn get_accuracy_from_epoch(&self, activations: &Vec<Vector>, y: &Vector) -> f64 {
        let mut correct = 0;

        for i in 0..y.shape.1 {
            if self.get_index_max_probability(&activations[activations.len() - 1].get_column(i)) == self.get_index_max_probability(&y.get_column(i)) {
                correct += 1;
            }
        }

        correct as f64 / y.shape.1 as f64
    }

    pub fn train(&mut self, x: &Vector, y: &Vector, x_test: &Vector, y_test: &Vector, epochs: usize, learning_rate: f64, epochs_interval_test: usize, display: bool) {
        for i in 0..epochs {
            let activations = self.forward_propagation(x);
            let (dw, db) = self.back_propagation(y, &activations);
            self.update(dw, db, learning_rate);
            if display {
                println!("Epoch: {}, Train Accuracy: {}", i, self.get_accuracy_from_epoch(&activations, y));
            }
            if i % epochs_interval_test == 0 {
                println!("Test Accuracy : {}", self.accuracy(x_test, y_test));
            }
        }
    }

    pub fn probability(&self, input: &Vector) -> Vector {
        let mut a: Vector = (*input).clone();

        for i in 0..self.layers.len() {
            let z = self.layers[i].w.dot(&a).add(&self.layers[i].b);
            a = self.layers[i].activation.compute(&z);
        }

        a
    }

    pub fn predict(&self, input: &Vector) -> bool {
        return self.probability(input).data[0][0] > 0.5;
    }

    pub fn get_index_max_probability(&self, output: &Vector) -> usize {
        let mut max = 0.0;
        let mut index = 0;

        for i in 0..output.shape.0 {
            if output.data[i][0] > max {
                max = output.data[i][0];
                index = i;
            }
        }

        index
    }

    pub fn accuracy(&self, x: &Vector, y: &Vector) -> f64 {
        let mut correct = 0;

        for i in 0..x.shape.1 {
            if self.get_index_max_probability(&self.probability(&x.get_column(i))) == self.get_index_max_probability(&y.get_column(i)) {
                correct += 1;
            }
        }

        correct as f64 / y.shape.1 as f64
    }

    pub fn display_layers(&self) {
        for i in 0..self.layers.len() {
            println!("Layer {}", i);
            self.layers[i].w.display();
            self.layers[i].b.display();
        }
        println!();
    }

    pub fn save(&self, file_name: &str) {
        let mut file = File::create(format!("networks/{}.json", file_name)).unwrap();
        let serialized = serde_json::to_string(self).unwrap();
        file.write_all(serialized.as_bytes()).unwrap();
    }

    pub fn load(file_name: &str) -> Network {
        let mut file = File::open(format!("networks/{}.json", file_name)).unwrap();
        let mut serialized = String::new();
        file.read_to_string(&mut serialized).unwrap();
        serde_json::from_str(&serialized).unwrap()
    }
}
