use crate::math::vector::{*};

pub struct Network {
    w: Vec<Vector>,
    b: Vec<Vector>
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

    fn add_layer(&mut self, input: u16, neurons_count: u16) {
        let mut w_layer: Vec<Vec<f64>> = Vec::new();
        let mut b_layer: Vec<Vec<f64>> = Vec::new();

        for _ in 0..neurons_count {
            w_layer.push(create_random_vector(input));
            b_layer.push(create_random_vector(1))
        }

        self.w.push(Vector::new(w_layer));
        self.b.push(Vector::new(b_layer));
    }

    pub fn display_layers(&self) {
        for i in 0..self.w.len() {
            println!("Layer {}", i);
            println!("{:?}", self.w[i].shape());
            println!("{:?}", self.w[i].display());
            println!("{:?}", self.b[i].shape());
            println!("{:?}", self.b[i].display());
        }
    }
}
