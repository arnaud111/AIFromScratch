use crate::{create_random_vector, generate_number};

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


}
