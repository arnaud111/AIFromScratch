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

    pub fn init_layers(&mut self, layers_size: Vec<u16>, input: u16) {
        self.add_layer(input, layers_size[0]);
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

    pub fn forward_propagation(&self, input: &Vector) -> Vec<Vector> {
        let mut a: Vec<Vector> = Vec::new();
        a.push((*input).clone());
        a[0].display();

        for i in 0..self.w.len() {
            let z = self.w[i].dot(&a[i]).add(&self.b[i]);
            a.push(z.sigmoid());
            a[i + 1].display();
        }

        a
    }

    pub fn display_layers(&self) {
        for i in 0..self.w.len() {
            println!("Layer {}", i);
            self.w[i].display();
            self.b[i].display();
        }
    }
}
