use crate::{create_random_vector, generate_number};

pub struct Network {
    w: Vec<Vec<Vec<f64>>>,
    b: Vec<Vec<f64>>
}

impl Network {

    pub fn new(layers_size: Vec<u16>, input: &Vec<f64>) -> Network {
        let mut w: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut b: Vec<Vec<f64>> = Vec::new();

        let mut w_layer: Vec<Vec<f64>> = Vec::new();
        let mut b_layer: Vec<f64> = Vec::new();
        for _ in 0..layers_size[0] {
            w_layer.push(create_random_vector(input.len() as u16));
            b_layer.push(generate_number())
        }
        w.push(w_layer);
        b.push(b_layer);

        for i in 1..layers_size.len() {
            let mut w_layer: Vec<Vec<f64>> = Vec::new();
            let mut b_layer: Vec<f64> = Vec::new();
            for _ in 0..layers_size[i] {
                w_layer.push(create_random_vector(layers_size[i - 1]));
                b_layer.push(generate_number())
            }
            w.push(w_layer);
            b.push(b_layer);
        }

        Network {
            w,
            b
        }
    }


}
