use crate::deep_neural_network::layer::Layer;

pub struct Network {
    pub layers: Vec<Layer>
}

impl Network {

    pub fn new(layers_size: Vec<u16>, input: &[f64]) -> Network {
        let mut network = Network {
            layers: vec![]
        };

        network.layers.push(Layer::new(layers_size[0], input.len() as u16));
        for i in 1..layers_size.len() {
            network.layers.push(Layer::new(layers_size[i], layers_size[i - 1]));
        }
        network.layers.push(Layer::new(1, layers_size[layers_size.len() - 1]));

        network
    }

    pub fn get_probability(&self, x: Vec<f64>) -> f64 {
        let mut y = x;
        println!("y[0] = {:?}", y);
        for i in 0..self.layers.len() {
            y = self.layers[i].get_probability(&y);
            println!("y[{}] = {:?}", i, y);
        }
        y[0]
    }

}
