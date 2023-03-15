use crate::deep_neural_network::neuron::Neuron;

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub input: u16
}

impl Layer {

    pub fn new(neurons: u16, input: u16) -> Layer {
        let mut layer = Layer {
            neurons: vec![],
            input
        };
        for _ in 0..neurons {
            layer.neurons.push(Neuron::new(input));
        }
        layer
    }

    pub fn get_probability(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut z = vec![];
        for i in 0..self.neurons.len() {
            z.push(self.neurons[i].get_probability(x));
        }
        z
    }
}