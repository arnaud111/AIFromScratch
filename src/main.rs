#[macro_use]
extern crate rustacuda;

use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use crate::cuda::matrix_operations::launch_matrix_multiply_cuda;
use crate::data::dataset::{convert_y, load_dataset_csv};
use crate::deep_neural_network::activations::ActivationEnum;
use crate::deep_neural_network::network::Network;
use crate::math::vector::{*};

mod deep_neural_network;
mod math;
mod data;
mod cuda;

fn main() {
    let (mut x, mut y) = load_dataset_csv("mnist");
    y = convert_y(&y);
    let x_test = x.sub_vector(50000, 60000);
    let y_test = y.sub_vector(50000, 60000);
    x = x.sub_vector(0, 1000);
    y = y.sub_vector(0, 1000);
    create_network(x, y, x_test, y_test);
}

fn load_network(x: Vector, y: Vector, x_test: Vector, y_test: Vector) {
    let mut network = Network::load("network");
    network.train(&x, &y, &x_test, &y_test, 50, 0.1, 50, true);
    println!("Accuracy : {}", network.accuracy(&x_test, &y_test));
    network.save("network");
}

fn create_network(x: Vector, y: Vector, x_test: Vector, y_test: Vector) {
    let mut network = Network::new();
    let layers = vec![
        (784, ActivationEnum::Sigmoid),
        (10, ActivationEnum::Softmax),
    ];
    network.init_layers(layers, x.shape.0 as u16);
    network.train(&x, &y, &x_test, &y_test, 500, 0.1, 50, true);
    println!("Accuracy : {}", network.accuracy(&x_test, &y_test));
    network.save("network");
}
