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
    let mut a = Vector::new(vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]]);
    let mut b = Vector::new(vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]]);
    a.display();
    b.display();
    a.dot_cuda(&b).display();
    /*
    let (mut x, mut y) = load_dataset_csv("mnist");
    y = convert_y(&y);
    let x_test = x.sub_vector(50000, 60000);
    let y_test = y.sub_vector(50000, 60000);
    x = x.sub_vector(0, 1000);
    y = y.sub_vector(0, 1000);
    create_network(x, y, x_test, y_test);*/
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

fn launch_add_cuda() -> Result<(), Box<dyn Error>> {
    // Set up the context, load the module, and create a stream to run kernels in.
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    let module = Module::load_from_string(&ptx)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create buffers for data
    let mut in_x = DeviceBuffer::from_slice(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32])?;
    let mut in_y = DeviceBuffer::from_slice(&[2.0f32; 10])?;
    let mut out_1 = DeviceBuffer::from_slice(&[0.0f32; 10])?;

    // This kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
    unsafe {
        // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
        let result = launch!(module.sum<<<1, 1, 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_1.as_device_ptr(),
            out_1.len()
        ));
        result?;
        println!("out1 : {:?}", out_1);
    }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;

    // Copy the results back to host memory
    let mut out_host = [0.0f32; 10];
    out_1.copy_to(&mut out_host[0..10])?;

    for x in out_host.iter() {
        println!("x : {}", x);
    }

    println!("Launched kernel successfully.");
    Ok(())
}
