use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use crate::Vector;

pub fn launch_matrix_multiply_cuda(a: &[f32], b: &[f32], shape_a: (usize, usize), shape_b: (usize, usize)) -> Result<Vec<f32>, Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let ptx = CString::new(include_str!("../../resources/dot.ptx"))?;
    let module = Module::load_from_string(&ptx)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let rows_a = shape_a.0 as u32;
    let cols_a = shape_a.1 as u32;
    let rows_b = shape_b.0 as u32;
    let cols_b = shape_b.1 as u32;
    let mut A = DeviceBuffer::from_slice(a)?;
    let mut B = DeviceBuffer::from_slice(b)?;
    let mut C = DeviceBuffer::from_slice(Vector::from_shape((rows_a, cols_b)).to_vec_f32().as_slice())?;

    unsafe {
        let block_dim = (cols_b, rows_a, 1);
        let grid_dim = ((cols_b + block_dim.0 - 1) / block_dim.0, (rows_a + block_dim.1 - 1) / block_dim.1, 1);
        let result = launch!(module.dot<<<grid_dim, block_dim, 0, stream>>>(
            A.as_device_ptr(),
            B.as_device_ptr(),
            C.as_device_ptr(),
            rows_a,
            cols_a,
            cols_b
        ));
        result?;
        println!("C : {:?}", C);
    }

    stream.synchronize()?;

    let mut C_host = Vector::from_shape((rows_a, cols_b)).to_vec_f32();
    C.copy_to(&mut C_host[..])?;

    let mut result = Vec::new();
    for x in C_host.iter() {
        result.push(*x);
    }

    Ok(result)
}