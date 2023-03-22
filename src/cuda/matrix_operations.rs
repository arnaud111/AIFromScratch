use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use crate::Vector;

pub fn launch_matrix_multiply_cuda(a: &[f32], b: &[f32], shape_a: (usize, usize), shape_b: (usize, usize)) -> Result<Vec<f32>, Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let ptx = CString::new(include_str!("../../resources/dot1d.ptx")).expect("Failed to create CString");
    let module = Module::load_from_string(&ptx).expect("Failed to load module");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create stream");

    let rows_a = shape_a.0 as u32;
    let cols_a = shape_a.1 as u32;
    let rows_b = shape_b.0 as u32;
    let cols_b = shape_b.1 as u32;
    let mut A = DeviceBuffer::from_slice(a).expect("Failed to create device buffer");
    let mut B = DeviceBuffer::from_slice(b).expect("Failed to create device buffer");
    let mut result = Vec::new();

    for i in 0..4 {

        let mut C = DeviceBuffer::from_slice(&[0.0; 10000]).expect("Failed to create device buffer");

        unsafe {
            let block_dim = (100, 1, 1);
            let grid_dim = (100, 1, 1);
            let result = launch!(module.dot<<<grid_dim, block_dim, 0, stream>>>(
                A.as_device_ptr(),
                B.as_device_ptr(),
                C.as_device_ptr(),
                rows_a,
                cols_a,
                cols_b,
                i * 100 * 100
            ));
            result.expect("Failed to launch kernel");
        }
        stream.synchronize().expect("Failed to synchronize stream");

        let mut C_host = Vector::from_shape((100, 100)).to_vec_f32();
        C.copy_to(&mut C_host[..]).expect("Failed to copy data to host");

        for x in C_host.iter() {
            result.push(*x);
        }
    }

    Ok(result)
}