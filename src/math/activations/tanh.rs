
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

pub fn tanh_derivative(x: f32) -> f32 {
    1.0 - x.powi(2)
}