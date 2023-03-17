
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn tanh_derivative(x: f64) -> f64 {
    1.0 - x.powi(2)
}