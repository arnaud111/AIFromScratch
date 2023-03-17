
pub fn leaky_relu(x: f64) -> f64 {
    if x < 0.0 {
        0.01 * x
    } else {
        x
    }
}

pub fn leaky_relu_derivative(x: f64) -> f64 {
    if x < 0.0 {
        0.01
    } else {
        1.0
    }
}