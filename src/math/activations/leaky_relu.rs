
pub fn leaky_relu(x: f32) -> f32 {
    if x < 0.0 {
        0.01 * x
    } else {
        x
    }
}

pub fn leaky_relu_derivative(x: f32) -> f32 {
    if x < 0.0 {
        0.01
    } else {
        1.0
    }
}