
pub fn relu(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

pub fn relu_derivative(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        1.0
    }
}