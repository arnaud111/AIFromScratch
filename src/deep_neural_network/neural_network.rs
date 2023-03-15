use crate::create_random_vector;
use crate::math::vector::generate_number;

pub(crate) fn initialization(x: &[f64]) -> (Vec<f64>, f64) {
    let w =create_random_vector(x.len());
    let b = generate_number();
    (w, b)
}
