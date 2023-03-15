use rand::Rng;
use rand_distr::{Distribution, Normal};

pub fn create_random_vector(size: u16) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let distribution = Normal::new(0.0, 1.0).unwrap();
    let mut vector: Vec<f64> = Vec::new();

    for _ in 0..size {
        let num = distribution.sample(&mut rng);
        vector.push(num);
    }

    vector
}

pub fn generate_number() -> f64 {
    let mut rng = rand::thread_rng();
    let distribution = Normal::new(0.0, 1.0).unwrap();
    let num = distribution.sample(&mut rng);
    num
}