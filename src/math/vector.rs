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

pub fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub fn transpose(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut b: Vec<Vec<f64>> = Vec::new();
    for i in 0..a[0].len() {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..a.len() {
            row.push(a[j][i]);
        }
        b.push(row);
    }
    b
}