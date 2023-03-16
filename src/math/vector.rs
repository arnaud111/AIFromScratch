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

pub fn add(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut c: Vec<Vec<f64>> = Vec::new();
    for i in 0..a.len() {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..a[i].len() {
            row.push(a[i][j] + b[i][j]);
        }
        c.push(row);
    }
    c
}

pub fn subtract(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut c: Vec<Vec<f64>> = Vec::new();
    for i in 0..a.len() {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..a[i].len() {
            row.push(a[i][j] - b[i][j]);
        }
        c.push(row);
    }
    c
}

pub fn dot(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut c: Vec<Vec<f64>> = Vec::new();
    for i in 0..a.len() {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..b[0].len() {
            let mut sum = 0.0;
            for k in 0..a[i].len() {
                sum += a[i][k] * b[k][j];
            }
            row.push(sum);
        }
        c.push(row);
    }
    c
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn softmax(x: &Vec<f64>) -> Vec<f64> {
    let mut sum = 0.0;
    let mut y: Vec<f64> = Vec::new();
    for i in 0..x.len() {
        sum += x[i].exp();
    }
    for i in 0..x.len() {
        y.push(x[i].exp() / sum);
    }
    y
}

pub fn softmax_derivative(x: &Vec<f64>) -> Vec<f64> {
    let mut y: Vec<f64> = Vec::new();
    for i in 0..x.len() {
        let mut sum = 0.0;
        for j in 0..x.len() {
            if i == j {
                sum += x[j] * (1.0 - x[j]);
            } else {
                sum += -x[i] * x[j];
            }
        }
        y.push(sum);
    }
    y
}

pub fn cross_entropy(y: &Vec<f64>, y_hat: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..y.len() {
        sum += y[i] * y_hat[i].ln();
    }
    -sum
}

pub fn cross_entropy_derivative(y: &Vec<f64>, y_hat: &Vec<f64>) -> Vec<f64> {
    let mut y_hat_derivative: Vec<f64> = Vec::new();
    for i in 0..y.len() {
        y_hat_derivative.push(-y[i] / y_hat[i]);
    }
    y_hat_derivative
}

pub fn mean_squared_error(y: &Vec<f64>, y_hat: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..y.len() {
        sum += (y[i] - y_hat[i]).powi(2);
    }
    sum / (y.len() as f64)
}

pub fn mean_squared_error_derivative(y: &Vec<f64>, y_hat: &Vec<f64>) -> Vec<f64> {
    let mut y_hat_derivative: Vec<f64> = Vec::new();
    for i in 0..y.len() {
        y_hat_derivative.push(2.0 * (y_hat[i] - y[i]) / (y.len() as f64));
    }
    y_hat_derivative
}