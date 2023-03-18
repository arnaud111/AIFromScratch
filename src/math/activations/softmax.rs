use std::f64::consts::E;
use crate::Vector;

pub fn softmax(vec: &Vec<f64>) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();
    let mut sum = 0.0;
    for i in 0..vec.len() {
        let exp_val: f64 = E.powf(vec[i]);
        result.push(exp_val);
        sum += exp_val;
    }
    for i in 0..result.len() {
        result[i] /= sum;
    }
    result
}

pub fn softmax_derivative(vec: &Vec<f64>) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();
    for i in 0..vec.len() {
        let mut sum = 0.0;
        for j in 0..vec.len() {
            if i == j {
                sum += vec[j] * (1.0 - vec[j]);
            } else {
                sum += -vec[i] * vec[j];
            }
        }
        result.push(sum);
    }
    result
}