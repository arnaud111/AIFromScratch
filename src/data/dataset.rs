use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::Vector;

pub fn load_dataset_csv(file_name: &str) -> (Vector, Vector) {
    let mut x = Vec::new();
    let mut y= Vec::new();
    let file = File::open(format!("dataset/{}.csv", file_name)).unwrap();
    let reader = BufReader::new(file);
    let mut target_index = 0;

    for line in reader.lines() {
        let line_tmp = line.unwrap().clone();
        let split: Vec<&str> = line_tmp.split(",").collect();
        if target_index == 0 {
            for i in 0..split.len() {
                if split[i] == "target" {
                    target_index = i;
                    y.push(Vec::new())
                } else {
                    x.push(Vec::new())
                }
            }
        } else {
            for i in 0..split.len() {
                if i == target_index {
                    y[0].push(split[i].parse::<f64>().unwrap());
                } else {
                    x[i].push(split[i].parse::<f64>().unwrap());
                }
            }
        }
    }

    (Vector::new(x), Vector::new(y))
}