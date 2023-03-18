use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::Vector;

pub fn load_dataset_csv(file_name: &str) -> (Vector, Vector) {
    let mut x = Vec::new();
    let mut y= Vec::new();
    let file = File::open(format!("dataset/{}.csv", file_name)).unwrap();
    let reader = BufReader::new(file);
    let mut target_index = Vec::new();

    for line in reader.lines() {
        let line_tmp = line.unwrap().clone();
        let split: Vec<&str> = line_tmp.split(",").collect();
        if target_index.len() == 0 {
            for i in 0..split.len() {
                if split[i].contains("target") {
                    target_index.push(i);
                    y.push(Vec::new())
                } else {
                    x.push(Vec::new())
                }
            }
        } else {
            let mut x_index = 0;
            let mut y_index = 0;
            for i in 0..split.len() {
                if target_index.contains(&i) {
                    y[y_index].push(split[i].parse::<f64>().unwrap());
                    y_index += 1;
                } else {
                    x[x_index].push(split[i].parse::<f64>().unwrap());
                    x_index += 1;
                }
            }
        }
    }

    (Vector::new(x), Vector::new(y))
}

pub fn convert_y(y: &Vector) -> Vector {
    let mut y_new = Vec::new();
    for i in 0..y.shape().1 {
        let mut row = Vec::new();
        for j in 0..10 {
            if y.data[0][i] == j as f64 {
                row.push(1.0);
            } else {
                row.push(0.0);
            }
        }
        y_new.push(row);
    }
    Vector::new(y_new)
}