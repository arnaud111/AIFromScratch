use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Vector {
    pub data: Vec<Vec<f64>>,
    pub shape: (usize, usize)
}

impl Vector {

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn new(data: Vec<Vec<f64>>) -> Vector {
        let shape = (data.len(), data[0].len());
        Vector {
            data,
            shape
        }
    }

    pub fn display(&self) {
        println!("{:?} : {:?}", self.shape, self.data);
    }

    pub fn transpose(&self) -> Vector {
        let mut transposed: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.1 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..self.shape.0 {
                row.push(self.data[j][i]);
            }
            transposed.push(row);
        }
        Vector::new(transposed)
    }

    pub fn dot(&self, other: &Vector) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..other.shape.1 {
                let mut sum = 0.0;
                for k in 0..self.shape.1 {
                    sum += self.data[i][k] * other.data[k][j];
                }
                row.push(sum);
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn add(&self, other: &Vector) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..self.shape.1 {
                if other.shape.1 == 1 {
                    row.push(self.data[i][j] + other.data[i][0])
                } else {
                    row.push(self.data[i][j] + other.data[i][j]);
                }
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn sub(&self, other: &Vector) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..self.shape.1 {
                if other.shape.1 == 1 {
                    row.push(self.data[i][j] - other.data[i][0])
                } else {
                    row.push(self.data[i][j] - other.data[i][j]);
                }
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn number_sub(&self, n: f64) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(n - self.data[i][j]);
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn div_by_number(&self, n: f64) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(self.data[i][j] / n);
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn mul_by_number(&self, n: f64) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(self.data[i][j] * n);
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn sum(&self) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            let mut sum = 0.0;
            for j in 0..self.shape.1 {
                sum += self.data[i][j];
            }
            row.push(sum);
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn multiply_one_by_one(&self, other: &Vector) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(self.data[i][j] * other.data[i][j]);
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn get_column(&self, index: usize) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            row.push(self.data[i][index]);
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn get_column_as_vec(&self, index: usize) -> Vec<f64> {
        let mut result: Vec<f64> = Vec::new();
        for i in 0..self.shape.0 {
            result.push(self.data[i][index]);
        }
        result
    }

    pub fn apply(&self, func: fn(f64) -> f64) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(func(self.data[i][j]));
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn apply_to_vec(&self, func: fn(&Vec<f64>) -> Vec<f64>) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            result.push(vec![]);
        }
        for i in 0..self.shape.1 {
            let vec = func(&self.get_column_as_vec(i));
            for j in 0..self.shape.0 {
                result[j].push(vec[j]);
            }
        }

        Vector::new(result)
    }

    pub fn sub_vector(&self, start: usize, end: usize) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            result.push(Vec::new());
        }
        for i in start..end {
            for j in 0..self.shape.0 {
                result[j].push(self.data[j][i]);
            }
        }
        Vector::new(result)
    }
}
