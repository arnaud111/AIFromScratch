use serde::{Deserialize, Serialize};
use crate::launch_matrix_multiply_cuda;

#[derive(Clone, Serialize, Deserialize)]
pub struct Vector {
    pub data: Vec<Vec<f32>>,
    pub shape: (usize, usize)
}

impl Vector {

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn new(data: Vec<Vec<f32>>) -> Vector {
        let shape = (data.len(), data[0].len());
        Vector {
            data,
            shape
        }
    }

    pub fn display(&self) {
        println!("{:?} : {:?}", self.shape, self.data);
    }

    pub fn from_slice(slice: Vec<f32>, x: usize, y: usize) -> Vector {
        let mut data: Vec<Vec<f32>> = Vec::new();
        for i in 0..x {
            let mut row: Vec<f32> = Vec::new();
            for j in 0..y {
                row.push(slice[i * y + j] as f32);
            }
            data.push(row);
        }
        Vector::new(data)
    }

    pub fn from_shape(shape: (u32, u32)) -> Vector {
        let mut data: Vec<Vec<f32>> = Vec::new();
        for i in 0..shape.0 {
            let mut row: Vec<f32> = Vec::new();
            for j in 0..shape.1 {
                row.push(0.0);
            }
            data.push(row);
        }
        Vector::new(data)
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        let mut vec: Vec<f32> = Vec::new();
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                vec.push(self.data[i][j] as f32);
            }
        }
        vec
    }

    pub fn transpose(&self) -> Vector {
        let mut transposed: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.1 {
            let mut row: Vec<f32> = Vec::new();
            for j in 0..self.shape.0 {
                row.push(self.data[j][i]);
            }
            transposed.push(row);
        }
        Vector::new(transposed)
    }

    pub fn dot(&self, other: &Vector) -> Vector {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
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

    pub fn dot_cuda(&self, other: &Vector) -> Vector {
        // self.display();
        // other.display();
        let result = launch_matrix_multiply_cuda(self.to_vec_f32().as_slice(), other.to_vec_f32().as_slice(), self.shape, other.shape).expect("failed to launch cuda");
        Vector::from_slice(result, self.shape.0, other.shape.1)
    }

    pub fn add(&self, other: &Vector) -> Vector {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
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
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
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

    pub fn number_sub(&self, n: f32) -> Vector {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(n - self.data[i][j]);
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn div_by_number(&self, n: f32) -> Vector {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(self.data[i][j] / n);
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn mul_by_number(&self, n: f32) -> Vector {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(self.data[i][j] * n);
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn sum(&self) -> Vector {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
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
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(self.data[i][j] * other.data[i][j]);
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn get_column(&self, index: usize) -> Vector {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
            row.push(self.data[i][index]);
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn get_column_as_vec(&self, index: usize) -> Vec<f32> {
        let mut result: Vec<f32> = Vec::new();
        for i in 0..self.shape.0 {
            result.push(self.data[i][index]);
        }
        result
    }

    pub fn apply(&self, func: fn(f32) -> f32) -> Vector {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f32> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(func(self.data[i][j]));
            }
            result.push(row);
        }
        Vector::new(result)
    }

    pub fn apply_to_vec(&self, func: fn(&Vec<f32>) -> Vec<f32>) -> Vector {
        let mut result: Vec<Vec<f32>> = Vec::new();
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
        let mut result: Vec<Vec<f32>> = Vec::new();
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
