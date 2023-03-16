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

#[derive(Clone)]
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

    pub fn sigmoid(&self) -> Vector {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.shape.0 {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..self.shape.1 {
                row.push(1.0 / (1.0 + (-self.data[i][j]).exp()));
            }
            result.push(row);
        }
        Vector::new(result)
    }
}
