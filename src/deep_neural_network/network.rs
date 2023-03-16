use crate::math::vector::{*};

pub struct Network {
    w: Vec<Vector>,
    b: Vec<Vector>
}

impl Network {

    pub fn new() -> Network {
        Network {
            w: Vec::new(),
            b: Vec::new()
        }
    }

    pub fn init_layers(&mut self, layers_size: Vec<u16>, input: u16) {
        self.add_layer(input, layers_size[0]);
        for i in 1..layers_size.len() {
            self.add_layer(layers_size[i - 1], layers_size[i]);
        }
    }

    fn add_layer(&mut self, input: u16, neurons_count: u16) {
        let mut w_layer: Vec<Vec<f64>> = Vec::new();
        let mut b_layer: Vec<Vec<f64>> = Vec::new();

        for _ in 0..neurons_count {
            w_layer.push(create_random_vector(input));
            b_layer.push(create_random_vector(1))
        }

        self.w.push(Vector::new(w_layer));
        self.b.push(Vector::new(b_layer));
    }

    fn forward_propagation(&self, input: &Vector) -> Vec<Vector> {
        let mut a: Vec<Vector> = Vec::new();
        a.push((*input).clone());

        for i in 0..self.w.len() {
            let z = self.w[i].dot(&a[i]).add(&self.b[i]);
            a.push(z.sigmoid());
        }

        a
    }

    fn back_propagation(&self, y: &Vector, activations: &Vec<Vector>) -> (Vec<Vector>, Vec<Vector>) {
        let mut dw: Vec<Vector> = Vec::new();
        let mut db: Vec<Vector> = Vec::new();
        let m = y.shape.1 as f64;

        let mut dz = activations[activations.len() - 1].sub(&y);

        for i in (0..self.w.len()).rev() {
            let dw_layer = dz.dot(&activations[i].transpose()).div_by_number(m);
            let db_layer = dz.sum().div_by_number(m);

            if i > 0 {
                dz = self.w[i].transpose().dot(&dz).multiply_one_by_one(&activations[i]).multiply_one_by_one(&activations[i].number_sub(1.0));
            }

            dw.push(dw_layer);
            db.push(db_layer);
        }

        dw.reverse();
        db.reverse();

        (dw, db)
    }

    fn update(&mut self, dw: Vec<Vector>, db: Vec<Vector>, learning_rate: f64) {
        for i in 0..self.w.len() {
            self.w[i] = self.w[i].sub(&dw[i].mul_by_number(learning_rate));
            self.b[i] = self.b[i].sub(&db[i].mul_by_number(learning_rate));
        }
    }

    fn get_accuracy_from_epoch(&self, activations: &Vec<Vector>, y: &Vector) -> f64 {
        let mut correct = 0;

        for i in 0..y.shape.1 {
            if (activations[activations.len() - 1].data[0][i] > 0.5) == (y.data[0][i] > 0.5) {
                correct += 1;
            }
        }

        correct as f64 / y.shape.1 as f64
    }

    pub fn train(&mut self, x: &Vector, y: &Vector, epochs: usize, learning_rate: f64, display: bool) {
        for i in 0..epochs {
            let activations = self.forward_propagation(x);
            let (dw, db) = self.back_propagation(y, &activations);
            self.update(dw, db, learning_rate);
            if display {
                println!("Epoch: {}, Accuracy: {}", i, self.get_accuracy_from_epoch(&activations, y));
            }
        }
    }

    pub fn probability(&self, input: &Vector) -> Vector {
        let mut a: Vector = (*input).clone();

        for i in 0..self.w.len() {
            let z = self.w[i].dot(&a).add(&self.b[i]);
            a = z.sigmoid();
        }

        a
    }

    pub fn predict(&self, input: &Vector) -> bool {
        return self.probability(input).data[0][0] > 0.5;
    }

    pub fn accuracy(&self, x: &Vector, y: &Vector) -> f64 {
        let mut correct = 0;

        for i in 0..x.shape.1 {
            if self.predict(&x.get_column(i)) == (y.data[0][i] > 0.5) {
                correct += 1;
            }
        }

        correct as f64 / y.shape.1 as f64
    }

    pub fn display_layers(&self) {
        for i in 0..self.w.len() {
            println!("Layer {}", i);
            self.w[i].display();
            self.b[i].display();
        }
        println!();
    }
}
