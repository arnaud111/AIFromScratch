use crate::data::dataset::load_dataset_csv;
use crate::deep_neural_network::activations::ActivationEnum;
use crate::deep_neural_network::network::Network;
use crate::math::vector::{*};

mod deep_neural_network;
mod math;
mod data;

fn main() {
    let (x, y) = load_dataset_csv("mnist");
    create_network(x, y);
}

fn load_network() {
    let (x, y) = create_data();
    let mut network = Network::load("network");
    println!("Accuracy : {}", network.accuracy(&x, &y));
}

fn create_network(x: Vector, y: Vector) {
    let mut network = Network::new();
    let layers = vec![
        (16384, ActivationEnum::Sigmoid),
        (128, ActivationEnum::Sigmoid),
        (128, ActivationEnum::Sigmoid),
        (128, ActivationEnum::Sigmoid),
        (128, ActivationEnum::Sigmoid),
        (10, ActivationEnum::Sigmoid)
    ];
    network.init_layers(layers, x.shape.0 as u16);
    println!("Accuracy : {}", network.accuracy(&x, &y));
    network.train(&x, &y, 1000, 0.1, true);
    println!("Accuracy : {}", network.accuracy(&x, &y));
    network.save("network");
}

fn create_data() -> (Vector, Vector) {
    let x = Vector::new(vec![vec![4.21850347, 0.90779887, -0.27652528, 0.08848433,
                                  3.24329731, 1.44193252, 1.0220286, 3.97820955, 0.58894326,
                                  1.25185786, 0.62835793, 1.68608568, 1.18454506, 0.06897171,
                                  1.78726415, 4.4384123, 3.18190344, 0.30380963, 0.73936011,
                                  1.28535145, 1.1312175, 0.66471755, 1.31570453, -0.18887976,
                                  2.57854418, 0.87305123, 1.3105127, 0.9867701, 1.42013331,
                                  2.3535057, 2.43169305, 1.0427873, 0.95088418, 2.45127423,
                                  1.62011397, 2.15504965, 1.38093486, 1.98702592, 2.11567076,
                                  0.56400993, -0.07228289, 2.50904929, -0.73000011, 1.37861172,
                                  0.88214412, 2.22194102, 2.0159847, 1.70127361, -0.65392827,
                                  0.57309313, 1.956815, 0.76241061, 2.46452227, 1.33263648,
                                  3.2460247, 1.10318217, 2.85942078, 1.59973502, 2.97612635,
                                  2.68049897, 1.41942144, 1.9263585, -0.09448254, 2.72756228,
                                  1.12031365, 1.55723507, 0.10547293, 2.84382807, 3.15492712,
                                  1.84070628, 1.28933778, 1.64164854, 3.92282648, 1.70536064,
                                  0.1631238, 0.34194798, 1.02102468, 2.66934689, 0.4666179,
                                  0.94808785, 1.19404184, 1.15369622, -0.29421492, 1.7373078,
                                  0.46546494, 0.08080352, 3.00251949, 2.20656076, 1.36069966,
                                  2.63185834, 2.82705807, 2.91209813, 1.1424453, 1.05505217,
                                  2.47034915, -1.57671974, 1.41164912, 3.00468833, -0.63762777,
                                  0.829832],
                             vec![2.23419161, 0.45984362, 5.08127768, 2.32299086, 1.21460627,
                                  2.76754364, 4.11660348, 2.37817845, 4.00148458, 0.20811388,
                                  4.4601363, 0.65828448, 5.28042636, 4.35573272, 1.70012006,
                                  1.84214315, -0.18226785, 3.94423417, 0.43607906, 1.43691285,
                                  4.68194985, 4.35995267, 2.44067826, 5.20461381, 0.72611733,
                                  4.71438583, 0.07122512, 6.08965782, 4.63746165, 2.22404956,
                                  -0.20173713, 4.60625923, 0.94982874, -0.19539785, 2.74692739,
                                  4.12386249, 0.92949422, 2.61100638, 3.06896151, 1.33705536,
                                  2.88376939, 5.7731461, 6.25456272, 3.61897724, 2.84128485,
                                  1.5326951, -0.27042984, -0.47728763, 4.76656958, 5.5262324,
                                  0.23418537, 1.16471453, 6.1996765, 5.0103605, 2.84942165,
                                  4.70577669, 2.95602827, 0.91514282, 1.21639131, -0.704394,
                                  1.57409695, 4.15243012, 5.35823905, 1.3051255, 5.75806083,
                                  2.82719571, 3.72493766, 3.32650945, 1.55292739, 3.56162231,
                                  3.44969159, 0.15020885, 1.80370832, 4.43277024, 2.57750473,
                                  3.94104616, 1.57925818, 1.81987033, 3.86571303, 4.7321192,
                                  2.80772861, 3.90200639, 5.27318404, 4.42546234, 3.12315514,
                                  4.69068983, 0.74265357, 5.50616718, 0.74802912, 0.6893649,
                                  1.72116781, 0.24663807, 2.01467995, -0.64710744, 4.09862906,
                                  4.95740592, -1.32573949, 0.9852149, 4.09104705, 1.74202664]]);
    let y = Vector::new(vec![vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]]);
    (x, y)
}
