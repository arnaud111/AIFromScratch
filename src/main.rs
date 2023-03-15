use crate::deep_neural_network::network::Network;
use crate::deep_neural_network::neuron::Neuron;
use crate::deep_neural_network::layer::Layer;
use crate::math::vector::{create_random_vector, generate_number};

mod deep_neural_network;
mod math;



fn main() {

    let (x, y) = create_data();

    let mut network = Network::new(vec![3, 4, 3], &x[0]);
    println!("Probability: {}", network.get_probability(x[0].to_vec()));

}

fn create_data() -> (Vec<&'static [f64]>, Vec<bool>) {
    let x: Vec<&[f64]> = vec![&[ 4.21850347,  2.23419161], &[ 0.90779887,  0.45984362],
                 &[-0.27652528,  5.08127768], &[ 0.08848433,  2.32299086], &[ 3.24329731,  1.21460627],
                 &[ 1.44193252,  2.76754364], &[ 1.0220286 ,  4.11660348], &[ 3.97820955,  2.37817845],
                 &[ 0.58894326,  4.00148458], &[ 1.25185786,  0.20811388], &[ 0.62835793,  4.4601363 ],
                 &[ 1.68608568,  0.65828448], &[ 1.18454506,  5.28042636], &[ 0.06897171,  4.35573272],
                 &[ 1.78726415,  1.70012006], &[ 4.4384123 ,  1.84214315], &[ 3.18190344, -0.18226785],
                 &[ 0.30380963,  3.94423417], &[ 0.73936011,  0.43607906], &[ 1.28535145,  1.43691285],
                 &[ 1.1312175 ,  4.68194985], &[ 0.66471755,  4.35995267], &[ 1.31570453,  2.44067826],
                 &[-0.18887976,  5.20461381], &[ 2.57854418,  0.72611733], &[ 0.87305123,  4.71438583],
                 &[ 1.3105127 ,  0.07122512], &[ 0.9867701 ,  6.08965782], &[ 1.42013331,  4.63746165],
                 &[ 2.3535057 ,  2.22404956], &[ 2.43169305, -0.20173713], &[ 1.0427873 ,  4.60625923],
                 &[ 0.95088418,  0.94982874], &[ 2.45127423, -0.19539785], &[ 1.62011397,  2.74692739],
                 &[ 2.15504965,  4.12386249], &[ 1.38093486,  0.92949422], &[ 1.98702592,  2.61100638],
                 &[ 2.11567076,  3.06896151], &[ 0.56400993,  1.33705536], &[-0.07228289,  2.88376939],
                 &[ 2.50904929,  5.7731461 ], &[-0.73000011,  6.25456272], &[ 1.37861172,  3.61897724],
                 &[ 0.88214412,  2.84128485], &[ 2.22194102,  1.5326951 ], &[ 2.0159847 , -0.27042984],
                 &[ 1.70127361, -0.47728763], &[-0.65392827,  4.76656958], &[ 0.57309313,  5.5262324 ],
                 &[ 1.956815  ,  0.23418537], &[ 0.76241061,  1.16471453], &[ 2.46452227,  6.1996765 ],
                 &[ 1.33263648,  5.0103605 ], &[ 3.2460247 ,  2.84942165], &[ 1.10318217,  4.70577669],
                 &[ 2.85942078,  2.95602827], &[ 1.59973502,  0.91514282], &[ 2.97612635,  1.21639131],
                 &[ 2.68049897, -0.704394  ], &[ 1.41942144,  1.57409695], &[ 1.9263585 ,  4.15243012],
                 &[-0.09448254,  5.35823905], &[ 2.72756228,  1.3051255 ], &[ 1.12031365,  5.75806083],
                 &[ 1.55723507,  2.82719571], &[ 0.10547293,  3.72493766], &[ 2.84382807,  3.32650945],
                 &[ 3.15492712,  1.55292739], &[ 1.84070628,  3.56162231], &[ 1.28933778,  3.44969159],
                 &[ 1.64164854,  0.15020885], &[ 3.92282648,  1.80370832], &[ 1.70536064,  4.43277024],
                 &[ 0.1631238 ,  2.57750473], &[ 0.34194798,  3.94104616], &[ 1.02102468,  1.57925818],
                 &[ 2.66934689,  1.81987033], &[ 0.4666179 ,  3.86571303], &[ 0.94808785,  4.7321192 ],
                 &[ 1.19404184,  2.80772861], &[ 1.15369622,  3.90200639], &[-0.29421492,  5.27318404],
                 &[ 1.7373078 ,  4.42546234], &[ 0.46546494,  3.12315514], &[ 0.08080352,  4.69068983],
                 &[ 3.00251949,  0.74265357], &[ 2.20656076,  5.50616718], &[ 1.36069966,  0.74802912],
                 &[ 2.63185834,  0.6893649 ], &[ 2.82705807,  1.72116781], &[ 2.91209813,  0.24663807],
                 &[ 1.1424453 ,  2.01467995], &[ 1.05505217, -0.64710744], &[ 2.47034915,  4.09862906],
                 &[-1.57671974,  4.95740592], &[ 1.41164912, -1.32573949], &[ 3.00468833,  0.9852149 ],
                 &[-0.63762777,  4.09104705], &[ 0.829832,    1.74202664]];
    let y = vec![true, true, false, false, true, false, false, true, false, true, false, true, false, false,
                 true, true, true, false, true, true, false, false, true, false, true, false, true, false, false, true,
                 true, false, true, true, true, false, true, true, false, true, false, false, false, false, true, true,
                 true, true, false, false, true, true, false, false, false, false, false, true, true, true, true, false,
                 false, true, false, true, false, false, true, false, false, true, true, false, false, false, true, true,
                 false, false, true, false, false, false, false, false, true, false, true, true, true, true, true, true,
                 false, false, true, true, false, true];
    (x, y)
}
