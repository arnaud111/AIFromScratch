use crate::deep_neural_network::network::Network;
use crate::math::vector::{create_random_vector, generate_number};

mod deep_neural_network;
mod math;



fn main() {

    let (x, y) = create_data();

    let mut network = Network::new();
    network.init_layers(vec![2, 4, 1], &x[0]);

}

fn create_data() -> (Vec<Vec<f64>>, Vec<bool>) {
    let x = vec![vec![ 4.21850347,  2.23419161], vec![ 0.90779887,  0.45984362],
                 vec![-0.27652528,  5.08127768], vec![ 0.08848433,  2.32299086], vec![ 3.24329731,  1.21460627],
                 vec![ 1.44193252,  2.76754364], vec![ 1.0220286 ,  4.11660348], vec![ 3.97820955,  2.37817845],
                 vec![ 0.58894326,  4.00148458], vec![ 1.25185786,  0.20811388], vec![ 0.62835793,  4.4601363 ],
                 vec![ 1.68608568,  0.65828448], vec![ 1.18454506,  5.28042636], vec![ 0.06897171,  4.35573272],
                 vec![ 1.78726415,  1.70012006], vec![ 4.4384123 ,  1.84214315], vec![ 3.18190344, -0.18226785],
                 vec![ 0.30380963,  3.94423417], vec![ 0.73936011,  0.43607906], vec![ 1.28535145,  1.43691285],
                 vec![ 1.1312175 ,  4.68194985], vec![ 0.66471755,  4.35995267], vec![ 1.31570453,  2.44067826],
                 vec![-0.18887976,  5.20461381], vec![ 2.57854418,  0.72611733], vec![ 0.87305123,  4.71438583],
                 vec![ 1.3105127 ,  0.07122512], vec![ 0.9867701 ,  6.08965782], vec![ 1.42013331,  4.63746165],
                 vec![ 2.3535057 ,  2.22404956], vec![ 2.43169305, -0.20173713], vec![ 1.0427873 ,  4.60625923],
                 vec![ 0.95088418,  0.94982874], vec![ 2.45127423, -0.19539785], vec![ 1.62011397,  2.74692739],
                 vec![ 2.15504965,  4.12386249], vec![ 1.38093486,  0.92949422], vec![ 1.98702592,  2.61100638],
                 vec![ 2.11567076,  3.06896151], vec![ 0.56400993,  1.33705536], vec![-0.07228289,  2.88376939],
                 vec![ 2.50904929,  5.7731461 ], vec![-0.73000011,  6.25456272], vec![ 1.37861172,  3.61897724],
                 vec![ 0.88214412,  2.84128485], vec![ 2.22194102,  1.5326951 ], vec![ 2.0159847 , -0.27042984],
                 vec![ 1.70127361, -0.47728763], vec![-0.65392827,  4.76656958], vec![ 0.57309313,  5.5262324 ],
                 vec![ 1.956815  ,  0.23418537], vec![ 0.76241061,  1.16471453], vec![ 2.46452227,  6.1996765 ],
                 vec![ 1.33263648,  5.0103605 ], vec![ 3.2460247 ,  2.84942165], vec![ 1.10318217,  4.70577669],
                 vec![ 2.85942078,  2.95602827], vec![ 1.59973502,  0.91514282], vec![ 2.97612635,  1.21639131],
                 vec![ 2.68049897, -0.704394  ], vec![ 1.41942144,  1.57409695], vec![ 1.9263585 ,  4.15243012],
                 vec![-0.09448254,  5.35823905], vec![ 2.72756228,  1.3051255 ], vec![ 1.12031365,  5.75806083],
                 vec![ 1.55723507,  2.82719571], vec![ 0.10547293,  3.72493766], vec![ 2.84382807,  3.32650945],
                 vec![ 3.15492712,  1.55292739], vec![ 1.84070628,  3.56162231], vec![ 1.28933778,  3.44969159],
                 vec![ 1.64164854,  0.15020885], vec![ 3.92282648,  1.80370832], vec![ 1.70536064,  4.43277024],
                 vec![ 0.1631238 ,  2.57750473], vec![ 0.34194798,  3.94104616], vec![ 1.02102468,  1.57925818],
                 vec![ 2.66934689,  1.81987033], vec![ 0.4666179 ,  3.86571303], vec![ 0.94808785,  4.7321192 ],
                 vec![ 1.19404184,  2.80772861], vec![ 1.15369622,  3.90200639], vec![-0.29421492,  5.27318404],
                 vec![ 1.7373078 ,  4.42546234], vec![ 0.46546494,  3.12315514], vec![ 0.08080352,  4.69068983],
                 vec![ 3.00251949,  0.74265357], vec![ 2.20656076,  5.50616718], vec![ 1.36069966,  0.74802912],
                 vec![ 2.63185834,  0.6893649 ], vec![ 2.82705807,  1.72116781], vec![ 2.91209813,  0.24663807],
                 vec![ 1.1424453 ,  2.01467995], vec![ 1.05505217, -0.64710744], vec![ 2.47034915,  4.09862906],
                 vec![-1.57671974,  4.95740592], vec![ 1.41164912, -1.32573949], vec![ 3.00468833,  0.9852149 ],
                 vec![-0.63762777,  4.09104705], vec![ 0.829832,    1.74202664]];
    let y = vec![true, true, false, false, true, false, false, true, false, true, false, true, false, false,
                 true, true, true, false, true, true, false, false, true, false, true, false, true, false, false, true,
                 true, false, true, true, true, false, true, true, false, true, false, false, false, false, true, true,
                 true, true, false, false, true, true, false, false, false, false, false, true, true, true, true, false,
                 false, true, false, true, false, false, true, false, false, true, true, false, false, false, true, true,
                 false, false, true, false, false, false, false, false, true, false, true, true, true, true, true, true,
                 false, false, true, true, false, true];
    (x, y)
}
