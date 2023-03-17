use crate::dydx::public::candles::Candles;

pub fn get_data() {

    let candles = Candles::new("ETH-USD".to_string(), "5MINS".to_string(), "2023-03-17T03:25:00.000Z".to_string());
    println!("{:?}", candles);
}