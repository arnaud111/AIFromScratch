use std::fs::File;
use std::io::{Read, Write};
use serde::{Deserialize, Serialize};
use serde_json;
use crate::dydx::public::candles::Candles;

#[derive(Serialize, Deserialize)]
struct Data {
    ema_20: f64,
    ema_50: f64,
    ema_100: f64
}

pub fn save_data(file_name: &str) {

    let resolution = "5MINS".to_string();
    let symbol = "ETH-USD".to_string();
    let mut start_time = "2023-03-17T03:25:00.000Z".to_string();
    let mut all_candles = Candles::new(&symbol, &resolution, &start_time);
    let mut candles = all_candles.clone();

    for i in 0..6000 {
        if candles.candles.len() == 0 {
            break;
        }
        start_time = candles.candles[candles.candles.len() - 1].started_at.clone();
        candles = Candles::new(&symbol, &resolution, &start_time);
        all_candles.concat(&candles);
        println!("{}", i);
    }

    let mut file = File::create(format!("dataset/request/{}.json", file_name)).unwrap();
    let serialized = serde_json::to_string(&all_candles).unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
}

pub fn load_data(file_name: &str) -> Candles {
    let mut file = File::open(format!("dataset/request/{}.json", file_name)).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let deserialized: Candles = serde_json::from_str(&contents).unwrap();
    deserialized
}

pub fn compute_data(file_name: &str) {

    let data = load_data(file_name);

    let mut dataset = Vec::new();

    let ema_20 = ema(data.clone(), 20);
    let ema_50 = ema(data.clone(), 50);
    let ema_100 = ema(data.clone(), 100);
    let ema_200 = ema(data.clone(), 200);

    for i in 0..data.candles.len() {
        dataset.push(Data {
            ema_20: ema_20[i] / ema_200[i],
            ema_50: ema_50[i] / ema_200[i],
            ema_100: ema_100[i] / ema_200[i]
        });
    }

    let mut file = File::create(format!("dataset/{}.json", file_name)).unwrap();
    let serialized = serde_json::to_string(&dataset).unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
}

pub fn ema(data: Candles, period: usize) -> Vec<f64> {
    let mut ema = Vec::new();
    let mut sum = 0.0;

    for i in 0..period {
        sum += data.candles[i].close.parse::<f64>().unwrap();
    }

    ema.push(sum / period as f64);

    for i in period..data.candles.len() {
        ema.push(data.candles[i].close.parse::<f64>().unwrap() * 2.0 / (period as f64 + 1.0) + ema[i - period] * (period as f64 - 1.0) / (period as f64 + 1.0));
    }

    ema
}