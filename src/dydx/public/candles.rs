use reqwest;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candle {
    started_at: String,
    updated_at: String,
    market: String,
    resolution: String,
    low: String,
    high: String,
    open: String,
    close: String,
    base_token_volume: String,
    trades: String,
    usd_volume: String,
    starting_open_interest: String
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Candles {
    candles: Vec<Candle>
}

impl Candles {
    pub fn new(symbol: String, resolution: String, to_iso: String) -> Candles {

        let url = "https://api.dydx.exchange/v3/candles/".to_owned() + &symbol + "?resolution=" + &resolution + "&to_iso=" + &to_iso;
        let response = reqwest::blocking::get(url).unwrap();
        let text = response.text().unwrap();

        serde_json::from_str(&*text).unwrap()
    }
}
