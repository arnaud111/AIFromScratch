use reqwest;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candle {
    pub started_at: String,
    pub updated_at: String,
    pub market: String,
    pub resolution: String,
    pub low: String,
    pub high: String,
    pub open: String,
    pub close: String,
    pub base_token_volume: String,
    pub trades: String,
    pub usd_volume: String,
    pub starting_open_interest: String
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candles {
    pub candles: Vec<Candle>
}

impl Candles {
    pub fn new(symbol: &String, resolution: &String, to_iso: &String) -> Candles {

        let url = "https://api.dydx.exchange/v3/candles/".to_owned() + &symbol + "?resolution=" + &resolution + "&toISO=" + &to_iso;
        let response = reqwest::blocking::get(url).unwrap();
        let text = response.text().unwrap();

        serde_json::from_str(&*text).unwrap()
    }

    pub fn concat(&mut self, other: &Candles) {
        self.candles.extend(other.candles.clone());
    }
}
