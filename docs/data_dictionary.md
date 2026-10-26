# Data dictionary

All raw data come from Deribit's public API, with no account needed. Under Deribit's terms
they are for **personal use only** and are git-ignored. Rebuild everything with
`scripts/01_download_data.py` followed by `scripts/02_build_dataset.py`.

## Raw files (`data/raw/deribit/`, never modified)

| File(s) | Source endpoint | Content |
|---|---|---|
| `delivery_prices/btc_usd_page_NNN.json` | `www.deribit.com/api/v2/public/get_delivery_prices?index_name=btc_usd` | Daily 08:00 UTC settlement value of the BTC/USD index, newest first, 100 rows per page |
| `option_trades/BTC_YYYY-MM-DD_08-09UTC.json` | `history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time?currency=BTC&kind=option` | Every BTC option trade in that one-hour window (all API pages stored together) |
| `manifest_*.json` | written by `data_download.py` | URL, query parameters, retrieval time (UTC), SHA-256, bytes and record count for every raw file |

## `data/processed/index_daily.csv`

| Column | Unit | Meaning |
|---|---|---|
| `date` | UTC timestamp | 08:00 UTC fixing time |
| `index_price` | USD per BTC | Deribit btc_usd delivery price |
| `log_return` | – | $\ln(S_t/S_{t-1})$ if the previous fixing is exactly one day earlier, else empty. Weekly history before 2020-02-04 gives no daily returns |

## `data/processed/option_trades_clean.csv.gz`

| Column | Unit | Meaning |
|---|---|---|
| `trade_id` | – | Deribit trade id (unique after de-duplication) |
| `trade_time` | UTC | Trade timestamp (ms precision in the source) |
| `market_date` | UTC date | Calendar day of the trade, used as the bootstrap cluster |
| `instrument_name` | – | e.g. `BTC-31MAR23-24000-C` |
| `option_type` | call / put | From the last letter of the name |
| `strike` | USD | From the name |
| `expiry` | UTC | Expiry date at 08:00 UTC |
| `time_to_expiry` | years (365-day) | (expiry − trade_time) / 365 days |
| `price` | BTC per 1 BTC notional | **Observed trade price** |
| `price_usd` | USD | `price × index_price` (Deribit premiums are paid in BTC) |
| `mark_price` | BTC | Deribit's own model "mark" value at the trade. **Not** an observed price and not used as a target |
| `index_price` | USD | Index level at the trade |
| `reported_iv` | annual decimal | Deribit's reported IV for the trade (source field is in percent) |
| `forward` | USD | Forward recovered from (price_usd, reported_iv) by Black-76 |
| `annual_basis` | per year | $\ln(F/S)/\tau$ |
| `log_moneyness` | – | $\ln(K/F)$ |
| `direction` | buy / sell | Aggressor side of the trade |
| `amount` | BTC | Trade size |
| `split` | – | development / validation / test (see `docs/evaluation_protocol.md`) |

## Cleaning rules (applied in this order, each one counted in `exclusions.csv`)

1. duplicate `trade_id` (page overlap at the same millisecond);
2. unparseable instrument name;
3. missing or non-positive price, index price or IV;
4. liquidation trades, block-trade legs and combo legs. Their prices are negotiated or
   allocated across legs, so they are not ordinary screen trades;
5. days to expiry outside [3, 180];
6. price < 0.0005 BTC (5 ticks): at that level relative errors are meaningless;
7. reported IV outside [5%, 300%];
8. forward recovery failed, or $|\ln(F/S)/\tau|>50\%$ per year;
9. $|\ln(K/F)|>0.5$.

Then, in `05_track_b_market_pricing.py`:

10. no index fixing within one day before the trade, or not enough history for the
    longest (180-day) window;
11. development-period trades are not evaluated.

## Conventions

- Timestamps: UTC. Expiry: 08:00 UTC.
- Time: 365-day years (BTC trades every calendar day). Volatility is annualised with $\sqrt{365}$.
- Interest rate: 0 (Deribit convention); the carry sits in the forward.
- Price errors: model − trade. Primary unit: % of the index price at the trade.
- Hedge errors: hedge value at expiry − payoff. Unit: % of the spot price when the option is sold.
