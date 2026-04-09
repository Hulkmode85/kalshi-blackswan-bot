"""
Kalshi Black Swan Weather Bot
Systematically buys underpriced tail-event weather contracts on Kalshi.

Strategy:
  "The market systematically underprices tail events." — Nassim Taleb

  Prediction market participants anchor on recent history and consensus forecasts,
  leaving extreme outcomes (cat 5 hurricanes, record temps, 100-year floods) priced
  at less than their true actuarial probability.

  This bot:
  1. Identifies Kalshi markets for extreme weather outcomes
  2. Gets GFS/ECMWF ensemble probabilities for those outcomes
  3. Buys YES when:
     - Ensemble probability > RATIO_MIN × Kalshi implied probability
     - Kalshi price is < MAX_PRICE (genuine tail, not already priced in)
     - Remaining time to market close > MIN_HOURS_REMAINING

  Expected edge: ~300-1500% payout on hits, 2-10% hit rate
  Similar to buying out-of-the-money options — small bets, massive upside

Data Sources:
  - GFS Ensemble: Open-Meteo (https://ensemble-api.open-meteo.com)
  - ECMWF ERA5 climate normals for threshold calibration
  - NWS Storm Prediction Center for severe weather outlooks
  - NOAA HURDAT2 hurricane historical data

Market Categories:
  - KXHURRICANE: Hurricane landfalls, intensities, track markets
  - KXHIGHNY, KXHIGHMIA, KXHIGHATL, KXHIGHCHI: Record high temps
  - KXLOWNY, KXLOWCHI: Record low temps
  - KXSNOWNYC: NYC snowfall extremes
  - KXRAINLA: LA extreme precipitation
"""

import os
import time
from flask import Flask, jsonify
import threading
import json
import logging
import math
import uuid
import base64
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
import httpx
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from risk_guard import RiskManager

load_dotenv()

# ── Shadow Logging ────────────────────────────────────────────────────────────
SHADOW_LOG_FILE = os.getenv("SHADOW_LOG_FILE", "shadow_log.jsonl")

def shadow_log(opportunity: dict, taken: bool, reason: str = ""):
    entry = {"ts": time.time(), "taken": taken, "reason": reason, **opportunity}
    try:
        with open(SHADOW_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass

# ── Multi-strike: scan ALL strikes per event/series, not just one ────────────
MULTI_STRIKE = os.getenv("MULTI_STRIKE", "true").lower() == "true"
# When fetching markets, iterate through ALL contracts in each series/event
# and evaluate each strike independently. No single-ticker filtering.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

class Config:
    PAPER_MODE:             bool  = os.getenv("PAPER_MODE", "true").lower() == "true"
    PAPER_BALANCE:          float = float(os.getenv("PAPER_BALANCE", "5000"))
    KALSHI_API_KEY:         str   = os.getenv("KALSHI_API_KEY", "")
    KALSHI_KEY_ID:          str   = os.getenv("KALSHI_KEY_ID", "")

    # Black swan criteria
    RATIO_MIN:              float = float(os.getenv("RATIO_MIN", "2.5"))      # ensemble_prob / kalshi_prob
    MAKER_FEE:              float = float(os.getenv("MAKER_FEE", "0.0175"))
    MAX_PRICE:              int   = int(os.getenv("MAX_PRICE", "20"))         # only buy contracts ≤20¢
    MIN_ENSEMBLE_PROB:      float = float(os.getenv("MIN_ENSEMBLE_PROB", "0.03"))  # min 3% model probability
    MIN_HOURS_REMAINING:    int   = int(os.getenv("MIN_HOURS_REMAINING", "12"))

    BET_SIZE_USD:           float = float(os.getenv("BET_SIZE_USD", "5.0"))   # small bets on tails
    MAX_BET_USD:            float = float(os.getenv("MAX_BET_USD", "25.0"))
    KELLY_FRACTION:         float = float(os.getenv("KELLY_FRACTION", "1.0"))
    MAX_OPEN_POSITIONS:     int   = int(os.getenv("MAX_OPEN_POSITIONS", "15")) # diversified tails
    POLL_INTERVAL_SEC:      int   = int(os.getenv("POLL_INTERVAL_SEC", "1800")) # 30 min

    OPEN_METEO_BASE:        str   = "https://ensemble-api.open-meteo.com/v1/ensemble"
    KALSHI_BASE:            str   = "https://api.elections.kalshi.com/trade-api/v2"

# ── City / Station Config ─────────────────────────────────────────────────────

CITIES = {
    "NYC":   {"lat": 40.71, "lon": -74.01, "series": ["KXHIGHNY", "KXLOWNY", "KXSNOWNYC"]},
    "Miami": {"lat": 25.77, "lon": -80.19, "series": ["KXHIGHMIA"]},
    "Atlanta": {"lat": 33.75, "lon": -84.39, "series": ["KXHIGHATL"]},
    "Chicago": {"lat": 41.88, "lon": -87.63, "series": ["KXHIGHCHI", "KXLOWCHI"]},
    "LA":    {"lat": 34.05, "lon": -118.24, "series": ["KXRAINLA", "KXHIGHLA"]},
    "Houston": {"lat": 29.76, "lon": -95.37, "series": ["KXHIGHHOU"]},
    "Phoenix": {"lat": 33.45, "lon": -112.07, "series": ["KXHIGHPHX"]},
    "Boston": {"lat": 42.36, "lon": -71.06, "series": ["KXHIGHBOS", "KXSNOWBOS"]},
}

# Hurricane watch areas (Gulf + Atlantic + Pacific)
HURRICANE_SERIES = [
    "KXHURRICANE", "KXHURCAT", "KXHURATLANTIC", "KXHURGULF",
    "KXTROPICAL", "KXTROPSTORM",
]

# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class WeatherForecast:
    city: str
    lat: float
    lon: float
    forecast_time: datetime
    max_temp_c: float
    min_temp_c: float
    precip_mm: float
    wind_speed_kph: float
    # Ensemble spread (90th percentile extremes)
    max_temp_p90: float = 0.0
    min_temp_p10: float = 0.0
    precip_p90: float = 0.0

@dataclass
class TailOpportunity:
    city: str
    series_ticker: str
    market_ticker: str
    market_title: str
    side: str               # "YES" or "NO"
    kalshi_price: int       # in cents
    ensemble_prob: float    # 0-1
    kalshi_prob: float      # 0-1
    ratio: float            # ensemble_prob / kalshi_prob
    threshold_description: str
    contracts: int
    bet_usd: float
    close_time: datetime

# ── Kalshi Client ─────────────────────────────────────────────────────────────

@dataclass
class KalshiMarket:
    ticker: str
    title: str
    yes_price: int
    no_price: int
    volume: int
    close_time: datetime
    subtitle: str = ""

class KalshiClient:
    def __init__(self):
        self._client = httpx.Client(timeout=20)
        self._private_key = self._load_private_key()

    @staticmethod
    def _load_private_key():
        pem_str = os.getenv("KALSHI_PRIVATE_KEY", "")
        if not pem_str:
            return None
        if "\\n" in pem_str:
            pem_str = pem_str.replace("\\n", "\n")
        return serialization.load_pem_private_key(pem_str.encode(), password=None)

    def _get_auth_headers(self, method: str, path: str) -> dict:
        if not self._private_key:
            return {"Content-Type": "application/json"}
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + "/trade-api/v2" + path).encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )
        return {
            "Kalshi-Access-Key": os.getenv("KALSHI_API_KEY", ""),
            "Kalshi-Access-Signature": base64.b64encode(sig).decode(),
            "Kalshi-Access-Timestamp": ts,
            "Content-Type": "application/json",
        }

    def get_markets_for_series(self, series_ticker: str) -> list[KalshiMarket]:
        try:
            r = self._client.get(
                f"{Config.KALSHI_BASE}/markets",
                params={"series_ticker": series_ticker, "status": "open"},
                headers=self._get_auth_headers("GET", "/markets"),
            )
            r.raise_for_status()
            markets = []
            for m in r.json().get("markets", []):
                close_str = m.get("close_time", "")
                try:
                    close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                except Exception:
                    close_dt = datetime.now(timezone.utc) + timedelta(hours=24)
                markets.append(KalshiMarket(
                    ticker=m.get("ticker", ""),
                    title=m.get("title", ""),
                    subtitle=m.get("subtitle", ""),
                    yes_price=m.get("yes_ask", 0),
                    no_price=m.get("no_ask", 0),
                    volume=m.get("volume", 0),
                    close_time=close_dt,
                ))
            return markets
        except Exception as e:
            log.warning(f"get_markets_for_series({series_ticker}): {e}")
            return []

    def place_order(self, ticker: str, side: str, count: int, price: int) -> bool:
        if Config.PAPER_MODE:
            return True
        try:
            r = self._client.post(
                f"{Config.KALSHI_BASE}/portfolio/orders",
                json={"ticker": ticker, "client_order_id": str(uuid.uuid4()),
                      "action": "buy", "side": side.lower(),
                      "count": count, "type": "limit",
                      "yes_price": price if side == "YES" else 100 - price},
                headers=self._get_auth_headers("POST", "/portfolio/orders"),
            )
            r.raise_for_status()
            return True
        except Exception as e:
            log.error(f"place_order failed: {e}")
            return False


# ── Weather Forecast Engine ───────────────────────────────────────────────────

class WeatherEngine:
    def __init__(self):
        self._client = httpx.Client(timeout=30)

    def get_ensemble_forecast(self, city: str, lat: float, lon: float) -> Optional[WeatherForecast]:
        """Fetch GFS ensemble forecast for a city and compute tail probabilities."""
        try:
            r = self._client.get(
                Config.OPEN_METEO_BASE,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "models": "gfs_ensemble_025",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
                    "hourly": "temperature_2m,precipitation",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "precipitation_unit": "mm",
                    "forecast_days": 7,
                    "timezone": "UTC",
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()

            # Get daily ensemble members (Open-Meteo returns multiple members)
            daily = data.get("daily", {})
            temps_max = daily.get("temperature_2m_max", [])
            temps_min = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])

            if not temps_max:
                return None

            # Use tomorrow's forecast (index 1)
            idx = min(1, len(temps_max) - 1)

            # Flatten ensemble members if present (Open-Meteo may return lists of lists)
            def get_ensemble_vals(vals, idx):
                if vals and isinstance(vals[0], list):
                    return [row[idx] for row in vals if row and idx < len(row) and row[idx] is not None]
                v = vals[idx] if idx < len(vals) else None
                return [v] if v is not None else []

            temp_vals = get_ensemble_vals(temps_max, idx)
            min_vals  = get_ensemble_vals(temps_min, idx)
            prec_vals = get_ensemble_vals(precip, idx)

            if not temp_vals:
                return None

            temp_vals_sorted = sorted(temp_vals)
            n = len(temp_vals_sorted)

            return WeatherForecast(
                city=city, lat=lat, lon=lon,
                forecast_time=datetime.now(timezone.utc) + timedelta(days=1),
                max_temp_c=sum(temp_vals) / n,
                min_temp_c=sum(min_vals) / len(min_vals) if min_vals else 0,
                precip_mm=sum(prec_vals) / len(prec_vals) if prec_vals else 0,
                wind_speed_kph=0.0,
                max_temp_p90=temp_vals_sorted[int(n * 0.90)],
                min_temp_p10=sorted(min_vals)[int(len(min_vals) * 0.10)] if min_vals else 0,
                precip_p90=sorted(prec_vals)[int(len(prec_vals) * 0.90)] if prec_vals else 0,
            )
        except Exception as e:
            log.warning(f"Ensemble forecast failed for {city}: {e}")
            return None

    def prob_above_threshold(self, ensemble_values: list[float], threshold: float) -> float:
        """Fraction of ensemble members exceeding threshold."""
        if not ensemble_values:
            return 0.0
        return sum(1 for v in ensemble_values if v >= threshold) / len(ensemble_values)

    def get_raw_ensemble_temps(self, lat: float, lon: float) -> list[float]:
        """Get raw ensemble temperature values for tomorrow (all members)."""
        try:
            r = self._client.get(
                Config.OPEN_METEO_BASE,
                params={
                    "latitude": lat, "longitude": lon,
                    "models": "gfs_ensemble_025",
                    "daily": "temperature_2m_max",
                    "temperature_unit": "fahrenheit",
                    "forecast_days": 3, "timezone": "UTC",
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            temps = data.get("daily", {}).get("temperature_2m_max", [])
            if temps and isinstance(temps[0], list):
                # Multiple members: temps[member][day]
                return [row[1] for row in temps if len(row) > 1 and row[1] is not None]
            elif len(temps) > 1:
                return [temps[1]] if temps[1] is not None else []
            return []
        except Exception as e:
            log.warning(f"Raw ensemble temps failed: {e}")
            return []


# ── Opportunity Finder ────────────────────────────────────────────────────────

def parse_temp_threshold(title: str, subtitle: str) -> Optional[float]:
    """Extract temperature threshold from Kalshi market title (in °F)."""
    import re
    text = title + " " + subtitle
    # Match patterns like "above 95°F", "exceed 100", "reach 90", "at least 85°F"
    patterns = [
        r"(?:above|exceed|reach|at least|over)\s+(\d+)\s*°?[Ff]?",
        r"(\d+)\s*°F",
        r"(\d{2,3})\s*degrees",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_precip_threshold(title: str, subtitle: str) -> Optional[float]:
    """Extract precipitation threshold in inches."""
    import re
    text = title + " " + subtitle
    patterns = [
        r"(\d+\.?\d*)\s*inch(?:es)?",
        r"(?:above|exceed|reach|over)\s+(\d+\.?\d*)\s*\"",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def find_opportunities(
    city: str, city_cfg: dict,
    engine: WeatherEngine,
    kalshi: KalshiClient,
    existing_positions: set[str],
) -> list[TailOpportunity]:

    lat, lon = city_cfg["lat"], city_cfg["lon"]
    now = datetime.now(timezone.utc)
    opps = []

    # Get raw Fahrenheit ensemble temps for threshold comparison
    raw_temps_f = engine.get_raw_ensemble_temps(lat, lon)
    if not raw_temps_f:
        log.warning(f"[{city}] No ensemble data")
        return []

    n_members = len(raw_temps_f)
    log.info(f"[{city}] {n_members} ensemble members, "
             f"range {min(raw_temps_f):.0f}°F–{max(raw_temps_f):.0f}°F")

    for series_ticker in city_cfg["series"]:
        markets = kalshi.get_markets_for_series(series_ticker)
        if not markets:
            continue

        for market in markets:
            if market.ticker in existing_positions:
                continue

            # Check time remaining
            hours_remaining = (market.close_time - now).total_seconds() / 3600
            if hours_remaining < Config.MIN_HOURS_REMAINING:
                continue

            # Temperature-based markets
            if "HIGH" in series_ticker or "LOW" in series_ticker or "TEMP" in series_ticker:
                threshold = parse_temp_threshold(market.title, market.subtitle)
                if threshold is None:
                    continue

                if "HIGH" in series_ticker:
                    ensemble_prob = engine.prob_above_threshold(raw_temps_f, threshold)
                    kalshi_prob = market.yes_price / 100
                    side = "YES"
                else:  # LOW
                    ensemble_prob = engine.prob_above_threshold(
                        [-v for v in raw_temps_f], -threshold)
                    kalshi_prob = market.yes_price / 100
                    side = "YES"

                if ensemble_prob < Config.MIN_ENSEMBLE_PROB:
                    continue
                if kalshi_prob == 0:
                    continue

                ratio = ensemble_prob / kalshi_prob
                kalshi_price = market.yes_price if side == "YES" else market.no_price

                log.info(f"[{city}/{series_ticker}] {market.ticker} "
                         f"threshold={threshold}°F ensemble={ensemble_prob:.1%} "
                         f"kalshi={kalshi_prob:.1%} ratio={ratio:.1f}x price={kalshi_price}¢")

                edge = ensemble_prob - kalshi_prob
                ev_after_fees = edge - Config.MAKER_FEE
                if ev_after_fees <= 0:
                    shadow_log({"bot": "blackswan", "ticker": market.ticker, "city": city, "edge": edge, "ratio": ratio, "price": kalshi_price}, taken=False, reason=f"negative EV after fees ({ev_after_fees:.3f})")
                    continue
                if ratio >= Config.RATIO_MIN and kalshi_price <= Config.MAX_PRICE:
                    # Kelly criterion: f* = (model_prob - market_prob) / (1 - market_prob)
                    market_prob = kalshi_price / 100
                    kelly_f = max(0, (ensemble_prob - market_prob) / (1 - market_prob)) if market_prob < 1 else 0
                    kelly_bet = max(1, min(self.ledger.balance * kelly_f * Config.KELLY_FRACTION, Config.MAX_BET_USD))
                    contracts = max(1, int(kelly_bet * 100 / kalshi_price))
                    opps.append(TailOpportunity(
                        city=city, series_ticker=series_ticker,
                        market_ticker=market.ticker,
                        market_title=market.title[:80],
                        side=side, kalshi_price=kalshi_price,
                        ensemble_prob=ensemble_prob, kalshi_prob=kalshi_prob,
                        ratio=ratio,
                        threshold_description=f"{threshold}°F high",
                        contracts=contracts, bet_usd=contracts * kalshi_price / 100,
                        close_time=market.close_time,
                    ))

    return opps


# ── Paper Ledger ──────────────────────────────────────────────────────────────

class PaperLedger:
    def __init__(self):
        self.balance = Config.PAPER_BALANCE
        self.trades: list[dict] = []
        self.open_positions: dict[str, dict] = {}

    def open_position(self, opp: TailOpportunity) -> bool:
        if len(self.open_positions) >= Config.MAX_OPEN_POSITIONS:
            log.info(f"[PAPER] Max positions reached, skipping {opp.market_ticker}")
            return False
        cost = opp.kalshi_price * opp.contracts / 100
        if cost > self.balance:
            log.info(f"[PAPER] Insufficient balance for {opp.market_ticker}")
            return False
        self.balance -= cost
        rec = {
            "ticker": opp.market_ticker, "side": opp.side,
            "price": opp.kalshi_price, "contracts": opp.contracts, "cost": cost,
            "city": opp.city, "threshold": opp.threshold_description,
            "ensemble_prob": opp.ensemble_prob, "kalshi_prob": opp.kalshi_prob,
            "ratio": opp.ratio, "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.open_positions[opp.market_ticker] = rec
        self.trades.append({"action": "OPEN", **rec})
        log.info(
            f"[PAPER] OPEN {opp.side} {opp.market_ticker} @ {opp.kalshi_price}¢ × {opp.contracts} = ${cost:.2f} | "
            f"{opp.city} {opp.threshold_description} | ensemble={opp.ensemble_prob:.1%} "
            f"kalshi={opp.kalshi_prob:.1%} ratio={opp.ratio:.1f}x | balance=${self.balance:.2f}"
        )
        return True

    def close_position(self, ticker: str, exit_price: int, reason: str = ""):
        pos = self.open_positions.pop(ticker, None)
        if not pos:
            return
        pnl = (exit_price - pos["price"]) * pos["contracts"] / 100
        if pos["side"] == "NO":
            pnl = (pos["price"] - exit_price) * pos["contracts"] / 100
        self.balance += pos["cost"] + pnl
        self.trades.append({"action": "CLOSE", "ticker": ticker,
                             "exit_price": exit_price, "pnl": pnl, "reason": reason})
        log.info(f"[PAPER] CLOSE {ticker} @ {exit_price}¢ | PnL=${pnl:+.2f} | "
                 f"reason={reason} | balance=${self.balance:.2f}")


# ── Main Loop ─────────────────────────────────────────────────────────────────

# ── Stats HTTP server ─────────────────────────────────────────────────────────
_stats_app = Flask(__name__)
_bot_stats = {"trades": 0, "wins": 0, "pnl": 0.0, "balance": 0.0, "start": time.time()}

@_stats_app.route("/stats")
def _stats_endpoint():
    t = _bot_stats
    total = t["trades"]
    return jsonify({"bot": "kalshi-blackswan-bot", "paper_mode": True,
        "balance": t["balance"], "trades": total, "wins": t["wins"],
        "losses": total - t["wins"], "win_rate": round(t["wins"]/max(total,1), 4),
        "pnl": t["pnl"], "uptime_hours": round((time.time()-t["start"])/3600, 2)})

@_stats_app.route("/health")
def _health_endpoint():
    return jsonify({"status": "ok"})

def _run_stats_server():
    _stats_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


def main():
    log.info("=" * 60)
    log.info("Kalshi Black Swan Weather Bot starting")
    log.info(f"  Paper mode:     {Config.PAPER_MODE}")
    log.info(f"  Cities:         {list(CITIES.keys())}")
    log.info(f"  Min ratio:      {Config.RATIO_MIN}x (ensemble/market)")
    log.info(f"  Max price:      {Config.MAX_PRICE}¢ (tail contracts only)")
    log.info(f"  Min prob:       {Config.MIN_ENSEMBLE_PROB:.0%} ensemble")
    log.info(f"  Bet size:       ${Config.BET_SIZE_USD}")
    log.info(f"  Poll interval:  {Config.POLL_INTERVAL_SEC}s")
    log.info("=" * 60)

    engine = WeatherEngine()
    kalshi = KalshiClient()
    ledger = PaperLedger()
    risk_manager = RiskManager(starting_balance=Config.PAPER_BALANCE)
    _bot_stats['balance'] = ledger.balance
    threading.Thread(target=_run_stats_server, daemon=True).start()

    cycle = 0
    while True:
        cycle += 1
        log.info(f"── Cycle {cycle} ──────────────────────────")

        existing = set(ledger.open_positions.keys())
        all_opps: list[TailOpportunity] = []

        for city, cfg in CITIES.items():
            try:
                opps = find_opportunities(city, cfg, engine, kalshi, existing)
                all_opps.extend(opps)
            except Exception as e:
                log.error(f"[{city}] Error: {e}", exc_info=True)

        # Sort by ratio descending — highest edge first
        all_opps.sort(key=lambda o: o.ratio, reverse=True)

        if all_opps:
            log.info(f"[SCAN] Found {len(all_opps)} tail opportunities")
        else:
            log.info(f"[SCAN] No tail opportunities found this cycle")

        for opp in all_opps:
            if opp.market_ticker in ledger.open_positions:
                continue

            # Risk guard check
            if not Config.PAPER_MODE:
                allowed, reason, capped = risk_manager.pre_trade_check(
                    opp.city, opp.kalshi_price, opp.contracts, opp.side.lower(),
                    bot_name="blackswan-bot")
                if not allowed:
                    log.warning(f"Risk guard blocked: {reason}")
                    continue
                opp.contracts = capped or opp.contracts
            else:
                allowed, reason, capped = risk_manager.pre_trade_check(
                    opp.city, opp.kalshi_price, opp.contracts, opp.side.lower(),
                    bot_name="blackswan-bot")
                if not allowed:
                    log.info(f"[PAPER] Risk guard would block: {reason}")

            if Config.PAPER_MODE:
                ledger.open_position(opp)
                shadow_log({"bot": "blackswan", "ticker": opp.market_ticker, "city": opp.city, "side": opp.side, "price": opp.kalshi_price, "ratio": opp.ratio, "edge": opp.ensemble_prob - opp.kalshi_prob}, taken=True)
            else:
                if kalshi.place_order(opp.market_ticker, opp.side, opp.contracts, opp.kalshi_price):
                    shadow_log({"bot": "blackswan", "ticker": opp.market_ticker, "city": opp.city, "side": opp.side, "price": opp.kalshi_price, "ratio": opp.ratio, "edge": opp.ensemble_prob - opp.kalshi_prob}, taken=True)
                    log.info(f"[LIVE] {opp.side} {opp.market_ticker} @ {opp.kalshi_price}¢ × {opp.contracts}")

        # Summary
        open_count = len(ledger.open_positions)
        _bot_stats['balance'] = ledger.balance
        _bot_stats['trades'] = sum(1 for t in ledger.trades if t['action'] == 'OPEN')
        _bot_stats['wins'] = sum(1 for t in ledger.trades if t['action'] == 'CLOSE' and t.get('pnl', 0) > 0)
        _bot_stats['pnl'] = sum(t.get('pnl', 0) for t in ledger.trades if t['action'] == 'CLOSE')
        closed_pnl = sum(t.get("pnl", 0) for t in ledger.trades if t["action"] == "CLOSE")
        total_opened = sum(1 for t in ledger.trades if t["action"] == "OPEN")
        log.info(f"[SUMMARY] Balance=${ledger.balance:.2f} | Open={open_count} | "
                 f"Trades={total_opened} | Closed PnL=${closed_pnl:+.2f}")

        time.sleep(Config.POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
