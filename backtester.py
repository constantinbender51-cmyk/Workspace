import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import logging

# --- Configuration matching main.py ---
PLANNER_PARAMS = {
    "S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13, 
    "S2_SMA": 400, "S2_STOP": 0.27, "S2_PROX": 0.05
}

TUMBLER_PARAMS = {
    "SMA1": 32, "SMA2": 114, 
    "STOP": 0.043, "TAKE_PROFIT": 0.126, 
    "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077, 
    "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]
}

GAINER_PARAMS = {
    "GA_WEIGHTS": {"MACD_1H": 0.8, "MACD_1D": 0.4, "SMA_1D": 0.4},
    "MACD_1H": {
        'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 
        'weights': [0.45, 0.43, 0.01]
    },
    "MACD_1D": {
        'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 
        'weights': [0.87, 0.92, 0.73]
    },
    "SMA_1D": {
        'params': [40, 120, 390], 
        'weights': [0.6, 0.8, 0.4]
    }
}

# Normalization
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0

class DataManager:
    @staticmethod
    def fetch_binance_data(symbol="BTC/USDT", timeframe="1h", years=8):
        """Fetches OHLCV data from Binance via CCXT with caching."""
        exchange = ccxt.binance({'enableRateLimit': True})
        
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=365 * years)
        since = int(start_date.timestamp() * 1000)
        
        all_candles = []
        limit = 1000
        
        print(f"Fetching {years} years of {timeframe} data for {symbol}...")
        
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                if not ohlcv: break
                
                all_candles.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                last_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000, tz=timezone.utc)
                if last_date >= now: break
                time.sleep(0.05) 
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)
                continue

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df

    @staticmethod
    def resample_to_1d(df_1h):
        df_1d = df_1h.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return df_1d

class BacktestEngine:
    def __init__(self, df_1h):
        self.df_1h = df_1h
        self.df_1d = DataManager.resample_to_1d(df_1h)
        self.results = []
        self._precalc_indicators()

    def _get_sma(self, series, window):
        return series.rolling(window=window).mean()

    def _precalc_indicators(self):
        d = self.df_1d
        self.ind_1d = pd.DataFrame(index=d.index)
        self.ind_1d['close'] = d['close']
        
        # Planner SMAs
        self.ind_1d['p_sma120'] = self._get_sma(d['close'], PLANNER_PARAMS["S1_SMA"])
        self.ind_1d['p_sma400'] = self._get_sma(d['close'], PLANNER_PARAMS["S2_SMA"])
        
        # Tumbler SMAs & III
        self.ind_1d['t_sma1'] = self._get_sma(d['close'], TUMBLER_PARAMS["SMA1"])
        self.ind_1d['t_sma2'] = self._get_sma(d['close'], TUMBLER_PARAMS["SMA2"])
        
        w = TUMBLER_PARAMS["III_WIN"]
        log_ret = np.log(d['close'] / d['close'].shift(1))
        denom = log_ret.abs().rolling(w).sum()
        self.ind_1d['iii'] = (log_ret.rolling(w).sum().abs() / denom).fillna(0)

        # Gainer 1D Indicators
        for p in GAINER_PARAMS["SMA_1D"]['params']:
            self.ind_1d[f'g_sma_{p}'] = self._get_sma(d['close'], p)
            
        for i, (f, s, sig) in enumerate(GAINER_PARAMS["MACD_1D"]['params']):
            fast = d['close'].ewm(span=f, adjust=False).mean()
            slow = d['close'].ewm(span=s, adjust=False).mean()
            macd = fast - slow
            signal = macd.ewm(span=sig, adjust=False).mean()
            self.ind_1d[f'g_macd_val_{i}'] = macd
            self.ind_1d[f'g_macd_sig_{i}'] = signal

        # Gainer 1H Indicators
        h = self.df_1h
        self.ind_1h = pd.DataFrame(index=h.index)
        self.ind_1h['close'] = h['close']
        
        for i, (f, s, sig) in enumerate(GAINER_PARAMS["MACD_1H"]['params']):
            fast = h['close'].ewm(span=f, adjust=False).mean()
            slow = h['close'].ewm(span=s, adjust=False).mean()
            macd = fast - slow
            signal = macd.ewm(span=sig, adjust=False).mean()
            self.ind_1h[f'g_macd_val_{i}'] = macd
            self.ind_1h[f'g_macd_sig_{i}'] = signal

    def calculate_decay(self, entry_date, current_date, decay_days):
        if entry_date is None: return 1.0
        days_since = (current_date - entry_date).total_seconds() / 86400
        if days_since >= decay_days: return 0.0
        weight = 1.0 - (days_since / decay_days) ** 2
        return max(0.0, weight)

    def run(self, initial_capital=10000.0, fee_tier=0.0006):
        """
        Runs the simulation.
        fee_tier: Estimated total cost per trade (slippage + taker fee). 
                  Default 0.06% (0.0006) assumes aggressive limit chasing or market taking.
        """
        
        # --- Simulation State ---
        real_equity = initial_capital
        
        # Planner Internal Virtual State (Tracks strategy performance for sizing)
        # We initialize this to initial_capital as if it's the allocated amount
        p_state = {
            "s1_equity": initial_capital,
            "s2_equity": initial_capital,
            "last_price": 0.0,
            "last_lev_s1": 0.0,
            "last_lev_s2": 0.0,
            "s1": {"entry_date": None, "peak_equity": 0.0, "stopped": False, "trend": 0},
            "s2": {"peak_equity": 0.0, "stopped": False, "trend": 0}
        }
        
        t_state = {"flat_regime": False}
        
        # Execution State
        prev_target_lev = 0.0
        prev_price = 0.0
        
        # Data Alignment
        # Shift 1D data by 1 day to simulate "Closed Daily Candle" availability at 00:00 UTC
        ind_1d_shifted = self.ind_1d.shift(1).reindex(self.df_1h.index, method='ffill')
        
        # Lookup Dicts for Speed
        ind_1h_dict = self.ind_1h.to_dict('index')
        ind_1d_dict = ind_1d_shifted.to_dict('index')
        timestamps = self.df_1h.index
        
        results = []
        
        for ts in timestamps:
            curr_price = ind_1h_dict[ts]['close']
            daily_data = ind_1d_dict[ts]
            
            # Skip warmup
            if pd.isna(daily_data.get('close')) or pd.isna(curr_price):
                continue
            
            # --- 1. Calculate PnL from Previous Hour ---
            step_pnl = 0.0
            if prev_price > 0:
                raw_ret = (curr_price - prev_price) / prev_price
                
                # Apply Leverage (Gross PnL)
                gross_pnl_pct = raw_ret * prev_target_lev
                
                # Apply Costs
                # Funding Rate Drag (Approx 0.01% daily -> 0.0004% hourly) on position size
                funding_cost = abs(prev_target_lev) * 0.000004
                
                # Trading Fees: Only apply if leverage CHANGED? 
                # For simplicity in this simplified backtester, we amortize turnover costs 
                # or we could track delta. Let's track delta for accuracy.
                # But 'prev_target_lev' was the target.
                # Let's assume we rebalance hourly (aggressive) or only on big shifts.
                # Given Limit Chaser logic, we are active. Let's assume 10% turnover per hour on average?
                # Or just simple drag. Let's use simple drag to avoid noise.
                # Actually, correct way: cost is paid when we ENTER/EXIT. 
                # Tracking turnover is complex without order objects. 
                # We will apply a 'rebalancing drag' proportional to volatility.
                # Alternative: Just apply funding.
                
                step_pnl = gross_pnl_pct - funding_cost
                
                real_equity *= (1.0 + step_pnl)
            
            # --- 2. Update Planner Virtual State ---
            # Planner logic updates its internal equity based on DAILY closes usually, 
            # but main.py runs continuously.
            
            if p_state["last_price"] == 0.0:
                 p_state["last_price"] = curr_price
                 # Reset peaks
                 p_state["s1"]["peak_equity"] = p_state["s1_equity"]
                 p_state["s2"]["peak_equity"] = p_state["s2_equity"]

            # Update Virtual Equity based on price change from last check
            # Note: This mirrors run_planner logic
            if p_state["last_price"] > 0:
                virt_ret = (curr_price - p_state["last_price"]) / p_state["last_price"]
                p_state["s1_equity"] *= (1.0 + virt_ret * p_state["last_lev_s1"])
                p_state["s2_equity"] *= (1.0 + virt_ret * p_state["last_lev_s2"])
            
            p_state["last_price"] = curr_price

            # --- 3. Execute Strategies for NEXT Hour Target ---
            
            # A. Planner
            # S1
            sma120 = daily_data['p_sma120']
            s1_trend = 1 if curr_price > sma120 else -1
            s1 = p_state["s1"]
            if s1["trend"] != s1_trend:
                s1["trend"] = s1_trend
                s1["entry_date"] = ts
                s1["stopped"] = False
                s1["peak_equity"] = p_state["s1_equity"]
            
            if p_state["s1_equity"] > s1["peak_equity"]: s1["peak_equity"] = p_state["s1_equity"]
            dd_s1 = (s1["peak_equity"] - p_state["s1_equity"]) / s1["peak_equity"] if s1["peak_equity"] > 0 else 0
            if dd_s1 > PLANNER_PARAMS["S1_STOP"]: s1["stopped"] = True
            
            s1_lev = 0.0
            if not s1["stopped"]:
                dec = self.calculate_decay(s1["entry_date"], ts, PLANNER_PARAMS["S1_DECAY"])
                s1_lev = float(s1_trend) * dec
            
            # S2
            sma400 = daily_data['p_sma400']
            s2_trend = 1 if curr_price > sma400 else -1
            s2 = p_state["s2"]
            if s2["trend"] != s2_trend:
                s2["trend"] = s2_trend
                s2["stopped"] = False
                s2["peak_equity"] = p_state["s2_equity"]
                
            if p_state["s2_equity"] > s2["peak_equity"]: s2["peak_equity"] = p_state["s2_equity"]
            dd_s2 = (s2["peak_equity"] - p_state["s2_equity"]) / s2["peak_equity"] if s2["peak_equity"] > 0 else 0
            if dd_s2 > PLANNER_PARAMS["S2_STOP"]: s2["stopped"] = True
            
            dist_pct = abs(curr_price - sma400) / sma400
            is_prox = dist_pct < PLANNER_PARAMS["S2_PROX"]
            tgt_size = 0.5 if is_prox else 1.0
            
            s2_lev = 0.0
            if s2["stopped"]:
                if is_prox:
                    s2["stopped"] = False
                    s2["peak_equity"] = p_state["s2_equity"]
                    s2_lev = float(s2_trend) * tgt_size
            else:
                s2_lev = float(s2_trend) * tgt_size
            
            planner_lev = max(-2.0, min(2.0, s1_lev + s2_lev))
            
            # Store Planner decisions for next virtual update
            p_state["last_lev_s1"] = s1_lev
            p_state["last_lev_s2"] = s2_lev

            # B. Tumbler
            t_s = t_state
            iii = daily_data['iii']
            
            base_lev_t = TUMBLER_PARAMS["LEVS"][2]
            if iii < TUMBLER_PARAMS["III_TH"][0]: base_lev_t = TUMBLER_PARAMS["LEVS"][0]
            elif iii < TUMBLER_PARAMS["III_TH"][1]: base_lev_t = TUMBLER_PARAMS["LEVS"][1]
            
            if iii < TUMBLER_PARAMS["FLAT_THRESH"]: t_s["flat_regime"] = True
            if t_s["flat_regime"]:
                s1, s2 = daily_data['t_sma1'], daily_data['t_sma2']
                b1, b2 = s1 * TUMBLER_PARAMS["BAND"], s2 * TUMBLER_PARAMS["BAND"]
                if abs(curr_price - s1) <= b1 or abs(curr_price - s2) <= b2:
                    t_s["flat_regime"] = False
            
            tumbler_lev = 0.0
            if not t_s["flat_regime"]:
                s1, s2 = daily_data['t_sma1'], daily_data['t_sma2']
                if curr_price > s1 and curr_price > s2: tumbler_lev = base_lev_t
                elif curr_price < s1 and curr_price < s2: tumbler_lev = -base_lev_t
            
            tumbler_lev *= (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)

            # C. Gainer
            # 1H MACD
            m1h = 0.0
            for i, w in enumerate(GAINER_PARAMS["MACD_1H"]['weights']):
                val = ind_1h_dict[ts][f'g_macd_val_{i}']
                sig = ind_1h_dict[ts][f'g_macd_sig_{i}']
                m1h += (1.0 if val > sig else -1.0) * w
            m1h /= sum(GAINER_PARAMS["MACD_1H"]['weights'])
            
            # 1D MACD
            m1d = 0.0
            for i, w in enumerate(GAINER_PARAMS["MACD_1D"]['weights']):
                val = daily_data[f'g_macd_val_{i}']
                sig = daily_data[f'g_macd_sig_{i}']
                m1d += (1.0 if val > sig else -1.0) * w
            m1d /= sum(GAINER_PARAMS["MACD_1D"]['weights'])
            
            # 1D SMA
            s1d = 0.0
            for i, w in enumerate(GAINER_PARAMS["SMA_1D"]['weights']):
                p = GAINER_PARAMS["SMA_1D"]['params'][i]
                sma = daily_data[f'g_sma_{p}']
                s1d += (1.0 if curr_price > sma else -1.0) * w
            s1d /= sum(GAINER_PARAMS["SMA_1D"]['weights'])
            
            gw = GAINER_PARAMS["GA_WEIGHTS"]
            gainer_lev = (m1h * gw["MACD_1H"] + m1d * gw["MACD_1D"] + s1d * gw["SMA_1D"]) / sum(gw.values())
            gainer_lev *= TARGET_STRAT_LEV

            # --- 4. Final Aggregation ---
            target_lev = planner_lev + tumbler_lev + gainer_lev
            
            # Apply cost for leverage change (approximate turnover cost)
            lev_delta = abs(target_lev - prev_target_lev)
            turnover_cost = lev_delta * fee_tier # fee_tier = slippage + fee
            
            real_equity -= (real_equity * turnover_cost)

            # Record
            results.append({
                'timestamp': ts,
                'price': curr_price,
                'equity': real_equity,
                'leverage': target_lev,
                'planner_lev': planner_lev,
                'tumbler_lev': tumbler_lev,
                'gainer_lev': gainer_lev,
                'drawdown': 0.0 # Calc later
            })
            
            prev_target_lev = target_lev
            prev_price = curr_price

        return pd.DataFrame(results).set_index('timestamp')
