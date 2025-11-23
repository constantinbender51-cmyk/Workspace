#!/usr/bin/env python3
"""
Railway Portrait – Binance-Bitcoin Daily 2022 → Linear-Regression Server
------------------------------------------------------------------------
A single, self-contained Python script that

1. Pulls daily BTCUSDT candles from Binance for 1 Dec 2021 – 31 Dec 2022
2. Builds a tiny feature set:
   - 7-day  SMA of close price
   - 365-day SMA of close price
   - 5-day  SMA of volume (USDT)
   - 10-day SMA of volume (USDT)
3. Trains a plain-vanilla Linear-Regression model to predict *next-day* close
4. Spins up a web server on 0.0.0.0:8080 and plots
   - true vs. predicted prices
   - cumulative absolute error
   - feature importance (coefs)
   - daily-compounding capital curve (€) that goes long when actual > predicted
     and short when actual < predicted (mean-reversion)
   All served at the root route “/” as an interactive Bokeh page
"""

import datetime as dt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column
from bokeh.models import HoverTool, Div
from flask import Flask, render_template_string
import requests
import threading
import time

# --------------------------------------------------
# 1.  Binance REST endpoint & date range
# --------------------------------------------------
URL      = "https://api.binance.com/api/v3/klines"
START_TS = int(dt.datetime(2021, 1, 1).timestamp() * 1000)   # 1 Dec 2021
END_TS   = int(dt.datetime(2024, 1, 1).timestamp() * 1000)    # 1 Jan 2023

def fetch_2022_daily():
    """
    Télécharge les chandeliers journaliers BTCUSDT sur Binance
    période : 1 déc 2021 → 31 déc 2022
    """
    params = dict(
        symbol='BTCUSDT',
        interval='1d',
        startTime=START_TS,
        endTime=END_TS,
        limit=1000
    )
    all_rows = []
    while True:
        r = requests.get(URL, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        if not js:
            break
        all_rows.extend(js)
        params['startTime'] = js[-1][6] + 1          # dernier close_time + 1 ms
        if len(js) < 1000:
            break

    df = pd.DataFrame(all_rows,
                      columns=['open_time', 'open', 'high', 'low', 'close',
                               'volume', 'close_time', 'quote_vol', 'trades',
                               'taker_base', 'taker_quote', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df = df[['open_time', 'close', 'volume']].rename(columns={'open_time': 'date'})
    return df.set_index('date').sort_index()

# --------------------------------------------------
# 2. Feature engineering & train/test split
# --------------------------------------------------
def add_features(df):
    df = df.copy()
    df['sma7_price']   = df['close'].rolling(7).mean()
    df['sma365_price'] = df['close'].rolling(365).mean()
    df['sma5_vol']     = df['volume'].rolling(5).mean()
    df['sma10_vol']    = df['volume'].rolling(10).mean()
    # target: next-day close
    df['target'] = df['close'].shift(-1)
    return df.dropna()

def train_model(df):
    feats = ['sma7_price', 'sma365_price', 'sma5_vol', 'sma10_vol']
    X, y = df[feats], df['target']
    if len(X) < 50:
        raise RuntimeError(f'Not enough data: {len(X)} rows after feature engineering')

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError('Train/test split yielded empty array')

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, X_test, y_test, preds, feats

# --------------------------------------------------
# 3. Build Bokeh plots – mobile first
# --------------------------------------------------
def build_dashboard(y_test, preds, model, feats, X_test):
    # 1. True vs Predicted
    p1 = figure(
        title="BTCUSDT – True vs Predicted (2022 test split)",
        x_axis_label='Date', y_axis_label='Price (USDT)',
        x_axis_type='datetime', sizing_mode="stretch_width",
        height=350, toolbar_location='above', tools='pan,xwheel_zoom,reset'
    )
    src = ColumnDataSource(data={'date': y_test.index,
                                 'true': y_test.values,
                                 'pred': preds})
    p1.line('date', 'true', legend_label='True', color='black', source=src)
    p1.line('date', 'pred', legend_label='Pred',  color='red',  source=src)
    p1.add_tools(HoverTool(
        tooltips=[('date', '@date{%F}'), ('true', '@true{0,0.0}'), ('pred', '@pred{0,0.0}')],
        formatters={'@date': 'datetime'}))

    # 2. Cumulative absolute error
    cumerr = np.cumsum(np.abs(y_test.values - preds))
    p2 = figure(title="Cumulative Absolute Error", x_axis_type='datetime',
                sizing_mode="stretch_width", height=250,
                toolbar_location='above', tools='pan,xwheel_zoom,reset')
    p2.line(y_test.index, cumerr, color='orange', line_width=2)

    # 3. Feature importance
    coef_df = pd.DataFrame({'feature': feats, 'coef': model.coef_})
    p3 = figure(x_range=feats, title="Linear-Regression Coefficients",
                sizing_mode="stretch_width", height=250,
                toolbar_location='above', tools='pan,xwheel_zoom,reset')
    p3.vbar(x='feature', top='coef', width=0.7, source=ColumnDataSource(coef_df))
    p3.xaxis.major_label_orientation = 0.8

    # 4. DAILY RETURNS (fraction) – red=long, blue=short
    returns = []
    position_log = []          # 1 = long, -1 = short

    for i in range(1, len(y_test)):
        prev_price = y_test.iloc[i-1]
        curr_price = y_test.iloc[i]
        pred       = preds[i-1]
        pos = 1 if prev_price > pred else -1
        position_log.append(pos)

        r = (curr_price / prev_price - 1.0)
        returns.append(r if pos > 0 else -r)      # flip sign for short

    ret_ser = pd.Series(returns, index=y_test.index[1:])   # skip first day

    # background bands
    long_days  = [(y_test.index[i], y_test.index[i+1]) for i, p in enumerate(position_log) if p == 1]
    short_days = [(y_test.index[i], y_test.index[i+1]) for i, p in enumerate(position_log) if p == -1]

    p4 = figure(title="Daily strategy returns – red=long, blue=short",
                x_axis_type='datetime', y_axis_label='Return (fraction)',
                sizing_mode="stretch_width", height=350,
                toolbar_location='above', tools='pan,xwheel_zoom,reset')

    for start, end in long_days:
        p4.vbar(x=start, width=(end-start), top=1e6, bottom=-1e6, color='red', alpha=0.08)
    for start, end in short_days:
        p4.vbar(x=start, width=(end-start), top=1e6, bottom=-1e6, color='blue', alpha=0.08)

    # zero line
    p4.line([ret_ser.index[0], ret_ser.index[-1]], [0, 0], color='black', line_width=1, line_dash='dashed')

    # returns line
    p4.line(ret_ser.index, ret_ser.values, color='green', line_width=2)

    p4.add_tools(HoverTool(tooltips=[('date', '@x{%F}'), ('return', '@y{0.00 %}')],
                           formatters={'@x': 'datetime'}))

    # stats banner
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    ret_ser = pd.Series(returns, index=y_test.index[1:])   # already done above
    ...
    final_ret = ret_ser.iloc[-1] if len(ret_ser) else 0.0
    stats = Div(text=f"<b>MAE :</b> {mae:,.2f} USDT &nbsp;|&nbsp; <b>R² :</b> {r2:.3f} &nbsp;|&nbsp; <b>Final return :</b> {final_ret:+.2%}")

    return column(stats, p1, p2, p3, p4, sizing_mode="stretch_width")

# --------------------------------------------------
# 5. Flask glue
# --------------------------------------------------
app = Flask(__name__)
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Railway Portrait – BTC 2022 LR</title>
  {{ bokeh_resources | safe }}
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
  <h2>Railway Portrait – Binance BTCUSDT Daily 2022 → Linear-Regression</h2>
  {{ bokeh_plot | safe }}
</body>
</html>
"""

@app.route("/")
def index():
    from bokeh.embed import components
    from bokeh.resources import CDN
    script, div = components(layout)
    return render_template_string(HTML_PAGE,
                                  bokeh_resources=CDN.render(),
                                  bokeh_plot=div + script)

# --------------------------------------------------
# 6. Main
# --------------------------------------------------
if __name__ == "__main__":
    print("Fetching 2021-2022 daily candles …")
    raw = fetch_2022_daily()
    print("Engineering features …")
    feat_df = add_features(raw)
    print("Training model …")
    model, X_test, y_test, preds, feats = train_model(feat_df)
    print("Building dashboard …")
    layout = build_dashboard(y_test, preds, model, feats, X_test)
    print("Starting web server on 0.0.0.0:8080 …")
    app.run(host="0.0.0.0", port=8080, debug=False)
