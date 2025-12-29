import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backtester import DataManager, BacktestEngine
import numpy as np

st.set_page_config(page_title="Master Trader Backtest", layout="wide")

st.title("🚂 Master Trader Strategy Backtester")
st.markdown("Visualizing the `Planner`, `Tumbler`, and `Gainer` ensemble strategy on Binance Data.")

# Sidebar Controls
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
years = st.sidebar.slider("Years of History", 1, 8, 4)
capital = st.sidebar.number_input("Initial Capital ($)", 1000, 1000000, 10000)

@st.cache_data
def get_data(sym, yrs):
    return DataManager.fetch_binance_data(sym, "1h", yrs)

if st.sidebar.button("Run Backtest"):
    with st.spinner("Fetching Data from Binance (this may take a moment first time)..."):
        try:
            df = get_data(symbol, years)
            st.success(f"Loaded {len(df)} candles for {symbol}")
            
            with st.spinner("Running Backtest Engine..."):
                engine = BacktestEngine(df)
                results = engine.run(initial_capital=capital)
                
                # --- Metrics ---
                final_eq = results['equity'].iloc[-1]
                total_ret = (final_eq - capital) / capital * 100
                
                # Drawdown
                results['peak'] = results['equity'].cummax()
                results['dd'] = (results['peak'] - results['equity']) / results['peak']
                max_dd = results['dd'].max() * 100
                
                # CAGR
                days = (results.index[-1] - results.index[0]).days
                cagr = ((final_eq / capital) ** (365/days) - 1) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Final Equity", f"${final_eq:,.2f}")
                col2.metric("Total Return", f"{total_ret:.2f}%")
                col3.metric("Max Drawdown", f"{max_dd:.2f}%")
                col4.metric("CAGR", f"{cagr:.2f}%")
                
                # --- Plots ---
                
                # 1. Equity Curve
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=results.index, y=results['equity'], mode='lines', name='Equity', line=dict(color='#00ff00')))
                fig_eq.update_layout(title="Equity Curve", template="plotly_dark", height=500)
                st.plotly_chart(fig_eq, use_container_width=True)
                
                # 2. Drawdown
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=results.index, y=-results['dd']*100, mode='lines', name='Drawdown', fill='tozeroy', line=dict(color='#ff0000')))
                fig_dd.update_layout(title="Drawdown (%)", template="plotly_dark", height=300)
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # 3. Leverage Usage
                fig_lev = go.Figure()
                fig_lev.add_trace(go.Scatter(x=results.index, y=results['leverage'], name='Total Leverage', line=dict(width=1, color='white')))
                fig_lev.add_trace(go.Scatter(x=results.index, y=results['planner_lev'], name='Planner', line=dict(width=0.5, color='cyan')))
                fig_lev.add_trace(go.Scatter(x=results.index, y=results['tumbler_lev'], name='Tumbler', line=dict(width=0.5, color='yellow')))
                fig_lev.add_trace(go.Scatter(x=results.index, y=results['gainer_lev'], name='Gainer', line=dict(width=0.5, color='magenta')))
                fig_lev.update_layout(title="Leverage Contribution", template="plotly_dark", height=400)
                st.plotly_chart(fig_lev, use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

else:
    st.info("Click 'Run Backtest' in the sidebar to start.")
