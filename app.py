"""
Retail Sales Forecasting Dashboard
===================================
Run:  pip install streamlit pandas prophet plotly
      streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide", page_icon="📈")

st.title("📈 Retail Sales Forecasting Dashboard")
st.markdown("Upload your sales CSV or use the **sample data** to generate a forecast.")

# ── Sidebar controls ──────────────────────────────────────────────
st.sidebar.header("⚙️ Forecast Settings")
forecast_days = st.sidebar.slider("Forecast horizon (days)", 30, 180, 90, step=30)
uploaded = st.sidebar.file_uploader(
    "Upload CSV (columns: date, sales)", type="csv"
)
st.sidebar.markdown("---")
st.sidebar.info("**CSV format**\n\ndate,sales\n2022-01-01,1200\n2022-01-02,980\n...")

# ── Data loading ──────────────────────────────────────────────────
@st.cache_data
def make_sample_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    trend  = np.linspace(1000, 1800, len(dates))
    season = 300 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise  = rng.normal(0, 80, len(dates))
    sales  = trend + season + noise
    return pd.DataFrame({"date": dates, "sales": sales.clip(min=0)})

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["date"])
else:
    df = make_sample_data()
    st.info("🔔 Using **sample data**. Upload your own CSV from the sidebar.")

df = df.rename(columns={"date": "ds", "sales": "y"})
df["ds"] = pd.to_datetime(df["ds"])

# ── KPIs ─────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Records",     f"{len(df):,}")
k2.metric("Avg Daily Sales",   f"${df['y'].mean():,.0f}")
k3.metric("Peak Sales",        f"${df['y'].max():,.0f}")
k4.metric("Date Range",        f"{df['ds'].min().date()} → {df['ds'].max().date()}")

st.divider()

# ── Fit Prophet ───────────────────────────────────────────────────
with st.spinner("Training forecast model…"):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)
    future   = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

# ── Main chart ────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["ds"], y=df["y"],
    mode="lines", name="Actual Sales",
    line=dict(color="#4F8EF7", width=1.5)
))
fig.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    mode="lines", name="Forecast",
    line=dict(color="#FF6B6B", width=2, dash="dot")
))
fig.add_trace(go.Scatter(
    x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
    y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(255,107,107,0.15)",
    line=dict(color="rgba(255,107,107,0)"),
    name="Confidence Interval"
))
fig.update_layout(
    title="Sales Forecast with Confidence Interval",
    xaxis_title="Date", yaxis_title="Sales ($)",
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified", height=450
)
st.plotly_chart(fig, use_container_width=True)

# ── Components ────────────────────────────────────────────────────
st.subheader("📊 Forecast Components")
col1, col2 = st.columns(2)

trend_fig = go.Figure()
trend_fig.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["trend"],
    mode="lines", line=dict(color="#6BCB77", width=2)
))
trend_fig.update_layout(title="Trend", height=280,
                         xaxis_title="Date", yaxis_title="Value")
col1.plotly_chart(trend_fig, use_container_width=True)

if "yearly" in forecast.columns:
    season_fig = go.Figure()
    season_fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yearly"],
        mode="lines", line=dict(color="#FFD93D", width=2)
    ))
    season_fig.update_layout(title="Yearly Seasonality", height=280,
                              xaxis_title="Date", yaxis_title="Effect")
    col2.plotly_chart(season_fig, use_container_width=True)

# ── Forecast table ────────────────────────────────────────────────
st.subheader(f"📋 Next {forecast_days}-Day Forecast")
future_only = forecast[forecast["ds"] > df["ds"].max()][
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].copy()
future_only.columns = ["Date", "Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]
for c in ["Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]:
    future_only[c] = future_only[c].round(2)
st.dataframe(future_only.reset_index(drop=True), use_container_width=True)
st.download_button("⬇️ Download Forecast CSV",
                   future_only.to_csv(index=False),
                   "forecast.csv", "text/csv")
