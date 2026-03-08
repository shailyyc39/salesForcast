# 📈 Retail Sales Forecasting Dashboard

A Streamlit web app that forecasts retail sales using Facebook's Prophet model.

## 🚀 How to Run (Step by Step)

1. **Install Python** (if you haven't): https://python.org/downloads

2. **Open your terminal / command prompt** and run:

```bash
pip install streamlit pandas prophet plotly
```

3. **Run the app:**
```bash
streamlit run app.py
```

4. Your browser will open automatically at `http://localhost:8501`

## 📂 Using Your Own Data

Create a CSV file with two columns:
```
date,sales
2023-01-01,1500
2023-01-02,1320
...
```
Then upload it in the sidebar.

## ✨ Features
- Auto-generates sample data if no file uploaded
- 30–180 day forecast horizon (adjustable)
- Confidence interval shading
- Trend & seasonality breakdown charts
- Download forecast as CSV
