# ☀️ Solar Power Forecaster

An interactive web app that forecasts solar power output from time-series data.
Upload your own CSV (timestamp + solar power/irradiance + optional weather columns),
choose a model, and compare it against a persistence baseline.

Built as the Build stage of my TKS Focus on AI for solar energy and northern microgrids.

## Features
- Upload any CSV (NASA POWER hourly files work directly) or use built-in demo data
- Linear Regression, Random Forest, Gradient Boosting
- RMSE, MAE, R², and skill score vs. persistence
- Interactive charts, error-by-hour, feature importance
- Download predictions as CSV

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
