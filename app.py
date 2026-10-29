"""
Solar Power Forecaster
An interactive tool: upload a time-series CSV (timestamp + solar power/irradiance
+ optional weather columns), pick a model, and see forecasts vs. a "persistence"
baseline, error metrics, and which inputs mattered most.
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Solar Power Forecaster", page_icon="☀️", layout="wide")


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
@st.cache_data
def make_demo_data() -> pd.DataFrame:
    """Synthetic hourly data for a northern location (~60N) over 2 years."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=24 * 730, freq="h")
    n = len(idx)
    doy = idx.dayofyear.values
    hour = idx.hour.values

    # Sun height above the horizon (simple astronomy formula, latitude 60 N)
    lat = np.radians(60)
    decl = np.radians(23.44 * np.sin(2 * np.pi * (284 + doy) / 365))
    hour_angle = np.radians(15 * (hour - 12))
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    clear_sky = np.clip(sin_elev, 0, None)

    # Cloud cover wanders slowly between 0 (clear) and 1 (overcast)
    cloud = np.zeros(n)
    c = 0.5
    for i in range(n):
        c = float(np.clip(0.9 * c + 0.05 + rng.normal(0, 0.12), 0, 1))
        cloud[i] = c

    irradiance = 1000 * clear_sky * (1 - 0.75 * cloud**1.5)
    temp = (
        -5
        + 15 * np.sin(2 * np.pi * (doy - 110) / 365)
        + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
        + rng.normal(0, 1.5, n)
    )
    power = 5 * irradiance / 1000 * (1 - 0.004 * (temp - 25)) + rng.normal(0, 0.05, n)
    power = np.clip(power, 0, None) * (clear_sky > 0)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "temperature_c": temp.round(2),
            "cloud_cover": cloud.round(3),
            "irradiance_wm2": irradiance.round(1),
            "power_kw": power.round(3),
        }
    )


@st.cache_data
def read_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Reads a CSV. Also handles NASA POWER files (header block, -999 = missing,
    separate YEAR/MO/DY/HR columns)."""
    text = raw_bytes.decode("utf-8", errors="ignore")
    if "-END HEADER-" in text:
        text = text.split("-END HEADER-", 1)[1]
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace(-999, np.nan)

    upper = {c.upper(): c for c in df.columns}
    if all(k in upper for k in ["YEAR", "MO", "DY", "HR"]):
        df.insert(
            0,
            "timestamp_built",
            pd.to_datetime(
                {
                    "year": df[upper["YEAR"]],
                    "month": df[upper["MO"]],
                    "day": df[upper["DY"]],
                    "hour": df[upper["HR"]],
                },
                errors="coerce",
            ),
        )
    return df


def guess_column(columns, keywords, default):
    for key in keywords:
        for col in columns:
            if key in str(col).lower():
                return col
    return default


def build_dataset(df, time_col, target_col, feature_cols, horizon):
    """Turns raw data into a table a model can learn from.
    Each row = 'what we know at time t'; label = target value `horizon` steps later."""
    d = df[[time_col, target_col] + feature_cols].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col]).drop_duplicates(subset=[time_col])
    d = d.sort_values(time_col).set_index(time_col)
    for col in [target_col] + feature_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.interpolate(limit=3)  # fill tiny gaps only

    hour = d.index.hour + d.index.minute / 60
    doy = d.index.dayofyear
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["season_cos"] = np.cos(2 * np.pi * doy / 365.25)

    d["lag_1"] = d[target_col].shift(1)
    d["lag_2"] = d[target_col].shift(2)
    d["lag_3"] = d[target_col].shift(3)
    d["rolling_mean_3"] = d[target_col].rolling(3).mean()

    d["target_future"] = d[target_col].shift(-horizon)
    return d.dropna()


@st.cache_data(show_spinner="Training model...")
def run_model(d, model_name, test_pct, target_col):
    X = d.drop(columns=["target_future"])
    y = d["target_future"]

    n_test = max(int(len(d) * test_pct / 100), 1)
    X_tr, X_te = X.iloc[:-n_test], X.iloc[-n_test:]  # chronological split
    y_tr, y_te = y.iloc[:-n_test], y.iloc[-n_test:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    model = models[model_name]
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    if y_tr.min() >= 0:
        pred = np.clip(pred, 0, None)  # solar output can't be negative

    persistence = X_te[target_col].values  # "tomorrow = right now" baseline

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = np.abs(model.coef_ * X_tr.std().values)
    importance = pd.Series(imp, index=X.columns).sort_values()

    return X_te.index, y_te.values, pred, persistence, importance


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# ----------------------------------------------------------------------------
# Page layout
# ----------------------------------------------------------------------------
st.title("☀️ Solar Power Forecaster")
st.caption(
    "Upload solar generation (or irradiance) data, train a machine-learning model, "
    "and see how well it predicts the future compared with a simple baseline. "
    "Built for a TKS Focus on AI for solar energy and northern microgrids."
)

# ---- Sidebar: data ---------------------------------------------------------
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Data source", ["Demo data (synthetic)", "Upload my own CSV"])

    if source == "Demo data (synthetic)":
        df = make_demo_data()
    else:
        file = st.file_uploader("Upload a CSV file", type=["csv"])
        if file is None:
            st.info("Upload a CSV with a timestamp column and a solar power/irradiance column.")
            st.stop()
        try:
            df = read_uploaded_csv(file.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file: {e}")
            st.stop()

    cols = list(df.columns)
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if len(numeric_cols) == 0:
        st.error("No numeric columns found in this file.")
        st.stop()

    st.header("2. Columns")
    time_default = guess_column(cols, ["timestamp", "datetime", "date", "time"], cols[0])
    time_col = st.selectbox("Timestamp column", cols, index=cols.index(time_default))

    target_options = [c for c in numeric_cols if c != time_col]
    if not target_options:
        st.error("Need at least one numeric column besides the timestamp.")
        st.stop()
    target_default = guess_column(
        target_options, ["power", "kw", "generation", "output", "allsky", "irradiance", "energy"],
        target_options[0],
    )
    target_col = st.selectbox(
        "What do you want to forecast? (solar power / irradiance)",
        target_options,
        index=target_options.index(target_default),
    )

    feature_options = [c for c in numeric_cols if c not in (time_col, target_col)]
    feature_cols = st.multiselect(
        "Extra weather/input columns (optional)", feature_options, default=feature_options
    )

    st.header("3. Model")
    horizon = st.number_input(
        "Forecast horizon (rows ahead)",
        min_value=1,
        max_value=168,
        value=6,
        help="1 row = 1 time step. For hourly data, 6 means forecasting 6 hours ahead.",
    )
    model_name = st.selectbox(
        "Model", ["Gradient Boosting", "Random Forest", "Linear Regression"]
    )
    test_pct = st.slider("Percent of data held out for testing", 10, 50, 25)

# ---- Data preview ------------------------------------------------------------
with st.expander("Preview of your data", expanded=False):
    st.write(f"{len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(50), use_container_width=True)

# ---- Build + train -----------------------------------------------------------
data = build_dataset(df, time_col, target_col, feature_cols, int(horizon))
if len(data) < 100:
    st.error(
        f"Only {len(data)} usable rows after cleaning. Need at least 100. "
        "Check that the timestamp column is correct and the data has few gaps."
    )
    st.stop()
if len(data) > 300_000:
    st.warning("That's a big file. Training may be slow; consider Linear Regression.")

idx, actual, pred, persistence, importance = run_model(
    data, model_name, test_pct, target_col
)

step = idx.to_series().diff().median()
target_times = idx + step * int(horizon)

# ---- Metrics -------------------------------------------------------------------
m_rmse = rmse(actual, pred)
p_rmse = rmse(actual, persistence)
m_mae = float(mean_absolute_error(actual, pred))
m_r2 = float(r2_score(actual, pred))
skill = (1 - m_rmse / p_rmse) * 100 if p_rmse > 0 else float("nan")
worst_over = float((pred - actual).max())

st.subheader(f"Results: {model_name}, forecasting {int(horizon)} step(s) ahead")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("RMSE (model)", f"{m_rmse:.3f}")
c2.metric("RMSE (persistence baseline)", f"{p_rmse:.3f}")
c3.metric("MAE", f"{m_mae:.3f}")
c4.metric("R²", f"{m_r2:.3f}")
c5.metric("Skill vs. persistence", f"{skill:.1f}%")

if skill > 0:
    st.success(
        f"The model's error is {skill:.1f}% lower than simply assuming the future "
        "looks like right now. That means it is learning real patterns."
    )
else:
    st.warning(
        "The model does not beat the persistence baseline here. Try a different model, "
        "add weather columns, or use a longer forecast horizon."
    )

st.info(
    f"**Microgrid view:** the worst case where the model over-predicted solar was "
    f"**{worst_over:.2f}** (same units as your target). In a real microgrid, that is the "
    "size of the surprise shortfall a battery or diesel generator would need to cover."
)

# ---- Main chart ------------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=target_times, y=actual, name="Actual", line=dict(width=1.5)))
fig.add_trace(go.Scatter(x=target_times, y=pred, name=f"{model_name} forecast", line=dict(width=1.5)))
fig.add_trace(
    go.Scatter(
        x=target_times, y=persistence, name="Persistence baseline",
        line=dict(width=1, dash="dot"), visible="legendonly",
    )
)
fig.update_layout(
    title="Actual vs. predicted (test period only, data the model never saw)",
    xaxis_title="Time", yaxis_title=target_col, height=480,
    xaxis=dict(rangeslider=dict(visible=True)),
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig, use_container_width=True)

# ---- Secondary charts --------------------------------------------------------------
left, right = st.columns(2)

with left:
    err = pd.DataFrame({"hour": target_times.hour, "abs_error": np.abs(pred - actual)})
    by_hour = err.groupby("hour")["abs_error"].mean()
    fig2 = go.Figure(go.Bar(x=by_hour.index, y=by_hour.values))
    fig2.update_layout(
        title="Average error by hour of day", xaxis_title="Hour", yaxis_title="Mean absolute error",
        height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    fig3 = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h"))
    fig3.update_layout(
        title="What the model relied on most", xaxis_title="Relative importance", height=380
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---- Download ----------------------------------------------------------------------
out = pd.DataFrame(
    {"forecast_for_time": target_times, "actual": actual, "predicted": pred, "persistence": persistence}
)
st.download_button(
    "Download predictions as CSV",
    out.to_csv(index=False).encode("utf-8"),
    file_name="solar_forecast_results.csv",
    mime="text/csv",
)

# ---- Explanation ---------------------------------------------------------------------
with st.expander("How this works and its limitations"):
    st.markdown(
        """
**What the model sees at each moment:** the latest weather/input columns you selected, the last
few values of the target (lags), a 3-step rolling average, and time-of-day / time-of-year signals.
It then predicts the target value *horizon* steps into the future.

**Honest evaluation:** the last part of your data is held out and never used for training
(a chronological split, no shuffling). Results are compared with a **persistence baseline**
("the future equals right now"), which is the standard benchmark in solar forecasting research.

**Limitations:** weather inputs are the *observed* values at time t, not real weather forecasts
(NWP) for t + horizon, so this is a simplified version of what operators use. It assumes
evenly spaced data, and it cannot see events outside your dataset (ice fog, snow on panels).
        """
    )
