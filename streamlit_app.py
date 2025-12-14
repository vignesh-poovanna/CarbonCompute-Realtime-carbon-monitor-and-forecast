from datetime import datetime, timedelta
from pathlib import Path
import time
import os

import numpy as np
import pandas as pd
import plotly.express as px
import psutil
import streamlit as st

# Set Keras backend before importing
os.environ["KERAS_BACKEND"] = "jax"
import keras

from emissions_core import (
    CPU_WATTS_DEFAULT,
    MODEL_PATH_DEFAULT,
    CSV_INTENSITY_PATH_DEFAULT,
    load_intensity_csv,
    hourly_intensity_profile,
    grid_intensity_for_time,
    build_or_load_lstm,
    prepare_lstm_input,
)

# ---------------- Streamlit config ----------------

st.set_page_config(
    page_title="CarbonCompute - Carbon Emissions Monitor",
    page_icon="♻️",
    layout="wide",
)

st.title("CarbonCompute - Carbon Emissions Monitor & Forecaster")
st.caption(
    "🔴 **LIVE** Real-time CPU tracking. Based on India 2024 grid-intensity data. "
    "Times shown in Indian Standard Time (IST).Built Using Keras 3 with JAX backend."
)

# ---------------- Sidebar controls ----------------

with st.sidebar:
    st.header("Controls")

    polling_interval = st.slider(
        "Sampling interval (seconds)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
    )
    cpu_tdp = st.slider(
        "CPU power baseline (W)",
        min_value=15,
        max_value=200,
        value=int(CPU_WATTS_DEFAULT),
        step=5,
    )
    forecast_horizon = st.slider(
        "Forecast horizon (hours)",
        min_value=1,
        max_value=24,
        value=6,
        step=1,
    )
    clear_btn = st.button("Clear history")
    
    st.markdown("---")
    st.markdown("### System Info")
    st.markdown(f"**Backend:** Keras 3 + JAX")
    st.markdown(f"**Python:** 3.14")
    st.markdown(f"**Timezone:** IST (UTC+5:30)")
    st.markdown(f"**Current Time:** {datetime.now().strftime('%H:%M:%S IST')}")


# ---------------- Cached data & model ----------------

@st.cache_data
def get_intensity_data(csv_path: str):
    """Load and cache intensity data"""
    try:
        df = load_intensity_csv(csv_path)
        profile = hourly_intensity_profile(df)
        return df, profile
    except FileNotFoundError:
        st.error(f"❌ CSV file not found: {csv_path}")
        st.info("Please ensure the CSV file exists in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        st.stop()


@st.cache_resource
def get_lstm_model(csv_path: str, model_path: str, cpu_watts: float):
    """Load or train the LSTM model"""
    try:
        with st.spinner("Loading/training LSTM model... This may take 10-30 seconds on first run."):
            df, _ = get_intensity_data(csv_path)
            model = build_or_load_lstm(model_path, df, cpu_watts)
        st.success("✓ Model ready!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading/training model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


df_intensity, profile = get_intensity_data(CSV_INTENSITY_PATH_DEFAULT)
lstm_model = get_lstm_model(
    CSV_INTENSITY_PATH_DEFAULT, MODEL_PATH_DEFAULT, cpu_tdp
)

# ---------------- In-memory tracking ----------------

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            "timestamp",
            "cpu_percent",
            "power_watts",
            "energy_kwh",
            "grid_intensity_kg_per_kwh",
            "emissions_kg",
        ]
    )

if clear_btn:
    st.session_state.history = st.session_state.history.iloc[0:0]
    st.rerun()

# ---------------- Take one new sample ----------------

# Get current time in both UTC and local
now_utc = datetime.utcnow()
now_local = datetime.now()  # Local time (IST for India)

cpu_percent = psutil.cpu_percent(interval=0.2)
power_watts_now = cpu_percent / 100.0 * cpu_tdp
energy_kwh_now = power_watts_now / 1000.0 * (polling_interval / 3600.0)

# Use UTC time for grid intensity lookup (CSV data is in UTC)
grid_kg_now = grid_intensity_for_time(profile, when=now_utc)
emissions_kg_now = energy_kwh_now * grid_kg_now

new_row = pd.DataFrame(
    [
        {
            "timestamp": now_local,  # Display local time in charts
            "cpu_percent": cpu_percent,
            "power_watts": power_watts_now,
            "energy_kwh": energy_kwh_now,
            "grid_intensity_kg_per_kwh": grid_kg_now,
            "emissions_kg": emissions_kg_now,
        }
    ]
)
st.session_state.history = pd.concat(
    [st.session_state.history, new_row], ignore_index=True
)

hist = st.session_state.history.copy()
hist["timestamp"] = pd.to_datetime(hist["timestamp"])

# ---------------- KPIs ----------------

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

total_emissions = hist["emissions_kg"].sum() if not hist.empty else 0.0
avg_cpu = float(hist["cpu_percent"].mean()) if not hist.empty else 0.0
avg_grid = float(hist["grid_intensity_kg_per_kwh"].mean()) if not hist.empty else 0.0
runtime_minutes = 0.0
if len(hist) > 1:
    runtime_minutes = (hist["timestamp"].iloc[-1] - hist["timestamp"].iloc[0]).total_seconds() / 60.0

with col_kpi1:
    st.metric("Total emissions (kg CO₂)", f"{total_emissions:,.4f}")
with col_kpi2:
    st.metric("Average CPU utilisation (%)", f"{avg_cpu:,.1f}")
with col_kpi3:
    st.metric("Average grid intensity (kg CO₂/kWh)", f"{avg_grid:,.3f}")
with col_kpi4:
    st.metric("Tracking duration (min)", f"{runtime_minutes:,.1f}")

st.markdown("---")

# ---------------- Live tracking charts ----------------

st.subheader("Live Tracking")

if hist.shape[0] >= 2:
    live_cols = st.columns(2)

    with live_cols[0]:
        fig_cpu = px.line(
            hist,
            x="timestamp",
            y="cpu_percent",
            title="CPU utilisation over time (Real-time)",
            labels={"timestamp": "Time (IST)", "cpu_percent": "CPU (%)"},
        )
        fig_cpu.update_layout(height=280, margin=dict(l=30, r=10, t=40, b=30))
        st.plotly_chart(fig_cpu, use_container_width=True)

    with live_cols[1]:
        fig_em = px.line(
            hist,
            x="timestamp",
            y="emissions_kg",
            title="Real-time emissions per sample (kg CO₂)",
            labels={"timestamp": "Time (IST)", "emissions_kg": "Emissions (kg)"},
        )
        fig_em.update_layout(height=280, margin=dict(l=30, r=10, t=40, b=30))
        st.plotly_chart(fig_em, use_container_width=True)
else:
    st.info("Collecting data… keep this app running to build up history.")

# ---------------- Historical charts ----------------

st.subheader("History & Grid Intensity Context")

if not hist.empty:
    hist["cumulative_emissions_kg"] = hist["emissions_kg"].cumsum()

hist_cols = st.columns(2)

with hist_cols[0]:
    if not hist.empty:
        fig_cum = px.area(
            hist,
            x="timestamp",
            y="cumulative_emissions_kg",
            title="Cumulative emissions since app start (kg CO₂)",
            labels={
                "timestamp": "Time",
                "cumulative_emissions_kg": "Cumulative (kg)",
            },
        )
        fig_cum.update_layout(height=300, margin=dict(l=30, r=10, t=40, b=30))
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.info("No data yet to display cumulative emissions.")

with hist_cols[1]:
    fig_hourly = px.line(
        hourly_intensity_profile(df_intensity),
        x="hour",
        y="ci_kg_per_kwh",
        title="Typical India grid intensity by hour of day (2024 averages)",
        labels={"hour": "Hour (0–23, UTC)", "ci_kg_per_kwh": "kg CO₂/kWh"},
    )
    fig_hourly.update_traces(mode="lines+markers")
    fig_hourly.update_layout(height=300, margin=dict(l=30, r=10, t=40, b=30))
    st.plotly_chart(fig_hourly, use_container_width=True)

# ---------------- Forecast panel ----------------

st.subheader("Short-Term Emissions Forecast")

forecast_cols = st.columns([2, 1])

recent = hist.tail(10).copy()
if recent.shape[0] < 10:
    with forecast_cols[0]:
        st.warning(
            f"Need at least 10 samples of history to run the LSTM forecast. "
            f"Currently have {recent.shape[0]} samples. Keep the app running."
        )
else:
    recent = recent.sort_values("timestamp")
    seq_em = recent["emissions_kg"].values.astype(np.float32)
    seq_ci_kg = recent["grid_intensity_kg_per_kwh"].values.astype(np.float32)

    try:
        X_in = prepare_lstm_input(seq_em, seq_ci_kg)
        next_emission_pred = float(lstm_model.predict(X_in, verbose=0)[0, 0])

        last_timestamp = recent["timestamp"].iloc[-1]
        future_times = [
            last_timestamp + timedelta(hours=i + 1) for i in range(forecast_horizon)
        ]

        future_emissions = []
        ema = next_emission_pred
        for _ in range(forecast_horizon):
            future_emissions.append(max(ema, 0.0))
            ema = 0.7 * ema + 0.3 * next_emission_pred

        df_forecast = pd.DataFrame(
            {
                "timestamp": future_times,
                "predicted_emissions_kg": future_emissions,
            }
        )

        with forecast_cols[0]:
            fig_fc = px.bar(
                df_forecast,
                x="timestamp",
                y="predicted_emissions_kg",
                title=f"Next {forecast_horizon} hours: predicted per-hour emissions (kg CO₂)",
                labels={
                    "timestamp": "Future time",
                    "predicted_emissions_kg": "Predicted kg CO₂",
                },
            )
            fig_fc.update_layout(height=320, margin=dict(l=30, r=10, t=40, b=30))
            st.plotly_chart(fig_fc, use_container_width=True)

        with forecast_cols[1]:
            st.markdown("##### Forecast summary")
            st.write(
                f"- Next-hour predicted emissions: **{next_emission_pred:.5f} kg CO₂**\n"
                f"- Average forecast (next {forecast_horizon} h): "
                f"**{np.mean(future_emissions):.5f} kg CO₂/h**\n"
                f"- Assumes grid intensity and workload similar to recent history."
            )
    except Exception as e:
        with forecast_cols[0]:
            st.error(f"Error generating forecast: {str(e)}")

# ---------------- Narrative summary ----------------

st.markdown("---")
st.subheader("Narrative Summary")

if hist.shape[0] < 5:
    st.write(
        "The monitor has just started collecting data. Leave this app running to "
        "build up a richer picture of CPU utilisation, grid intensity and emissions."
    )
else:
    last_cpu_str = f"{cpu_percent:.1f}%"
    last_grid_str = f"{grid_kg_now:.3f} kg CO₂/kWh"
    last_em_str = f"{emissions_kg_now:.6f} kg CO₂"
    total_str = f"{total_emissions:.4f} kg CO₂"
    dur_str = f"{runtime_minutes:.1f} minutes"

    st.write(
        f"Over the last {dur_str}, your system averaged about **{avg_cpu:.1f}% CPU utilisation**, "
        f"on an Indian grid with an average intensity of **{avg_grid:.3f} kg CO₂/kWh** "
        f"based on the 2024 hourly dataset."
    )
    st.write(
        f"This produced an estimated **{total_str}** of CO₂ emissions. "
        f"Most recently, CPU usage was **{last_cpu_str}**, the grid intensity was **{last_grid_str}**, "
        f"and that sampling interval emitted roughly **{last_em_str}**."
    )

st.caption(
    f"Sampling every {polling_interval} seconds. The page will auto-refresh. "
    f"Model training takes 10-30 seconds on first run."
)

# ---------------- Emissions Guidelines & Recommendations ----------------

st.markdown("---")
st.subheader(" Carbon Emissions Guidelines & Recommendations")

guidelines_cols = st.columns([2, 1])

with guidelines_cols[0]:
    # Calculate hourly and daily rates
    if len(hist) > 1:
        time_span_hours = runtime_minutes / 60.0
        hourly_rate = total_emissions / time_span_hours if time_span_hours > 0 else 0
        daily_projection = hourly_rate * 24
        monthly_projection = daily_projection * 30
        yearly_projection = daily_projection * 365
    else:
        hourly_rate = 0
        daily_projection = 0
        monthly_projection = 0
        yearly_projection = 0
    
    # Emission thresholds (kg CO2)
    HOURLY_THRESHOLD_LOW = 0.05      # Good - minimal usage
    HOURLY_THRESHOLD_MEDIUM = 0.15   # Moderate - typical usage
    HOURLY_THRESHOLD_HIGH = 0.30     # High - heavy workload
    HOURLY_THRESHOLD_CRITICAL = 0.50 # Critical - excessive usage
    
    # Determine current status
    if hourly_rate < HOURLY_THRESHOLD_LOW:
        status = "🟢 Excellent"
        status_color = "green"
        recommendation = "Your emissions are minimal. This is sustainable for continuous operation."
    elif hourly_rate < HOURLY_THRESHOLD_MEDIUM:
        status = "🟡 Good"
        status_color = "orange"
        recommendation = "Emissions are within normal range for typical computing tasks."
    elif hourly_rate < HOURLY_THRESHOLD_HIGH:
        status = "🟠 Moderate"
        status_color = "orange"
        recommendation = "Emissions are elevated. Consider optimizing workloads or reducing CPU-intensive tasks."
    elif hourly_rate < HOURLY_THRESHOLD_CRITICAL:
        status = "🔴 High"
        status_color = "red"
        recommendation = "Emissions are high. Review running applications and consider energy-saving measures."
    else:
        status = "⚠️ Critical"
        status_color = "red"
        recommendation = "⚠️ Emissions are critically high! Immediate action recommended: close unnecessary applications, review system processes."
    
    st.markdown(f"### Current Status: :{status_color}[{status}]")
    st.markdown(f"**Hourly Rate:** {hourly_rate:.4f} kg CO₂/hour")
    st.info(recommendation)
    
    # Projections
    st.markdown("#### Projected Emissions (if current rate continues)")
    proj_cols = st.columns(4)
    with proj_cols[0]:
        st.metric("Daily", f"{daily_projection:.3f} kg CO₂")
    with proj_cols[1]:
        st.metric("Weekly", f"{daily_projection * 7:.3f} kg CO₂")
    with proj_cols[2]:
        st.metric("Monthly", f"{monthly_projection:.3f} kg CO₂")
    with proj_cols[3]:
        st.metric("Yearly", f"{yearly_projection:.2f} kg CO₂")
    
    # Context comparisons
    st.markdown("#### Real-World Context")
    comparison_text = ""
    
    if yearly_projection > 0:
        # 1 kg CO2 = driving ~4 km in average car
        km_equivalent = yearly_projection * 4
        # 1 tree absorbs ~21 kg CO2 per year
        trees_needed = yearly_projection / 21
        # Average smartphone charge = 0.008 kg CO2
        phone_charges = yearly_projection / 0.008
        
        comparison_text = f"""
        At your current rate, your annual emissions ({yearly_projection:.2f} kg CO₂) are equivalent to:
        -  Driving approximately **{km_equivalent:.1f} km** in an average car
        -  Would require **{trees_needed:.1f} trees** to offset (for one year)
        -  Charging a smartphone approximately **{phone_charges:,.0f} times**
        """
    else:
        comparison_text = "Keep the app running for a few minutes to see meaningful projections."
    
    st.markdown(comparison_text)

with guidelines_cols[1]:
    st.markdown("####  Recommended Thresholds")
    st.markdown(f"""
    **Per Hour:**
    - 🟢 Excellent: < {HOURLY_THRESHOLD_LOW} kg CO₂
    - 🟡 Good: < {HOURLY_THRESHOLD_MEDIUM} kg CO₂
    - 🟠 Moderate: < {HOURLY_THRESHOLD_HIGH} kg CO₂
    - 🔴 High: < {HOURLY_THRESHOLD_CRITICAL} kg CO₂
    - ⚠️ Critical: > {HOURLY_THRESHOLD_CRITICAL} kg CO₂
    
    **Per Day (24h):**
    - 🟢 Excellent: < {HOURLY_THRESHOLD_LOW * 24:.2f} kg CO₂
    - 🟡 Good: < {HOURLY_THRESHOLD_MEDIUM * 24:.2f} kg CO₂
    - 🟠 Moderate: < {HOURLY_THRESHOLD_HIGH * 24:.2f} kg CO₂
    - 🔴 High: < {HOURLY_THRESHOLD_CRITICAL * 24:.2f} kg CO₂
    
    **Per Year:**
    - 🟢 Target: < {HOURLY_THRESHOLD_MEDIUM * 24 * 365:.1f} kg CO₂
    - ⚠️ Concern: > {HOURLY_THRESHOLD_HIGH * 24 * 365:.1f} kg CO₂
    """)
    
    st.markdown("---")
    st.markdown("####  Tips to Reduce Emissions")
    st.markdown("""
    1. Close unused applications
    2. Use energy-saving mode
    3. Optimize background processes
    4. Schedule heavy tasks during low-carbon hours
    5. Consider using renewable energy
    6. Upgrade to energy-efficient hardware
    """)

# Visual threshold indicator
st.markdown("####  Your Position on Emissions Scale")
if hourly_rate > 0:
    # Create a visual scale
    max_scale = max(HOURLY_THRESHOLD_CRITICAL * 1.5, hourly_rate * 1.2)
    
    threshold_df = pd.DataFrame({
        'Category': ['Excellent', 'Good', 'Moderate', 'High', 'Critical', 'Your Current Rate'],
        'Emissions (kg CO₂/hour)': [
            HOURLY_THRESHOLD_LOW,
            HOURLY_THRESHOLD_MEDIUM,
            HOURLY_THRESHOLD_HIGH,
            HOURLY_THRESHOLD_CRITICAL,
            max_scale,
            hourly_rate
        ],
        'Type': ['Threshold', 'Threshold', 'Threshold', 'Threshold', 'Threshold', 'Current']
    })
    
    fig_threshold = px.bar(
        threshold_df[threshold_df['Type'] == 'Threshold'],
        x='Category',
        y='Emissions (kg CO₂/hour)',
        title='Emission Thresholds vs Your Current Rate',
        color='Emissions (kg CO₂/hour)',
        color_continuous_scale=['green', 'yellow', 'orange', 'red']
    )
    
    # Add a line for current rate
    fig_threshold.add_hline(
        y=hourly_rate,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Your Rate: {hourly_rate:.4f} kg CO₂/h",
        annotation_position="right"
    )
    
    fig_threshold.update_layout(height=300, margin=dict(l=30, r=10, t=40, b=30))
    st.plotly_chart(fig_threshold, use_container_width=True)
else:
    st.info("Keep the app running to see your position on the emissions scale.")

# ---------------- Auto-refresh ----------------
# Auto-refresh the page based on polling interval
time.sleep(polling_interval)
st.rerun() 