from pathlib import Path
from datetime import datetime
from typing import Optional
import os

import numpy as np
import pandas as pd

# Set Keras backend to JAX before importing keras
os.environ["KERAS_BACKEND"] = "jax"
import keras

CPU_WATTS_DEFAULT = 65.0
MODEL_PATH_DEFAULT = "lstm_forecast.keras"
CSV_INTENSITY_PATH_DEFAULT = "IN_2024_hourly.csv"


def load_intensity_csv(csv_path: str) -> pd.DataFrame:
    """
    Load India 2024 hourly carbon intensity CSV from Electricity Maps.
    Expects at least:
      - 'Datetime (UTC)'
      - 'Carbon intensity gCO₂eq/kWh (direct)' (with special characters)
    Returns sorted DataFrame with:
      - datetime_utc
      - ci_direct_g_per_kwh
      - ci_kg_per_kwh
      - hour (0–23)
    """
    df = pd.read_csv(csv_path)
    
    # Convert datetime column
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    
    # Handle the carbon intensity column - it might have special characters
    # Try to find the correct column name
    carbon_col = None
    possible_names = [
        "Carbon intensity gCO₂eq/kWh (direct)",
        "Carbon intensity gCOâ‚‚eq/kWh (direct)",  # Your CSV has this encoding
        "Carbon intensity gCO2eq/kWh (direct)",
    ]
    
    for col_name in possible_names:
        if col_name in df.columns:
            carbon_col = col_name
            break
    
    if carbon_col is None:
        # If none found, try to find any column containing "Carbon intensity" and "direct"
        matching_cols = [col for col in df.columns if "Carbon intensity" in col and "direct" in col]
        if matching_cols:
            carbon_col = matching_cols[0]
        else:
            raise ValueError(
                f"Could not find carbon intensity column. Available columns: {list(df.columns)}"
            )
    
    # Rename columns
    df = df.rename(
        columns={
            "Datetime (UTC)": "datetime_utc",
            carbon_col: "ci_direct_g_per_kwh",
        }
    )
    
    # Sort and process
    df = df.sort_values("datetime_utc")
    df["ci_kg_per_kwh"] = df["ci_direct_g_per_kwh"] / 1000.0
    df["hour"] = df["datetime_utc"].dt.hour
    
    return df


def hourly_intensity_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean intensity per clock hour using ci_kg_per_kwh.
    Returns DataFrame with columns: hour, ci_kg_per_kwh
    """
    prof = df.groupby("hour")["ci_kg_per_kwh"].mean().reset_index()
    return prof


def grid_intensity_for_time(df_profile: pd.DataFrame, when: Optional[datetime] = None) -> float:
    """
    Look up intensity (kg CO2/kWh) from hourly profile for given UTC time.
    """
    if when is None:
        when = datetime.utcnow()
    hour = when.hour
    row = df_profile[df_profile["hour"] == hour]
    if row.empty:
        return float(df_profile["ci_kg_per_kwh"].mean())
    return float(row["ci_kg_per_kwh"].iloc[0])


def build_or_load_lstm(
    model_path: str,
    df_intensity: pd.DataFrame,
    cpu_watts: float = CPU_WATTS_DEFAULT,
):
    """
    Train or load a simple LSTM mapping 10-hour histories of:
      - time index
      - emissions (kg)
      - intensity (g/kWh)
    to next-hour emissions (kg).
    Uses intensity from df_intensity and synthetic CPU load.
    
    Training typically takes 10-30 seconds depending on data size and CPU.
    """
    model_path_obj = Path(model_path)
    
    # Load existing model if available
    if model_path_obj.exists():
        try:
            print(f"Loading existing model from {model_path}...")
            model = keras.saving.load_model(model_path)
            print("✓ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Training new model...")

    # Train new model
    print("Training LSTM model... This will take 10-30 seconds.")
    intensities = df_intensity["ci_kg_per_kwh"].values.astype(np.float32)
    X, y = [], []

    for i in range(10, len(intensities) - 1):
        seq_int = intensities[i - 10 : i]              # kg CO2/kWh
        cpu_seq = np.random.beta(2, 5, 10) * 90.0      # synthetic CPU %, 0–90
        power_watts = cpu_seq / 100.0 * cpu_watts
        energy_kwh = power_watts / 1000.0              # 1 hour interval
        emissions_kg = energy_kwh * seq_int

        t_idx = np.arange(10, dtype=np.float32)
        X.append(
            np.column_stack(
                [
                    t_idx,
                    emissions_kg.astype(np.float32),
                    (seq_int * 1000.0).astype(np.float32),  # back to g/kWh
                ]
            )
        )
        y.append(np.float32(emissions_kg[-1]))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    print(f"Training data prepared: {X.shape[0]} samples")

    # Build model
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(10, 3)),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ],
        name="LSTM_Emissions_Forecaster"
    )
    
    model.compile(optimizer="adam", loss="mse")
    
    print("Training model (5 epochs)...")
    model.fit(X, y, epochs=5, batch_size=64, verbose=1)

    # Save model
    keras.saving.save_model(model, model_path)
    print(f"✓ Model trained and saved to {model_path}")
    
    return model


def prepare_lstm_input(
    recent_emissions: np.ndarray,
    recent_intensities_kg: np.ndarray,
) -> np.ndarray:
    """
    Build LSTM input (1, 10, 3) from last 10 values of emissions (kg) and
    intensities (kg/kWh).
    """
    assert recent_emissions.shape[0] == 10, f"Expected 10 emissions values, got {recent_emissions.shape[0]}"
    assert recent_intensities_kg.shape[0] == 10, f"Expected 10 intensity values, got {recent_intensities_kg.shape[0]}"
    
    t_idx = np.arange(10, dtype=np.float32)
    X = np.column_stack(
        [
            t_idx,
            recent_emissions.astype(np.float32),
            (recent_intensities_kg * 1000.0).astype(np.float32),  # convert to g/kWh
        ]
    )
    return X[None, :, :]  # shape (1, 10, 3)
    