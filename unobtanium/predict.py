"""
Aurora Borealis Predictive Model
================================
Predicts magnetic perturbations (X component at Kiruna) and Kp index
from ACE/DSCOVR solar wind data, with seasonal/temporal feature engineering.

Outputs:
  - Predicted Kp index (perturbation scale)
  - Aurora probability at mid-latitudes
  - Estimated equatorward auroral boundary (latitude)
  - Predicted X component at Kiruna

Usage:
    python predict_old.py train
    python predict_old.py retrain
    python predict_old.py forecast --hours 10 --speed 600
    python predict_old.py validate
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

torch.serialization.add_safe_globals([QuantileRegression])

# ===========================================================================
# CONFIGURATION
# ===========================================================================
DATA_DIR = Path("../data")
MODEL_DIR = Path(__file__).resolve().parent / "models"
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "hourly_dataset.parquet"

INPUT_CHUNK = 336
OUTPUT_CHUNK = 24
N_EPOCHS = 1
BATCH_SIZE = 64
HIDDEN_SIZE = 128
LSTM_LAYERS = 2
ATTENTION_HEADS = 4
DROPOUT = 0.1
LR = 1e-3

L1_DISTANCE_KM = 1_500_000

TARGET_COLS = ["X", "Kp"]
PAST_COV_COLS = ["Speed", "Bz", "Density", "Bt", "Bz_south", "Pdyn", "coupling"]
FUTURE_COV_COLS = ["sin_doy", "cos_doy", "sin_hour", "cos_hour", "sin_rm", "cos_rm"]


# ===========================================================================
# DATA LOADING & FEATURE ENGINEERING
# ===========================================================================

def load_and_prepare(data_dir: Path = DATA_DIR, use_cache: bool = True) -> pd.DataFrame:
    """Load all data sources, merge, resample to 1 h, add features."""
    if use_cache and CACHE_PATH.exists():
        print(f"Loading cached dataset from {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    print("Loading solar wind (smooth-solarwinds.csv) …")
    sw = pd.read_csv(
        data_dir / "solarwinds-ace-compiled" / "smooth-solarwinds.csv",
        sep=";",
    )
    sw["Date"] = pd.to_datetime(sw["Date"], format="%Y%m%d%H%M%S")

    print("Loading Kiruna magnetometer (smooth.csv) …")
    kiruna = pd.read_csv(
        data_dir / "mag-kiruna-compiled" / "smooth.csv",
        sep=";",
    )
    kiruna["Date"] = pd.to_datetime(kiruna["Date"], format="%Y%m%d%H%M%S")

    print("Loading Kp index …")
    kp = pd.read_csv(data_dir / "kp-compiled" / "kp.csv", sep=";")
    kp["Date"] = pd.to_datetime(kp["Date"], format="%Y%m%d%H%M%S")

    kiruna = kiruna.set_index("Date").resample("1h").mean()
    sw = sw.set_index("Date").resample("1h").mean()
    kp = kp.set_index("Date").resample("1h").mean().interpolate(method="linear")

    df = kiruna.join(sw, how="inner").join(kp, how="inner")
    df = df.interpolate(method="linear").dropna().astype("float32")

    doy = df.index.dayofyear.values.astype("float32")
    hour = (df.index.hour + df.index.minute / 60.0).astype("float32")

    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")

    df["sin_hour"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24).astype("float32")

    df["sin_rm"] = np.sin(4 * np.pi * doy / 365.25).astype("float32")
    df["cos_rm"] = np.cos(4 * np.pi * doy / 365.25).astype("float32")

    df["Bz_south"] = df["Bz"].clip(upper=0).astype("float32")
    df["Pdyn"] = (df["Density"] * df["Speed"] ** 2 * 1.6726e-6).astype("float32")

    # Approximate |By| from Bt and Bz when By is not directly available
    by2 = (df["Bt"] ** 2 - df["Bz"] ** 2).clip(lower=0)
    by = np.sqrt(by2).astype("float32")
    theta_c = np.arctan2(by, -df["Bz"]).astype("float32")

    df["coupling"] = (
        df["Speed"].abs() ** (4 / 3)
        * df["Bt"].abs() ** (2 / 3)
        * np.abs(np.sin(theta_c / 2)) ** (8 / 3)
    ).astype("float32")

    df = df.reset_index()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    print(f"Cached {len(df)} hourly rows → {CACHE_PATH}")

    return df


def build_timeseries(df: pd.DataFrame):
    target = TimeSeries.from_dataframe(df, time_col="Date", value_cols=TARGET_COLS)
    past_cov = TimeSeries.from_dataframe(df, time_col="Date", value_cols=PAST_COV_COLS)
    future_cov = TimeSeries.from_dataframe(df, time_col="Date", value_cols=FUTURE_COV_COLS)
    return target, past_cov, future_cov


# ===========================================================================
# MODEL
# ===========================================================================

def create_model(n_epochs: int = N_EPOCHS, force_reset: bool = True) -> TFTModel:
    return TFTModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        num_attention_heads=ATTENTION_HEADS,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        n_epochs=n_epochs,
        optimizer_kwargs={"lr": LR},
        add_relative_index=True,
        random_state=42,
        model_name="aurora_tft",
        work_dir=str(MODEL_DIR),
        save_checkpoints=True,
        force_reset=force_reset,
        pl_trainer_kwargs={
            "accelerator": "mps",
            "devices": 1,
        },
        #pl_trainer_kwargs={
        #    "accelerator": "cpu",  # Change to "gpu" if both PCs have NVIDIA GPUs
        #    "devices": 1,  # 1 process per PC
        #    "num_nodes": 2,  # TOTAL number of PCs (Must be 2!)
        #    "strategy": "ddp",
        #},
    )

def load_trained_model() -> TFTModel:
    return TFTModel.load_from_checkpoint(
        "aurora_tft",
        work_dir=str(MODEL_DIR),
        best=True,
        weights_only=False,
    )


def _save_scalers(scaler_target, scaler_past, scaler_future):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "scalers.pkl", "wb") as f:
        pickle.dump(
            {
                "target": scaler_target,
                "past_cov": scaler_past,
                "future_cov": scaler_future,
            },
            f,
        )


def _load_scalers():
    with open(MODEL_DIR / "scalers.pkl", "rb") as f:
        s = pickle.load(f)
    return s["target"], s["past_cov"], s["future_cov"]


# ===========================================================================
# TRAINING
# ===========================================================================

def train_model(retrain: bool = False):
    df = load_and_prepare()
    target, past_cov, future_cov = build_timeseries(df)

    train_end_idx = int(len(df) * 0.8)
    val_end_idx = int(len(df) * 0.9)

    train_end_time = df["Date"].iloc[train_end_idx]
    val_end_time = df["Date"].iloc[val_end_idx]

    # Split brut avant scaling
    target_train, target_tmp = target.split_before(train_end_time)
    target_val, target_test = target_tmp.split_before(val_end_time)

    past_train, past_tmp = past_cov.split_before(train_end_time)
    past_val, past_test = past_tmp.split_before(val_end_time)

    future_train, future_tmp = future_cov.split_before(train_end_time)
    future_val, future_test = future_tmp.split_before(val_end_time)

    # Fit scalers sur train uniquement
    scaler_target = Scaler()
    scaler_past = Scaler()
    scaler_future = Scaler()

    ts_train = scaler_target.fit_transform(target_train)
    ts_val = scaler_target.transform(target_val)
    ts_test = scaler_target.transform(target_test)

    pc_train = scaler_past.fit_transform(past_train)
    pc_val = scaler_past.transform(past_val)
    pc_test = scaler_past.transform(past_test)

    fc_train = scaler_future.fit_transform(future_train)
    fc_val = scaler_future.transform(future_val)
    fc_test = scaler_future.transform(future_test)

    _save_scalers(scaler_target, scaler_past, scaler_future)

    if retrain:
        try:
            print("Loading existing model for continued training …")
            model = load_trained_model()
            model.model_params["n_epochs"] = 10
        except Exception:
            print("No checkpoint found — starting from scratch.")
            model = create_model(force_reset=True)
    else:
        model = create_model(force_reset=True)

    print(
        f"Training samples: {len(ts_train)}, "
        f"Validation samples: {len(ts_val)}, "
        f"Test samples: {len(ts_test)}"
    )

    model.fit(
        ts_train,
        past_covariates=pc_train,
        future_covariates=fc_train,
        val_series=ts_val,
        val_past_covariates=pc_val,
        val_future_covariates=fc_val,
        verbose=True,
    )

    print(f"Model saved to {MODEL_DIR / 'aurora_tft'}")
    return model


# ===========================================================================
# GEOPHYSICAL MAPPING
# ===========================================================================

def kp_to_aurora_latitude(kp: np.ndarray) -> np.ndarray:
    return 67.0 - 3.2 * np.clip(kp, 0, 9)


def kp_to_aurora_probability(kp: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(kp - 4.0)))


def travel_time_hours(speed_km_s: float) -> float:
    if speed_km_s <= 0:
        return float("inf")
    return L1_DISTANCE_KM / speed_km_s / 3600.0


# ===========================================================================
# FORECAST
# ===========================================================================

def forecast(n_hours: int = 5, current_speed: float | None = None):
    if n_hours <= 0:
        raise ValueError("--hours must be a positive integer")

    df = load_and_prepare()
    target, past_cov, future_cov = build_timeseries(df)

    scaler_target, scaler_past, scaler_future = _load_scalers()
    target_sc = scaler_target.transform(target)
    past_cov_sc = scaler_past.transform(past_cov)

    model = load_trained_model()

    # Darts/TFT requires future covariates to extend at least to output_chunk_length
    required_horizon = max(n_hours, OUTPUT_CHUNK)

    start_time = future_cov.start_time()
    end_time = target.end_time() + pd.Timedelta(hours=required_horizon)

    full_times = pd.date_range(
        start=start_time,
        end=end_time,
        freq="1h",
    )

    doy = full_times.dayofyear.values.astype("float32")
    hour = (full_times.hour + full_times.minute / 60.0).astype("float32")

    full_future_vals = np.column_stack([
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(4 * np.pi * doy / 365.25),
        np.cos(4 * np.pi * doy / 365.25),
    ]).astype("float32")

    future_cov_full = TimeSeries.from_times_and_values(
        full_times,
        full_future_vals,
        columns=FUTURE_COV_COLS,
    )
    future_cov_sc_full = scaler_future.transform(future_cov_full)

    print("target end:", target_sc.end_time())
    print("future cov end:", future_cov_sc_full.end_time())

    pred_sc = model.predict(
        n=n_hours,
        series=target_sc,
        past_covariates=past_cov_sc,
        future_covariates=future_cov_sc_full,
    )

    pred = scaler_target.inverse_transform(pred_sc)
    pdf = pred.to_dataframe()

    kp_pred = pdf["Kp"].clip(0, 9).values
    results = pd.DataFrame(
        {
            "Date": pdf.index,
            "X_predicted": pdf["X"].values,
            "Kp_predicted": kp_pred,
            "aurora_probability": kp_to_aurora_probability(kp_pred),
            "min_latitude_deg": kp_to_aurora_latitude(kp_pred),
        }
    )

    if current_speed and current_speed > 0:
        tt = travel_time_hours(current_speed)
        results["travel_time_h"] = tt
        scale = "forte" if kp_pred.max() >= 5 else "modérée" if kp_pred.max() >= 3 else "faible"
        print(f"\nVitesse vent solaire : {current_speed:.0f} km/s → arrivée estimée : {tt:.1f} h")
        print(f"Perturbation prévue : {scale} (Kp max = {kp_pred.max():.1f})")

    print("\n=== PRÉVISIONS ===")
    print(results.to_string(index=False))
    return results


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate():
    import matplotlib.pyplot as plt

    df = load_and_prepare()
    target, past_cov, future_cov = build_timeseries(df)

    scaler_target, scaler_past, scaler_future = _load_scalers()
    target_sc = scaler_target.transform(target)
    past_cov_sc = scaler_past.transform(past_cov)
    future_cov_sc = scaler_future.transform(future_cov)

    model = load_trained_model()

    print("Running historical forecasts (test set: last 10 %) …")
    bt_sc = model.historical_forecasts(
        series=target_sc,
        past_covariates=past_cov_sc,
        future_covariates=future_cov_sc,
        start=0.9,
        forecast_horizon=1,
        retrain=False,
        verbose=True,
    )

    if isinstance(bt_sc, list):
        bt_sc = concatenate(bt_sc)

    bt = scaler_target.inverse_transform(bt_sc)

    actual = target.slice(bt.start_time(), bt.end_time())

    for col in TARGET_COLS:
        act_vals = actual[col].univariate_values()
        pred_vals = bt[col].univariate_values()
        n = min(len(act_vals), len(pred_vals))
        act_vals, pred_vals = act_vals[:n], pred_vals[:n]

        mae = np.mean(np.abs(pred_vals - act_vals))
        rmse = np.sqrt(np.mean((pred_vals - act_vals) ** 2))

        if len(pred_vals) < 2 or np.std(pred_vals) == 0 or np.std(act_vals) == 0:
            r = float("nan")
        else:
            r = np.corrcoef(pred_vals, act_vals)[0, 1]

        print(f"\n[{col}] MAE = {mae:.4f}   RMSE = {rmse:.4f}   Pearson r = {r:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    actual["X"].plot(ax=axes[0], label="Mesuré (X Kiruna)", lw=0.6)
    bt["X"].plot(ax=axes[0], label="Prédit (X Kiruna)", lw=0.8)
    axes[0].set_title("X magnétique — Kiruna (modèle vs réalité)")
    axes[0].set_ylabel("nT")
    axes[0].legend()

    actual["Kp"].plot(ax=axes[1], label="Kp mesuré", lw=0.6)
    bt["Kp"].plot(ax=axes[1], label="Kp prédit", lw=0.8)
    axes[1].set_title("Indice Kp — modèle vs réalité")
    axes[1].set_ylabel("Kp")
    axes[1].legend()

    kp_pred_vals = bt["Kp"].univariate_values()
    prob = kp_to_aurora_probability(kp_pred_vals)
    lat = kp_to_aurora_latitude(kp_pred_vals)
    bt_times = bt.time_index

    axes[2].fill_between(bt_times, 0, prob, alpha=0.4, label="Probabilité aurore")
    ax2r = axes[2].twinx()
    ax2r.plot(bt_times, lat, color="red", lw=0.7, label="Latitude min. aurore (°)")
    ax2r.set_ylabel("Latitude (°)")
    axes[2].set_title("Probabilité d'aurore et latitude limite")
    axes[2].set_ylabel("Probabilité")
    axes[2].legend(loc="upper left")
    ax2r.legend(loc="upper right")

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "validation.png"
    plt.savefig(out, dpi=150)
    print(f"\nFigure sauvegardée → {out}")
    plt.show()


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Aurora Borealis Predictive Model")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Full training from scratch")
    sub.add_parser("retrain", help="Continue training on updated data")

    p_fc = sub.add_parser("forecast", help="Produce a forecast")
    p_fc.add_argument("--hours", type=int, default=5, help="Forecast horizon in hours (> 0)")
    p_fc.add_argument("--speed", type=float, default=None, help="Current solar wind speed (km/s)")

    sub.add_parser("validate", help="Backtest + metrics + visualization")

    args = parser.parse_args()

    if args.command == "train":
        train_model(retrain=False)
    elif args.command == "retrain":
        train_model(retrain=True)
    elif args.command == "forecast":
        forecast(n_hours=args.hours, current_speed=args.speed)
    elif args.command == "validate":
        validate()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()