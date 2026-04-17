"""
Aurora Borealis Predictive Model
================================
Predicts future geomagnetic perturbations at Kiruna from ACE/DSCOVR solar wind data,
and derives operational auroral indicators from the predicted local disturbance.

Core idea:
  - Train the main model on detrended Kiruna perturbation + solar wind + calendar features
  - Predict future local magnetic disturbance at Kiruna
  - Estimate Kp with a separate calibration model trained only when historical Kp is available

This design allows continued retraining of the main forecaster with new Kiruna + DSCOVR
measurements even when new official Kp labels are not yet available.

Outputs:
  - Predicted local magnetic perturbation at Kiruna
  - Estimated Kp index
  - Aurora probability at mid-latitudes
  - Estimated equatorward auroral boundary (latitude)

Usage:
    python predict.py train
    python predict.py retrain
    python predict.py forecast --hours 10 --speed 600
    python predict.py validate
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.strategies import DDPStrategy
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts.utils.likelihood_models.base import LikelihoodType

torch.serialization.add_safe_globals([QuantileRegression])
torch.serialization.add_safe_globals([QuantileRegression, LikelihoodType])
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# ===========================================================================
# CONFIGURATION
# ===========================================================================
DATA_DIR = Path("../data")
MODEL_DIR = Path(__file__).resolve().parent / "models"
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "hourly_dataset.parquet"

INPUT_CHUNK = 168
OUTPUT_CHUNK = 24
ROLLING_BASELINE_HOURS = 24

BATCH_SIZE = 64
N_EPOCHS = 30
HIDDEN_SIZE = 128
LSTM_LAYERS = 2
ATTENTION_HEADS = 4
DROPOUT = 0.1
LR = 1e-3

# Early stopping
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4

L1_DISTANCE_KM = 1_500_000

TARGET_COL = "X_pert"
TARGET_COLS = [TARGET_COL]
PAST_COV_COLS = [
    "X_pert",
    "Speed",
    "Bz",
    "Density",
    "Bt",
    "Bz_south",
    "Pdyn",
    "coupling",
]
FUTURE_COV_COLS = ["sin_doy", "cos_doy", "sin_hour", "cos_hour", "sin_rm", "cos_rm"]
KP_FEATURE_COLS = ["X_pert_abs", "coupling", "Bz_south_abs", "Pdyn", "Speed", "Bt"]


# ===========================================================================
# DATA LOADING & FEATURE ENGINEERING
# ===========================================================================

def load_and_prepare(data_dir: Path = DATA_DIR, use_cache: bool = True) -> pd.DataFrame:
    """Load all data sources, merge, resample to 1 h, add engineered features."""
    expected_cols = {
        "Date",
        "X",
        "X_pert",
        "X_pert_abs",
        "Speed",
        "Density",
        "Bt",
        "Bz",
        "Bz_south",
        "Bz_south_abs",
        "Pdyn",
        "coupling",
        "sin_doy",
        "cos_doy",
        "sin_hour",
        "cos_hour",
        "sin_rm",
        "cos_rm",
    }

    if use_cache and CACHE_PATH.exists():
        print(f"Loading cached dataset from {CACHE_PATH}")
        cached_df = pd.read_parquet(CACHE_PATH)
        if expected_cols.issubset(set(cached_df.columns)):
            return cached_df
        print("Cache schema is outdated; rebuilding dataset cache …")

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

    df = kiruna.join(sw, how="inner").join(kp, how="left")
    df = df.interpolate(method="linear")
    df = df.dropna(subset=["X", "Speed", "Density", "Bt", "Bz"])

    baseline = df["X"].rolling(ROLLING_BASELINE_HOURS, min_periods=1).mean()
    df["X_pert"] = (df["X"] - baseline).astype("float32")
    df["X_pert_abs"] = df["X_pert"].abs().astype("float32")

    doy = df.index.dayofyear.values.astype("float32")
    hour = (df.index.hour + df.index.minute / 60.0).astype("float32")

    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    df["sin_rm"] = np.sin(4 * np.pi * doy / 365.25).astype("float32")
    df["cos_rm"] = np.cos(4 * np.pi * doy / 365.25).astype("float32")

    df["Bz_south"] = df["Bz"].clip(upper=0).astype("float32")
    df["Bz_south_abs"] = np.abs(df["Bz_south"]).astype("float32")
    df["Pdyn"] = (df["Density"] * df["Speed"] ** 2 * 1.6726e-6).astype("float32")

    by2 = (df["Bt"] ** 2 - df["Bz"] ** 2).clip(lower=0)
    by = np.sqrt(by2).astype("float32")
    theta_c = np.arctan2(by, -df["Bz"]).astype("float32")

    df["coupling"] = (
        df["Speed"].abs() ** (4 / 3)
        * df["Bt"].abs() ** (2 / 3)
        * np.abs(np.sin(theta_c / 2)) ** (8 / 3)
    ).astype("float32")

    keep_cols = [
        "X",
        "X_pert",
        "X_pert_abs",
        "Speed",
        "Density",
        "Bt",
        "Bz",
        "Bz_south",
        "Bz_south_abs",
        "Pdyn",
        "coupling",
        "sin_doy",
        "cos_doy",
        "sin_hour",
        "cos_hour",
        "sin_rm",
        "cos_rm",
    ]
    if "Kp" in df.columns:
        keep_cols.append("Kp")

    df = df[keep_cols].astype("float32").reset_index()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    print(f"Cached {len(df)} hourly rows → {CACHE_PATH}")
    return df


def build_timeseries(df: pd.DataFrame):
    target = TimeSeries.from_dataframe(df, time_col="Date", value_cols=[TARGET_COL])
    past_cov = TimeSeries.from_dataframe(df, time_col="Date", value_cols=PAST_COV_COLS)
    future_cov = TimeSeries.from_dataframe(df, time_col="Date", value_cols=FUTURE_COV_COLS)
    return target, past_cov, future_cov


# ===========================================================================
# TRAINING CALLBACKS
# ===========================================================================

class EpochStatsPrinter(Callback):
    """Print training/validation stats after each validation epoch."""

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch + 1

        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")

        def _fmt(x):
            try:
                return f"{float(x):.6f}"
            except Exception:
                return "n/a"

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={_fmt(train_loss)} | "
            f"val_loss={_fmt(val_loss)}"
        )


# ===========================================================================
# MODEL
# ===========================================================================

def _get_accelerator_config():
    """Choose a safe accelerator config depending on available hardware."""
    if torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1}
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}


def create_model(n_epochs: int = N_EPOCHS, force_reset: bool = True) -> TFTModel:
    accelerator_cfg = _get_accelerator_config()

    # Read environment variables set by the PyTorch launcher
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # LOCAL_RANK identifies the specific process on the current machine
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        # We MUST use 'gloo' because 'nccl' does not support CPU or MPS
        ddp_strategy = DDPStrategy(process_group_backend="gloo")
        ddp_cfg = {
            "num_nodes": world_size,
            "strategy": ddp_strategy,
            "sync_batchnorm": False, # Important to disable for mixed hardware
        }
    else:
        ddp_cfg = {}

    # ---------------------------------------------------------
    # Logging and Callbacks: Only initialize on Rank 0
    # ---------------------------------------------------------
    loggers = []
    if local_rank == 0:
        loggers.append(CSVLogger(save_dir=str(MODEL_DIR / "logs"), name="aurora_tft"))

    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        mode="min",
        verbose=(local_rank == 0), # Only print on Master
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    callbacks = [early_stopper, lr_monitor]
    if local_rank == 0:
        callbacks.append(EpochStatsPrinter())

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
            **accelerator_cfg,
            **ddp_cfg,
            "callbacks": callbacks,
            "logger": loggers if loggers else False,
            "log_every_n_steps": 10,
            "enable_progress_bar": (local_rank == 0), # Clean terminal output
        },
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


def _save_kp_calibrator(model) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "kp_calibrator.pkl", "wb") as f:
        pickle.dump(model, f)


def _load_kp_calibrator():
    with open(MODEL_DIR / "kp_calibrator.pkl", "rb") as f:
        return pickle.load(f)


def build_kp_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[KP_FEATURE_COLS].astype("float32").copy()


def fit_kp_calibrator(df_train_kp: pd.DataFrame):
    df_kp = df_train_kp.dropna(subset=["Kp"]).copy()
    if len(df_kp) < 200:
        raise ValueError("Not enough rows with Kp to train the Kp calibrator.")

    X_kp = build_kp_features(df_kp)
    y_kp = df_kp["Kp"].astype("float32")

    kp_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    kp_model.fit(X_kp, y_kp)
    _save_kp_calibrator(kp_model)
    return kp_model


def estimate_kp_from_features(df_features: pd.DataFrame) -> np.ndarray:
    kp_model = _load_kp_calibrator()
    kp_pred = kp_model.predict(build_kp_features(df_features))
    return np.clip(kp_pred.astype("float32"), 0.0, 9.0)


def estimate_kp_from_disturbance(x_pred: np.ndarray, coupling: np.ndarray, bz_south: np.ndarray) -> np.ndarray:
    """Fallback proxy used only if the Kp calibrator is unavailable."""
    x_mag = np.abs(np.asarray(x_pred, dtype="float32"))
    coupling = np.maximum(np.asarray(coupling, dtype="float32"), 0.0)
    bz_south = np.abs(np.minimum(np.asarray(bz_south, dtype="float32"), 0.0))

    x_norm = np.log1p(x_mag) / np.log(1.0 + 800.0)
    coupling_norm = np.log1p(coupling) / np.log(1.0 + 5000.0)
    bz_norm = np.clip(bz_south / 20.0, 0.0, 1.0)

    kp = 9.0 * (0.55 * x_norm + 0.30 * coupling_norm + 0.15 * bz_norm)
    return np.clip(kp, 0.0, 9.0)


# ===========================================================================
# TRAINING
# ===========================================================================

# IMPORTANT:
# - The trainable neural model learns only future local perturbation at Kiruna.
# - Kp is not used as a target for this neural network.
# - A separate Kp calibration model is fitted only on historical periods where Kp exists.
# - New Kiruna + DSCOVR data can therefore still be used to continue training the main forecaster.

def train_model(retrain: bool = False):
    df = load_and_prepare(use_cache=True)
    target, past_cov, future_cov = build_timeseries(df)

    train_end_idx = int(len(df) * 0.8)
    val_end_idx = int(len(df) * 0.9)

    train_end_time = df["Date"].iloc[train_end_idx]
    val_end_time = df["Date"].iloc[val_end_idx]

    df_train = df[df["Date"] < train_end_time].copy()
    df_val = df[(df["Date"] >= train_end_time) & (df["Date"] < val_end_time)].copy()
    df_test = df[df["Date"] >= val_end_time].copy()

    target_train, target_tmp = target.split_before(train_end_time)
    target_val, target_test = target_tmp.split_before(val_end_time)

    past_train, past_tmp = past_cov.split_before(train_end_time)
    past_val, past_test = past_tmp.split_before(val_end_time)

    future_train, future_tmp = future_cov.split_before(train_end_time)
    future_val, future_test = future_tmp.split_before(val_end_time)

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
            model.model_params["n_epochs"] = N_EPOCHS
        except Exception:
            print("No checkpoint found — starting from scratch.")
            model = create_model(force_reset=True)
    else:
        model = create_model(force_reset=True)

    print(
        f"Training samples: {len(ts_train)}, "
        f"Validation samples: {len(ts_val)}, "
        f"Test samples: {len(ts_test)} | "
        f"Target={TARGET_COL} | Inputs={PAST_COV_COLS + FUTURE_COV_COLS}"
    )

    # model = torch.nn.parallel.DistributedDataParallel(model)

    model.fit(
        ts_train,
        past_covariates=pc_train,
        future_covariates=fc_train,
        val_series=ts_val,
        val_past_covariates=pc_val,
        val_future_covariates=fc_val,
        verbose=True,
        load_best=True,
    )

    try:
        fit_kp_calibrator(df_train)
        print(f"Kp calibrator saved to {MODEL_DIR / 'kp_calibrator.pkl'} (trained on train split only)")
    except Exception as exc:
        print(f"Warning: unable to fit Kp calibrator: {exc}")

    print(f"Train rows (time split): {len(df_train)} | Val rows: {len(df_val)} | Test rows: {len(df_test)}")
    print(f"Model saved to {MODEL_DIR / 'aurora_tft'}")
    print(f"Logs saved to {MODEL_DIR / 'logs'}")
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

    df = load_and_prepare(use_cache=True)
    target, past_cov, future_cov = build_timeseries(df)

    scaler_target, scaler_past, scaler_future = _load_scalers()
    target_sc = scaler_target.transform(target)
    past_cov_sc = scaler_past.transform(past_cov)

    model = load_trained_model()

    required_horizon = max(n_hours, OUTPUT_CHUNK)
    start_time = future_cov.start_time()
    end_time = target.end_time() + pd.Timedelta(hours=required_horizon)

    full_times = pd.date_range(start=start_time, end=end_time, freq="1h")
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

    pred_sc = model.predict(
        n=n_hours,
        series=target_sc,
        past_covariates=past_cov_sc,
        future_covariates=future_cov_sc_full,
    )

    pred = scaler_target.inverse_transform(pred_sc)
    pdf = pred.to_dataframe()

    recent_past = past_cov.to_dataframe().iloc[-n_hours:].copy()
    if len(recent_past) < n_hours:
        recent_past = past_cov.to_dataframe().iloc[-1:].copy()
        recent_past = pd.concat([recent_past] * n_hours, ignore_index=False)
        recent_past = recent_past.iloc[:n_hours]

    recent_past = recent_past.reset_index(drop=True)
    x_pert_pred = pdf[TARGET_COL].values.astype("float32")

    features_for_kp = pd.DataFrame(
        {
            "X_pert_abs": np.abs(x_pert_pred),
            "coupling": recent_past["coupling"].values.astype("float32"),
            "Bz_south_abs": np.abs(recent_past["Bz_south"].values.astype("float32")),
            "Pdyn": recent_past["Pdyn"].values.astype("float32"),
            "Speed": recent_past["Speed"].values.astype("float32"),
            "Bt": recent_past["Bt"].values.astype("float32"),
        }
    )

    try:
        kp_pred = estimate_kp_from_features(features_for_kp)
    except Exception as exc:
        print(f"Warning: Kp calibrator unavailable, fallback to simple proxy. Reason: {exc}")
        kp_pred = estimate_kp_from_disturbance(
            x_pred=x_pert_pred,
            coupling=features_for_kp["coupling"].values,
            bz_south=-features_for_kp["Bz_south_abs"].values,
        )

    results = pd.DataFrame(
        {
            "Date": pdf.index,
            "X_pert_predicted": x_pert_pred,
            "Kp_estimated": kp_pred,
            "aurora_probability": kp_to_aurora_probability(kp_pred),
            "min_latitude_deg": kp_to_aurora_latitude(kp_pred),
        }
    )

    if current_speed and current_speed > 0:
        tt = travel_time_hours(current_speed)
        results["travel_time_h"] = tt
        scale = "forte" if kp_pred.max() >= 5 else "modérée" if kp_pred.max() >= 3 else "faible"
        print(f"\nVitesse vent solaire : {current_speed:.0f} km/s → arrivée estimée : {tt:.1f} h")
        print(f"Perturbation prévue : {scale} (Kp estimé max = {kp_pred.max():.1f})")

    print("\n=== PRÉVISIONS ===")
    print(results.to_string(index=False))
    return results


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate():
    import matplotlib.pyplot as plt

    df = load_and_prepare(use_cache=True)
    df_with_kp = df.dropna(subset=["Kp"]).reset_index(drop=True)
    if len(df_with_kp) < 300:
        raise ValueError("Not enough rows with Kp to validate the Kp calibration.")

    train_end_idx = int(len(df_with_kp) * 0.8)
    val_end_idx = int(len(df_with_kp) * 0.9)

    train_end_time = df_with_kp["Date"].iloc[train_end_idx]
    val_end_time = df_with_kp["Date"].iloc[val_end_idx]

    df_train_kp = df_with_kp[df_with_kp["Date"] < train_end_time].copy()
    df_test_kp = df_with_kp[df_with_kp["Date"] >= val_end_time].copy()

    if len(df_train_kp) < 200:
        raise ValueError("Not enough Kp rows in the train split to fit the calibrator.")
    if len(df_test_kp) < 50:
        raise ValueError("Not enough Kp rows in the test split to validate.")

    fit_kp_calibrator(df_train_kp)

    target, past_cov, future_cov = build_timeseries(df_with_kp)

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
        start=val_end_time,
        forecast_horizon=1,
        retrain=False,
        verbose=True,
    )

    if isinstance(bt_sc, list):
        bt_sc = concatenate(bt_sc)

    bt = scaler_target.inverse_transform(bt_sc)
    actual_target = target.slice(bt.start_time(), bt.end_time())

    df_eval = df_with_kp.set_index("Date")
    df_eval = df_eval.loc[bt.start_time(): bt.end_time()].copy()

    x_true = actual_target[TARGET_COL].univariate_values()
    x_pred = bt[TARGET_COL].univariate_values()
    n = min(len(x_true), len(x_pred), len(df_eval))

    x_true = x_true[:n]
    x_pred = x_pred[:n]
    df_eval = df_eval.iloc[:n]

    mae_x = mean_absolute_error(x_true, x_pred)
    rmse_x = np.sqrt(mean_squared_error(x_true, x_pred))
    if len(x_pred) < 2 or np.std(x_pred) == 0 or np.std(x_true) == 0:
        r_x = float("nan")
    else:
        r_x = float(np.corrcoef(x_pred, x_true)[0, 1])

    features_for_kp = df_eval.copy()
    features_for_kp["X_pert_abs"] = np.abs(x_pred).astype("float32")

    try:
        kp_est = estimate_kp_from_features(features_for_kp)
    except Exception as exc:
        print(f"Warning: Kp calibrator unavailable during validation, fallback proxy used. Reason: {exc}")
        kp_est = estimate_kp_from_disturbance(
            x_pred=x_pred,
            coupling=df_eval["coupling"].values.astype("float32"),
            bz_south=df_eval["Bz_south"].values.astype("float32"),
        )

    kp_true = df_eval["Kp"].values.astype("float32")

    mae_kp = mean_absolute_error(kp_true, kp_est)
    rmse_kp = np.sqrt(mean_squared_error(kp_true, kp_est))
    if len(kp_est) < 2 or np.std(kp_est) == 0 or np.std(kp_true) == 0:
        r_kp = float("nan")
    else:
        r_kp = float(np.corrcoef(kp_est, kp_true)[0, 1])

    print(f"\n[{TARGET_COL}] MAE = {mae_x:.4f}   RMSE = {rmse_x:.4f}   Pearson r = {r_x:.4f}")
    print(f"\n[Kp calibré - vrai hors échantillon] MAE = {mae_kp:.4f}   RMSE = {rmse_kp:.4f}   Pearson r = {r_kp:.4f}")
    print(f"Validation temporelle: calibrateur Kp entraîné avant {train_end_time}, évaluation TFT/Kp à partir de {val_end_time}")

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    bt_times = bt.time_index[:n]

    axes[0].plot(bt_times, x_true, label="X_pert mesuré (Kiruna)", lw=0.6)
    axes[0].plot(bt_times, x_pred, label="X_pert prédit (Kiruna)", lw=0.8)
    axes[0].set_title("Perturbation magnétique locale — Kiruna (modèle vs réalité)")
    axes[0].set_ylabel("nT")
    axes[0].legend()

    axes[1].plot(bt_times, kp_true, label="Kp mesuré", lw=0.6)
    axes[1].plot(bt_times, kp_est, label="Kp estimé calibré", lw=0.8)
    axes[1].set_title("Indice Kp — validation hors échantillon")
    axes[1].set_ylabel("Kp")
    axes[1].legend()

    prob = kp_to_aurora_probability(kp_est)
    lat = kp_to_aurora_latitude(kp_est)

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
    print("in file")
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