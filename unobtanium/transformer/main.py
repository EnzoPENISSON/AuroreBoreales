import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from darts import TimeSeries
from darts.models import TransformerModel
from darts.dataprocessing.transformers import Scaler


# =========================================================
# 1. CONFIG
# =========================================================
KIRUNA_PATH = "../../data/mag-kiruna-compiled/smooth.csv"
SOLAR_PATH = "../../data/solarwinds-ace-compiled/smooth-solarwinds.csv"

DATE_COL = "Date"
RAW_TARGET_COL = "X"
TARGET_COL = "X_delta"
COV_COLS = ["Speed", "Density", "Bt", "Bz"]

RESAMPLE_FREQ = "1h"

# Pour aller vite
USE_ONLY_LAST_N_ROWS = None
TRAIN_RATIO = 0.80

# Baseline Kiruna
BASELINE_X = 10500.0

# Modèle
INPUT_CHUNK_LENGTH = 24
OUTPUT_CHUNK_LENGTH = 1
N_EPOCHS = 10
BATCH_SIZE = 64
RANDOM_STATE = 42

# Évaluation rolling forecast limitée
MAX_TEST_STEPS = 200

# Mets "cpu" si MPS pose problème
ACCELERATOR = "mps"
DEVICES = 1


# =========================================================
# 2. OUTILS
# =========================================================
def safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df[DATE_COL]

    df["hour"] = dt.dt.hour
    df["dayofyear"] = dt.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    return df


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Bz_neg"] = np.minimum(df["Bz"], 0)
    df["pressure_proxy"] = df["Density"] * (df["Speed"] ** 2)

    for col in ["Speed", "Density", "Bt", "Bz", "Bz_neg", "pressure_proxy", "X_delta"]:
        df[f"{col}_roll3"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_roll6"] = df[col].rolling(6, min_periods=1).mean()
        df[f"{col}_roll12"] = df[col].rolling(12, min_periods=1).mean()

    return df


def estimate_delay_hours_from_speed(speed_kms: float) -> float:
    if pd.isna(speed_kms) or speed_kms <= 0:
        return np.nan
    distance_km = 1_500_000
    return distance_km / speed_kms / 3600.0


def add_delay_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delay_hours"] = df["Speed"].apply(estimate_delay_hours_from_speed)
    return df


def classify_storm_from_delta(delta_value: float):
    ax = abs(float(delta_value))

    if ax < 50:
        return 0.05, "faible", "hautes latitudes (> 65°)"
    elif ax < 100:
        return 0.20, "modérée", "60°–65°"
    elif ax < 200:
        return 0.50, "forte", "55°–60°"
    elif ax < 400:
        return 0.75, "très forte", "50°–55°"
    else:
        return 0.90, "extrême", "< 50° possible"


def ts_from_df(dataframe: pd.DataFrame, time_col: str, value_cols, freq: str = "1h") -> TimeSeries:
    df_local = dataframe.copy()
    df_local = df_local.sort_values(time_col).drop_duplicates(subset=[time_col])

    if isinstance(value_cols, str):
        value_cols = [value_cols]

    return TimeSeries.from_dataframe(
        df_local,
        time_col=time_col,
        value_cols=value_cols,
        fill_missing_dates=True,
        freq=freq
    )


def ts_to_float32(ts: TimeSeries) -> TimeSeries:
    return ts.astype(np.float32)


def build_continuous_hourly_df(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL])
    df = (
        df.set_index(DATE_COL)
          .asfreq(freq)
          .interpolate(method="time")
          .ffill()
          .bfill()
          .reset_index()
    )
    return df


def rolling_forecast(model, target_series, past_covariates, start_idx, horizon, max_steps=200):
    pred_times = []
    pred_values = []

    end_idx = len(target_series) - horizon + 1
    end_idx = min(end_idx, start_idx + max_steps)

    if start_idx >= end_idx:
        raise ValueError(
            f"Zone de test trop courte pour horizon={horizon}. "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )

    for i in range(start_idx, end_idx):
        hist_target = target_series[:i]
        hist_covs = past_covariates[:i + horizon]

        pred = model.predict(
            n=horizon,
            series=hist_target,
            past_covariates=hist_covs,
            verbose=False,
            show_warnings=False
        )

        pred_array = pred.values()

        if pred_array.ndim == 2:
            value = pred_array[0, 0]
        elif pred_array.ndim == 3:
            value = pred_array[0, 0, 0]
        else:
            raise ValueError(f"Shape inattendue pour pred.values(): {pred_array.shape}")

        pred_times.append(pred.time_index[0])
        pred_values.append(np.float32(value))

    pred_df = pd.DataFrame({
        DATE_COL: pred_times,
        "pred_X_delta": pred_values
    })
    pred_df["pred_X_delta"] = pred_df["pred_X_delta"].astype("float32")
    return pred_df


# =========================================================
# 3. CHARGEMENT
# =========================================================
df1 = pd.read_csv(KIRUNA_PATH, sep=";")
df2 = pd.read_csv(SOLAR_PATH, sep=";")

df1[DATE_COL] = pd.to_datetime(df1[DATE_COL], format="%Y%m%d%H%M%S", errors="coerce")
df2[DATE_COL] = pd.to_datetime(df2[DATE_COL], format="%Y%m%d%H%M%S", errors="coerce")

df1 = df1[[DATE_COL, RAW_TARGET_COL]].dropna()
df2 = df2[[DATE_COL] + COV_COLS].dropna()

df = pd.merge(df1, df2, on=DATE_COL, how="inner").sort_values(DATE_COL)

df[TARGET_COL] = df[RAW_TARGET_COL] - BASELINE_X

df = (
    df.set_index(DATE_COL)
      .resample(RESAMPLE_FREQ)
      .mean()
      .interpolate(method="time")
      .ffill()
      .bfill()
      .reset_index()
)

if USE_ONLY_LAST_N_ROWS is not None and len(df) > USE_ONLY_LAST_N_ROWS:
    df = df.iloc[-USE_ONLY_LAST_N_ROWS:].copy()

df = add_time_features(df)
df = add_physics_features(df)
df = add_delay_feature(df)

df = df.dropna().reset_index(drop=True)
df = build_continuous_hourly_df(df, RESAMPLE_FREQ)

# recalcul après remise en grille
df[TARGET_COL] = df[RAW_TARGET_COL] - BASELINE_X
df = add_time_features(df)
df = add_physics_features(df)
df = add_delay_feature(df)

df = df.dropna().reset_index(drop=True)

float_cols = [col for col in df.columns if col != DATE_COL]
df[float_cols] = df[float_cols].astype("float32")

print("Aperçu des données :")
print(df.head())
print("\nColonnes :")
print(df.columns.tolist())
print(f"\nNombre de lignes retenues : {len(df)}")


# =========================================================
# 4. SPLIT TRAIN / TEST
# =========================================================
split_idx = int(len(df) * TRAIN_RATIO)
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

print(f"\nTrain size: {len(df_train)}")
print(f"Test size : {len(df_test)}")

feature_cols = [
    "Speed", "Density", "Bt", "Bz",
    "Bz_neg", "pressure_proxy", "delay_hours",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "X_delta_roll3", "X_delta_roll6", "X_delta_roll12",
]

missing_cols = [col for col in [TARGET_COL] + feature_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Colonnes manquantes après preprocessing : {missing_cols}")

target_full = ts_to_float32(ts_from_df(df, DATE_COL, TARGET_COL, RESAMPLE_FREQ))
cov_full = ts_to_float32(ts_from_df(df, DATE_COL, feature_cols, RESAMPLE_FREQ))

target_train = ts_to_float32(ts_from_df(df_train, DATE_COL, TARGET_COL, RESAMPLE_FREQ))
cov_train = ts_to_float32(ts_from_df(df_train, DATE_COL, feature_cols, RESAMPLE_FREQ))

scaler_target = Scaler()
scaler_cov = Scaler()

target_train_scaled = ts_to_float32(scaler_target.fit_transform(target_train))
cov_train_scaled = ts_to_float32(scaler_cov.fit_transform(cov_train))

target_full_scaled = ts_to_float32(scaler_target.transform(target_full))
cov_full_scaled = ts_to_float32(scaler_cov.transform(cov_full))


# =========================================================
# 5. MODEL
# =========================================================
model = TransformerModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    model_name="kiruna_transformer_small_v2",
    random_state=RANDOM_STATE,
    pl_trainer_kwargs={
        "accelerator": ACCELERATOR,
        "devices": DEVICES,
        "enable_progress_bar": True,
        "logger": False,
        "enable_model_summary": False,
    },
)

print("\nEntraînement...")
model.fit(
    series=target_train_scaled,
    past_covariates=cov_train_scaled,
    verbose=True
)


# =========================================================
# 6. PRÉDICTION SUR LE TEST
# =========================================================
start_test_idx = split_idx

print(f"\nRolling forecast sur {MAX_TEST_STEPS} pas max...")
pred_df_scaled = rolling_forecast(
    model=model,
    target_series=target_full_scaled,
    past_covariates=cov_full_scaled,
    start_idx=start_test_idx,
    horizon=OUTPUT_CHUNK_LENGTH,
    max_steps=MAX_TEST_STEPS
)

pred_series_scaled = ts_to_float32(ts_from_df(pred_df_scaled, DATE_COL, "pred_X_delta", RESAMPLE_FREQ))
pred_series_real = ts_to_float32(scaler_target.inverse_transform(pred_series_scaled))

pred_df = pred_series_real.to_dataframe().reset_index()
pred_df.columns = [DATE_COL, "pred_X_delta"]
pred_df["pred_X_delta"] = pred_df["pred_X_delta"].astype("float32")

truth_df = df[[DATE_COL, TARGET_COL]].copy()
truth_df.columns = [DATE_COL, "true_X_delta"]

results = pd.merge(pred_df, truth_df, on=DATE_COL, how="inner").dropna()

results["pred_X_real"] = results["pred_X_delta"] + BASELINE_X
results["true_X_real"] = results["true_X_delta"] + BASELINE_X

mae = mean_absolute_error(results["true_X_delta"], results["pred_X_delta"])
rmse = safe_rmse(results["true_X_delta"], results["pred_X_delta"])
r2 = r2_score(results["true_X_delta"], results["pred_X_delta"])

# baseline naïve : prédiction = dernière valeur connue
results["naive_pred_delta"] = df[TARGET_COL].shift(1).reindex(results.index).values
naive_valid = results.dropna(subset=["naive_pred_delta"]).copy()

naive_mae = mean_absolute_error(naive_valid["true_X_delta"], naive_valid["naive_pred_delta"])
naive_rmse = safe_rmse(naive_valid["true_X_delta"], naive_valid["naive_pred_delta"])
naive_r2 = r2_score(naive_valid["true_X_delta"], naive_valid["naive_pred_delta"])

print("\n========================")
print("ÉVALUATION SUR LE TEST")
print("========================")
print(f"Nb points évalués : {len(results)}")
print(f"Transformer MAE  : {mae:.4f}")
print(f"Transformer RMSE : {rmse:.4f}")
print(f"Transformer R2   : {r2:.4f}")
print()
print(f"Naive MAE        : {naive_mae:.4f}")
print(f"Naive RMSE       : {naive_rmse:.4f}")
print(f"Naive R2         : {naive_r2:.4f}")

print("\nExemples de comparaison :")
print(results[[
    DATE_COL,
    "true_X_real", "pred_X_real",
    "true_X_delta", "pred_X_delta"
]].head(15))


# =========================================================
# 7. INTERPRÉTATION MÉTIER
# =========================================================
storm_prob = []
storm_scale = []
storm_lat_band = []

for delta in results["pred_X_delta"].values:
    p, s, lat = classify_storm_from_delta(delta)
    storm_prob.append(p)
    storm_scale.append(s)
    storm_lat_band.append(lat)

results["storm_probability"] = storm_prob
results["storm_scale"] = storm_scale
results["probable_latitude_band"] = storm_lat_band

print("\n========================")
print("SORTIE MÉTIER")
print("========================")
print(results[[
    DATE_COL,
    "true_X_real", "pred_X_real",
    "true_X_delta", "pred_X_delta",
    "storm_probability", "storm_scale", "probable_latitude_band"
]].head(20))


# =========================================================
# 8. PRÉVISION FUTURE
# =========================================================
n_forecast = 6

last_time = cov_full_scaled.end_time()

future_times = pd.date_range(
    start=last_time + pd.Timedelta(hours=1),
    periods=n_forecast,
    freq=RESAMPLE_FREQ
)

last_cov_values = cov_full_scaled[-1:].values().astype(np.float32)
future_cov_values = np.repeat(last_cov_values, repeats=n_forecast, axis=0).astype(np.float32)

future_cov = TimeSeries.from_times_and_values(
    times=future_times,
    values=future_cov_values
)
future_cov = ts_to_float32(future_cov)

cov_extended = ts_to_float32(cov_full_scaled.append(future_cov))

future_pred_scaled = model.predict(
    n=n_forecast,
    series=target_full_scaled,
    past_covariates=cov_extended,
    verbose=False,
    show_warnings=False
)

future_pred = ts_to_float32(scaler_target.inverse_transform(future_pred_scaled))

future_pred_df = future_pred.to_dataframe().reset_index()
future_pred_df.columns = [DATE_COL, "pred_X_delta"]
future_pred_df["pred_X_delta"] = future_pred_df["pred_X_delta"].astype("float32")
future_pred_df["pred_X_real"] = future_pred_df["pred_X_delta"] + BASELINE_X

storm_prob = []
storm_scale = []
storm_lat_band = []

for delta in future_pred_df["pred_X_delta"].values:
    p, s, lat = classify_storm_from_delta(delta)
    storm_prob.append(p)
    storm_scale.append(s)
    storm_lat_band.append(lat)

future_pred_df["storm_probability"] = storm_prob
future_pred_df["storm_scale"] = storm_scale
future_pred_df["probable_latitude_band"] = storm_lat_band

print("\n========================")
print("PRÉVISIONS FUTURES")
print("========================")
print(future_pred_df[[
    DATE_COL,
    "pred_X_real", "pred_X_delta",
    "storm_probability", "storm_scale", "probable_latitude_band"
]])


# =========================================================
# 9. SAUVEGARDE
# =========================================================
results.to_csv("comparison_test_predictions.csv", index=False, sep=";")
future_pred_df.to_csv("future_predictions.csv", index=False, sep=";")

print("\nFichiers sauvegardés :")
print("- comparison_test_predictions.csv")
print("- future_predictions.csv")