from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

class SolarWindLSTM(nn.Module):
	def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=10):
		super().__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		lstm_out, _ = self.lstm(x)
		out = self.fc(lstm_out[:, -1, :])
		return out


def load_input_data(csv_path, resample_15min=True):
	df = pd.read_csv(csv_path)

	required_columns = ["time_tag", "speed", "density", "bt", "bz_gsm"]
	missing = [column for column in required_columns if column not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in input CSV: {missing}")

	df = df[required_columns].copy()
	df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
	for column in ["speed", "density", "bt", "bz_gsm"]:
		df[column] = pd.to_numeric(df[column], errors="coerce")

	df = df.dropna(subset=required_columns)

	# Normalisation selon les valeurs min/max utilisées durant l'entraînement
	df["speed"] = (df["speed"] - 0.0) / (1209.40 - 0.0)
	df["density"] = (df["density"] - 0.0) / (199.70 - 0.0)
	df["bt"] = (df["bt"] - 0.0) / (74.66 - 0.0)
	df["bz_gsm"] = (df["bz_gsm"] - (-49.85)) / (49.03 - (-49.85))

	if resample_15min:
		df = (
			df.set_index("time_tag")
			.resample("15min")
			.mean(numeric_only=True)
			.dropna()
			.reset_index()
		)

	if df.empty:
		raise ValueError("No valid rows left after cleaning/resampling input data.")

	return df


def load_model(checkpoint_path, device):
	model = SolarWindLSTM()
	checkpoint = torch.load(checkpoint_path, map_location=device)

	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
	elif isinstance(checkpoint, dict):
		state_dict = checkpoint
	else:
		raise ValueError("Unsupported checkpoint format.")

	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()
	return model


def predict_kp(model, input_df, device):
	features = input_df[["speed", "density", "bt", "bz_gsm"]].to_numpy(dtype="float32")
	tensor_x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(1)

	with torch.no_grad():
		logits = model(tensor_x)
		probabilities = torch.softmax(logits, dim=1)
		predicted_classes = torch.argmax(probabilities, dim=1)
		confidence = torch.max(probabilities, dim=1).values

	result = input_df[["time_tag"]].copy()
	result["kp_pred"] = predicted_classes.cpu().numpy().astype(int)
	result["confidence"] = confidence.cpu().numpy()
	return result

import sys

if __name__ == "__main__":
	checkpoints_dir = Path("model")
	
	if len(sys.argv) > 1:
		input_files = [Path(f) for f in sys.argv[1:] if Path(f).suffix == '.csv']
	else:
		input_files = [
			Path("data/json/merged-storm.csv"),
			Path("data/json/merged.csv"),  # adapte si besoin
		]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoints = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pt"))

	for input_path in input_files:
		if not input_path.exists():
			print(f"File not found: {input_path}")
			continue

		print(f"\n=== DATASET: {input_path.name} ===")

		input_df = load_input_data(input_path, resample_15min=True)

		for checkpoint in checkpoints:
			print(f"\n--- Model: {checkpoint.name} ---")

			model = load_model(checkpoint, device)
			predictions_df = predict_kp(model, input_df, device)

			# 💾 nom de sortie propre
			output_path = Path("results") / f"{input_path.stem}_{checkpoint.stem}.csv"
			output_path.parent.mkdir(parents=True, exist_ok=True)

			predictions_df.to_csv(output_path, index=False)

			print(f"Saved: {output_path}")
			print(predictions_df.head(3))