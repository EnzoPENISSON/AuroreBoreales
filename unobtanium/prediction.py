import argparse
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


def main():
	parser = argparse.ArgumentParser(description="Predict Kp from merged.csv using a trained checkpoint.")
	parser.add_argument(
		"-i",
		"--input",
		type=Path,
		default=Path("data/json/merged.csv"),
		help="Input CSV path (default: data/json/merged.csv)",
	)
	parser.add_argument(
		"-c",
		"--checkpoint",
		type=Path,
		default=Path("model/checkpoint_epoch_4000.pt"),
		help="Checkpoint path (default: model/checkpoint_epoch_4000.pt)",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=Path("data/json/merged_kp_predictions.csv"),
		help="Output CSV path (default: data/json/merged_kp_predictions.csv)",
	)
	parser.add_argument(
		"--no-resample",
		action="store_true",
		help="Disable 15-minute resampling before prediction.",
	)

	args = parser.parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input CSV not found: {args.input}")
	if not args.checkpoint.exists():
		raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	input_df = load_input_data(args.input, resample_15min=not args.no_resample)
	model = load_model(args.checkpoint, device)
	predictions_df = predict_kp(model, input_df, device)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	predictions_df.to_csv(args.output, index=False)

	print(f"Predictions created: {args.output}")
	print(f"Rows predicted: {len(predictions_df)}")
	print(predictions_df.head(5))


if __name__ == "__main__":
	main()
