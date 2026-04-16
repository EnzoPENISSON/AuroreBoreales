from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DATE_FORMAT = "%Y%m%d%H%M%S"
DEFAULT_INPUT_CANDIDATES = [
	Path("data/solarwinds-ace-compiled/smooth_solarwinds_filled.csv"),
	Path("data/solarwinds-ace-compiled/smooth-solarwinds_filled.csv"),
]
DEFAULT_KP_INPUT = Path("data/kp-compiled/smooth.csv")
DEFAULT_OUTPUT = Path("data/solarwinds-ace-compiled/smooth_solarwinds_15min_metrics.csv")


@dataclass
class WindowState:
	start: datetime
	bt_integral: float = 0.0
	bt_max: float | None = None
	bt_min: float | None = None
	density_integral: float = 0.0
	density_max: float | None = None
	density_min: float | None = None
	speed_integral: float = 0.0
	speed_max: float | None = None
	speed_min: float | None = None
	prev_bt_value: float | None = None
	prev_bt_time: datetime | None = None
	prev_density_value: float | None = None
	prev_density_time: datetime | None = None
	prev_speed_value: float | None = None
	prev_speed_time: datetime | None = None
	kp: float | None = None


def resolve_input_path(cli_input: str | None) -> Path:
	"""Résout le fichier d'entrée en privilégiant le chemin passé en argument."""
	if cli_input:
		input_path = Path(cli_input)
		if input_path.exists():
			return input_path
		raise FileNotFoundError(f"Fichier introuvable: {input_path}")

	for candidate in DEFAULT_INPUT_CANDIDATES:
		if candidate.exists():
			return candidate

	candidates = ", ".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
	raise FileNotFoundError(f"Aucun fichier d'entrée trouvé. Candidats testés: {candidates}")


def parse_optional_float(raw_value: str | None) -> float | None:
	"""Convertit une valeur texte en float ou None si invalide."""
	if raw_value is None:
		return None

	value = raw_value.strip()
	if value in {"", "null", "NULL"}:
		return None

	try:
		return float(value)
	except ValueError:
		return None


def floor_15min(timestamp: datetime) -> datetime:
	"""Ramène un horodatage au début de sa fenêtre 15 minutes."""
	minute = (timestamp.minute // 15) * 15
	return timestamp.replace(minute=minute, second=0, microsecond=0)


def write_window(writer: csv.DictWriter, state: WindowState, kp_by_window: dict[str, float | None]) -> None:
	"""Écrit l'état d'une fenêtre 15 minutes dans le CSV de sortie."""
	window_key = state.start.strftime(DATE_FORMAT)
	writer.writerow(
		{
			"Date": window_key,
			"Bt_integral_15m": state.bt_integral if state.prev_bt_time is not None else None,
			"Bt_max_15m": state.bt_max,
			"Bt_min_15m": state.bt_min,
			"Density_integral_15m": state.density_integral if state.prev_density_time is not None else None,
			"Density_max_15m": state.density_max,
			"Density_min_15m": state.density_min,
			"Speed_integral_15m": state.speed_integral if state.prev_speed_time is not None else None,
			"Speed_max_15m": state.speed_max,
			"Speed_min_15m": state.speed_min,
			"Kp": kp_by_window.get(window_key),
		}
	)


def load_kp_by_window(kp_path: Path) -> dict[str, float | None]:
	"""Charge Kp et ramène les valeurs sur des fenêtres de 15 minutes."""
	if not kp_path.exists():
		raise FileNotFoundError(f"Fichier Kp introuvable: {kp_path}")

	kp_by_window: dict[str, float | None] = {}
	with kp_path.open("r", encoding="utf-8", newline="") as kp_file:
		reader = csv.DictReader(kp_file, delimiter=";")
		if reader.fieldnames is None:
			raise ValueError(f"CSV Kp vide ou sans en-tête: {kp_path}")
		required_columns = {"Date", "Kp"}
		if not required_columns.issubset(set(reader.fieldnames)):
			missing = sorted(required_columns - set(reader.fieldnames))
			raise ValueError(f"Colonnes manquantes dans {kp_path.name}: {missing}")

		for row in reader:
			raw_date = (row.get("Date") or "").strip()
			try:
				timestamp = datetime.strptime(raw_date, DATE_FORMAT)
			except ValueError:
				continue

			window_key = floor_15min(timestamp).strftime(DATE_FORMAT)
			kp_value = parse_optional_float(row.get("Kp"))
			if kp_value is not None:
				# On garde la dernière valeur observée dans la fenêtre.
				kp_by_window[window_key] = kp_value

	return kp_by_window


def aggregate_15min_to_csv(input_path: Path, output_path: Path, kp_path: Path) -> int:
	"""Agrège le CSV source par fenêtres de 15 minutes et écrit le CSV de métriques."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	kp_by_window = load_kp_by_window(kp_path)

	total_rows = 0
	current: WindowState | None = None

	with input_path.open("r", encoding="utf-8", newline="") as in_file, output_path.open("w", encoding="utf-8", newline="") as out_file:
		reader = csv.DictReader(in_file, delimiter=";")
		required_columns = {"Date", "Speed", "Density", "Bt"}
		if reader.fieldnames is None:
			raise ValueError(f"CSV vide ou sans en-tête: {input_path}")
		if not required_columns.issubset(set(reader.fieldnames)):
			missing = sorted(required_columns - set(reader.fieldnames))
			raise ValueError(f"Colonnes manquantes dans {input_path.name}: {missing}")

		writer = csv.DictWriter(
			out_file,
			fieldnames=[
				"Date",
				"Bt_integral_15m",
				"Bt_max_15m",
				"Bt_min_15m",
				"Density_integral_15m",
				"Density_max_15m",
				"Density_min_15m",
				"Speed_integral_15m",
				"Speed_max_15m",
				"Speed_min_15m",
				"Kp",
			],
			delimiter=";",
		)
		writer.writeheader()

		for row in reader:
			raw_date = (row.get("Date") or "").strip()
			try:
				timestamp = datetime.strptime(raw_date, DATE_FORMAT)
			except ValueError:
				continue

			window_start = floor_15min(timestamp)
			if current is None:
				current = WindowState(start=window_start)
			elif window_start != current.start:
				write_window(writer, current, kp_by_window)
				total_rows += 1
				current = WindowState(start=window_start)

			bt = parse_optional_float(row.get("Bt"))
			density = parse_optional_float(row.get("Density"))
			speed = parse_optional_float(row.get("Speed"))

			if bt is not None:
				if current.bt_min is None or bt < current.bt_min:
					current.bt_min = bt
				if current.bt_max is None or bt > current.bt_max:
					current.bt_max = bt

				if current.prev_bt_value is not None and current.prev_bt_time is not None:
					dt_seconds = (timestamp - current.prev_bt_time).total_seconds()
					if dt_seconds > 0:
						current.bt_integral += 0.5 * (current.prev_bt_value + bt) * dt_seconds

				current.prev_bt_value = bt
				current.prev_bt_time = timestamp

			if density is not None:
				if current.density_min is None or density < current.density_min:
					current.density_min = density
				if current.density_max is None or density > current.density_max:
					current.density_max = density

				if current.prev_density_value is not None and current.prev_density_time is not None:
					dt_seconds = (timestamp - current.prev_density_time).total_seconds()
					if dt_seconds > 0:
						current.density_integral += 0.5 * (current.prev_density_value + density) * dt_seconds

				current.prev_density_value = density
				current.prev_density_time = timestamp

			if speed is not None:
				if current.speed_min is None or speed < current.speed_min:
					current.speed_min = speed
				if current.speed_max is None or speed > current.speed_max:
					current.speed_max = speed

				if current.prev_speed_value is not None and current.prev_speed_time is not None:
					dt_seconds = (timestamp - current.prev_speed_time).total_seconds()
					if dt_seconds > 0:
						current.speed_integral += 0.5 * (current.prev_speed_value + speed) * dt_seconds

				current.prev_speed_value = speed
				current.prev_speed_time = timestamp

		if current is not None:
			write_window(writer, current, kp_by_window)
			total_rows += 1

	return total_rows


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Génère un CSV de métriques SolarWinds sur 15 minutes (intégrales et min/max pour Bt, Density, Speed)."
	)
	parser.add_argument(
		"--input",
		default=None,
		help="Chemin du CSV source (smooth_solarwinds_filled.csv).",
	)
	parser.add_argument(
		"--kp-input",
		default=str(DEFAULT_KP_INPUT),
		help="Chemin du CSV source Kp (Date;Kp).",
	)
	parser.add_argument(
		"--output",
		default=str(DEFAULT_OUTPUT),
		help="Chemin du CSV de sortie des métriques 15 minutes.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path = resolve_input_path(args.input)
	kp_path = Path(args.kp_input)
	output_path = Path(args.output)

	row_count = aggregate_15min_to_csv(input_path, output_path, kp_path)

	print(f"CSV source : {input_path}")
	print(f"CSV Kp     : {kp_path}")
	print(f"CSV écrit  : {output_path}")
	print(f"Lignes     : {row_count}")


if __name__ == "__main__":
	main()
