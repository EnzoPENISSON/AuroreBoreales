import argparse
import json
from pathlib import Path

import pandas as pd


def resolve_records(payload):
	if isinstance(payload, list):
		return payload

	if isinstance(payload, dict):
		for value in payload.values():
			if isinstance(value, list):
				return value
		return [payload]

	raise ValueError("Unsupported JSON structure: expected a list or an object.")


def json_to_csv(input_path, output_path, sep=","):
	with input_path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	records = resolve_records(payload)
	if not records:
		raise ValueError("The JSON file is empty: no records to export.")

	df = pd.json_normalize(records)
	df.to_csv(output_path, index=False, sep=sep)

	print(f"CSV created: {output_path}")
	print(f"Rows: {len(df)} | Columns: {len(df.columns)}")


def main():
	parser = argparse.ArgumentParser(description="Convert a JSON file to CSV.")
	parser.add_argument(
		"-i",
		"--input",
		type=Path,
		default=Path("data/json/merged-storm.json"),
		help="Input JSON file path",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=Path("data/json/merged-storm.csv"),
		help="Output CSV file path",
	)
	parser.add_argument(
		"--sep",
		default=",",
		help="CSV delimiter (default: ,)",
	)

	args = parser.parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	args.output.parent.mkdir(parents=True, exist_ok=True)
	json_to_csv(args.input, args.output, sep=args.sep)


if __name__ == "__main__":
	main()
