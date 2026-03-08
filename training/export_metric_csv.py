import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SFT or GRPO metrics.json to CSV for charting.")
    parser.add_argument("--metrics-path", required=True, help="Path to metrics.json from SFT or GRPO.")
    parser.add_argument("--output-path", required=True, help="Path to write CSV.")
    return parser.parse_args()


def _write_rows(output_path: str, rows: list[dict[str, Any]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_grpo(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in metrics.get("reward_history", []):
        rows.append(dict(row))
    return rows


def export_sft(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in metrics.get("log_history", []):
        if "step" not in row:
            continue
        rows.append(dict(row))

    if not rows:
        rows = [
            {
                "series": "pre",
                "metric_name": metrics["pre"]["metric_name"],
                "metric_value": metrics["pre"]["metric_value"],
            },
            {
                "series": "post",
                "metric_name": metrics["post"]["metric_name"],
                "metric_value": metrics["post"]["metric_value"],
            },
        ]
    return rows


def main() -> None:
    args = parse_args()
    metrics = json.loads(Path(args.metrics_path).read_text())

    if "reward_history" in metrics:
        rows = export_grpo(metrics)
    else:
        rows = export_sft(metrics)

    if not rows:
        raise ValueError("No rows found to export.")

    _write_rows(args.output_path, rows)
    print(f"Wrote {len(rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
