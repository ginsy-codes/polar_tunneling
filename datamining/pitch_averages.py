"""
For each pitcher, compute average trajectory parameters for every pitch type they throw.

Reads the monthly pitch CSVs (pitches_{year}_{MM}.csv) produced by the chunked
pull script and writes a summary CSV with one row per pitcher × pitch type.

Usage:
    python pitch_averages.py --year 2025
    python pitch_averages.py --year 2026
"""

import argparse
import pandas as pd
from pathlib import Path

AVG_COLS = [
    "release_pos_x_ft",
    "release_pos_y_ft",
    "release_pos_z_ft",
    "vx0_ft_s",
    "vy0_ft_s",
    "vz0_ft_s",
    "ax_ft_s2",
    "ay_ft_s2",
    "az_ft_s2",
    "release_speed_mph",
    "plate_x_ft",
    "plate_z_ft",
]

def load_year(year: int) -> pd.DataFrame:
    pattern = f"datamining/outputs/pitches_{year}_*.csv"
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No monthly files found matching {pattern}")
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=[2025, 2026])
    args = parser.parse_args()

    print(f"Loading monthly files for {args.year}...")
    df = load_year(args.year)

    needed = ["pitcher", "pitch_type", "release_speed_mph"] + [c for c in AVG_COLS if c in df.columns]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    results = []
    for (pitcher_id, pitch_type), group in df.groupby(["pitcher", "pitch_type"]):
        if group.empty:
            continue
        row = {
            "pitcher": int(pitcher_id),
            "pitch_type": pitch_type,
            "n_pitches": len(group),
        }
        for col in AVG_COLS:
            if col in group.columns:
                row[col] = group[col].mean()
        results.append(row)

    summary = pd.DataFrame(results)
    output_path = Path(f"datamining/outputs/pitch_averages_{args.year}.csv")
    summary.to_csv(output_path, index=False)
    print(f"Wrote {len(summary):,} pitcher-pitch rows to {output_path}")

if __name__ == "__main__":
    main()
