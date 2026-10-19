"""
For each pitcher, compute average trajectory parameters for each fastball type
(FF, SI, FC) that they throw.

Reads the full-season pitch CSV produced by the pull scripts and writes
a summary CSV with one row per pitcher × fastball type.

Usage:
    python average_primary_fastball.py --year 2025
    python average_primary_fastball.py --year 2026
"""

import argparse
import pandas as pd
from pathlib import Path

FASTBALL_TYPES = ["FF", "SI", "FC"]  # four-seam, sinker, cutter

AVG_COLS = [
    "release_pos_x_ft",
    "release_pos_y_ft",
    "release_pos_z_ft",
    "ax_ft_s2",
    "ay_ft_s2",
    "az_ft_s2",
    "release_speed_mph",
    "pfx_x_ft",
    "pfx_z_ft",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=[2025, 2026])
    args = parser.parse_args()

    input_path = Path(f"datamining/pitches_{args.year}.csv")
    output_path = Path(f"datamining/fastball_averages_{args.year}.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Pitch data not found: {input_path}. Run the pull script first.")

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    # Drop rows missing pitch type or core trajectory fields
    needed = ["pitcher", "pitch_type", "release_speed_mph"] + [c for c in AVG_COLS if c in df.columns]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    # Keep only the three fastball types
    df = df[df["pitch_type"].isin(FASTBALL_TYPES)]

    results = []
    for (pitcher_id, pitch_type), group in df.groupby(["pitcher", "pitch_type"]):
        if group.empty:
            continue

        row = {
            "pitcher": int(pitcher_id),
            "fastball_type": pitch_type,
            "n_pitches": len(group),
        }
        for col in AVG_COLS:
            if col in group.columns:
                row[col] = group[col].mean()
        results.append(row)

    summary = pd.DataFrame(results)
    summary.to_csv(output_path, index=False)
    print(f"Wrote {len(summary):,} pitcher-fastball rows to {output_path}")

if __name__ == "__main__":
    main()
