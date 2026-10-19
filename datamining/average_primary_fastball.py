"""
For each pitcher, identify their primary fastball and compute average trajectory parameters.

Primary fastball is defined as the highest-usage pitch among FF, SI, FC
(or the highest-velocity pitch if none of those are present).

Reads the full-season pitch CSV produced by the pull scripts and writes
a summary CSV with one row per pitcher.

Usage:
    python average_primary_fastball.py --year 2025
    python average_primary_fastball.py --year 2026
"""

import argparse
import pandas as pd
from pathlib import Path

FASTBALL_TYPES = {"FF", "SI", "FC"}  # four-seam, sinker, cutter

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

def identify_primary_fastball(group: pd.DataFrame) -> str:
    """Return the pitch_type that will serve as the reference fastball."""
    usage = group["pitch_type"].value_counts(normalize=True)
    fastball_usage = usage[usage.index.isin(FASTBALL_TYPES)]
    if not fastball_usage.empty:
        return fastball_usage.idxmax()
    # Fallback: highest average velocity pitch type
    return group.groupby("pitch_type")["release_speed_mph"].mean().idxmax()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=[2025, 2026])
    args = parser.parse_args()

    input_path = Path(f"datamining/pitches_{args.year}.csv")
    output_path = Path(f"datamining/primary_fastball_avg_{args.year}.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Pitch data not found: {input_path}. Run the pull script first.")

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    # Drop rows missing pitch type or core trajectory fields
    df = df.dropna(subset=["pitcher", "pitch_type", "release_speed_mph"] + [c for c in AVG_COLS if c in df.columns])

    results = []
    for pitcher_id, group in df.groupby("pitcher"):
        primary = identify_primary_fastball(group)
        fb = group[group["pitch_type"] == primary]
        if fb.empty:
            continue

        row = {"pitcher": int(pitcher_id), "primary_fastball": primary, "n_pitches": len(fb)}
        for col in AVG_COLS:
            if col in fb.columns:
                row[col] = fb[col].mean()
        results.append(row)

    summary = pd.DataFrame(results)
    summary.to_csv(output_path, index=False)
    print(f"Wrote {len(summary):,} pitchers to {output_path}")

if __name__ == "__main__":
    main()
