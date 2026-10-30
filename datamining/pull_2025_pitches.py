"""
Pull 2025 pitch-level Statcast data via pybaseball and write selected columns to CSV.

Intended as the primary training set for the polar tunneling model.
Pitcher-focused only (batter ID removed).

Columns (with units where applicable):
- pitcher (MLBAM ID)
- release_pos_x_ft, release_pos_y_ft, release_pos_z_ft
- ax_ft_s2, ay_ft_s2, az_ft_s2
- release_speed_mph
- plate_x_ft, plate_z_ft
- description (per-pitch outcome)
- events (plate-appearance outcome)

Usage:
    pip install pybaseball pandas
    python pull_2025_pitches.py
"""

import pandas as pd
from pathlib import Path
from pybaseball import statcast, cache

# Enable caching so large queries can be resumed if interrupted
cache.enable()

# Date range for the 2025 regular season (approximate; adjust as needed)
START_DATE = "2025-03-20"
END_DATE = "2025-10-01"
OUTPUT_DIR = Path("datamining/outputs")
OUTPUT_PATH = OUTPUT_DIR / "pitches_2025.csv"

# Columns needed for trajectory work and identification
COLUMNS = [
    "pitcher",          # MLBAM ID of pitcher
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "ax",
    "ay",
    "az",
    "vx0",
    "vy0",
    "vz0",
    "release_speed",
    "plate_x",
    "plate_z",
    "description",      # per-pitch outcome
    "events",           # plate-appearance outcome (null on non-terminal pitches)
    # Useful extras
    "pitch_type",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
]

# Rename to include units
RENAME = {
    "release_pos_x": "release_pos_x_ft",
    "release_pos_y": "release_pos_y_ft",
    "release_pos_z": "release_pos_z_ft",
    "ax": "ax_ft_s2",
    "ay": "ay_ft_s2",
    "az": "az_ft_s2",
    "vx0": "vx0_ft_s",
    "vy0": "vy0_ft_s",
    "vz0": "vz0_ft_s",
    "release_speed": "release_speed_mph",
    "plate_x": "plate_x_ft",
    "plate_z": "plate_z_ft",
}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pulling Statcast data from {START_DATE} to {END_DATE}...")
    df = statcast(start_dt=START_DATE, end_dt=END_DATE, verbose=True, parallel=True)

    # Keep only the columns we care about (drop any that are missing)
    available = [c for c in COLUMNS if c in df.columns]
    df = df[available].copy()

    # Basic cleanup
    required = ["pitcher", "release_pos_x", "release_pos_y", "release_pos_z", "ax", "ay", "az"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    df["pitcher"] = df["pitcher"].astype(int)

    # Add units to column names
    df = df.rename(columns=RENAME)

    print(f"Writing {len(df):,} pitches to {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
