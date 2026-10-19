"""
Pull 2024 pitch-level Statcast data via pybaseball and write selected columns to CSV.

Required columns:
- pitcher, batter (MLBAM IDs)
- release_pos_x, release_pos_y, release_pos_z
- ax, ay, az
- release_speed
- pfx_x, pfx_z
- outcome (description)

Usage:
    pip install pybaseball pandas
    python pull_2024_pitches.py
"""

import pandas as pd
from pybaseball import statcast

# Date range for the 2024 regular season (approximate; adjust as needed)
START_DATE = "2024-03-20"
END_DATE = "2024-10-01"
OUTPUT_PATH = "datamining/pitches_2024.csv"

# Columns needed for trajectory work and identification
COLUMNS = [
    "pitcher",          # MLBAM ID of pitcher
    "batter",           # MLBAM ID of batter
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "ax",
    "ay",
    "az",
    "release_speed",
    "pfx_x",
    "pfx_z",
    "description",      # outcome (ball, called_strike, swinging_strike, in_play, etc.)
    # Useful extras often needed later (can be trimmed)
    "pitch_type",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
]

def main():
    print(f"Pulling Statcast data from {START_DATE} to {END_DATE}...")
    df = statcast(start_dt=START_DATE, end_dt=END_DATE, verbose=True, parallel=True)

    # Keep only the columns we care about (drop any that are missing)
    available = [c for c in COLUMNS if c in df.columns]
    df = df[available].copy()

    # Basic cleanup
    required = ["pitcher", "release_pos_x", "release_pos_y", "release_pos_z", "ax", "ay", "az"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    df["pitcher"] = df["pitcher"].astype(int)
    if "batter" in df.columns:
        df["batter"] = df["batter"].astype(int)

    print(f"Writing {len(df):,} pitches to {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
