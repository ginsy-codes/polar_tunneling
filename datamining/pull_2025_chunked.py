"""
Pull 2025 Statcast data in monthly chunks to avoid memory issues,
then concatenate into a single CSV.

Usage:
    python pull_2025_chunked.py
"""

import pandas as pd
from pathlib import Path
from pybaseball import statcast, cache

cache.enable()

YEAR = 2025
OUTPUT_DIR = Path("datamining/outputs")
FINAL_PATH = OUTPUT_DIR / f"pitches_{YEAR}.csv"

# Approximate regular-season months
MONTHS = [
    ("2025-03-20", "2025-03-31"),
    ("2025-04-01", "2025-04-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-10-01", "2025-10-01"),
]

COLUMNS = [
    "pitcher",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "release_speed",
    "plate_x",
    "plate_z",
    "description",
    "events",
    "pitch_type",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
]

RENAME = {
    "release_pos_x": "release_pos_x_ft",
    "release_pos_y": "release_pos_y_ft",
    "release_pos_z": "release_pos_z_ft",
    "vx0": "vx0_ft_s",
    "vy0": "vy0_ft_s",
    "vz0": "vz0_ft_s",
    "ax": "ax_ft_s2",
    "ay": "ay_ft_s2",
    "az": "az_ft_s2",
    "release_speed": "release_speed_mph",
    "plate_x": "plate_x_ft",
    "plate_z": "plate_z_ft",
}

def process_chunk(start, end):
    print(f"Pulling {start} to {end}...")
    df = statcast(start_dt=start, end_dt=end, verbose=False, parallel=True)
    if df is None or df.empty:
        return None
    available = [c for c in COLUMNS if c in df.columns]
    df = df[available].copy()
    required = ["pitcher", "release_pos_x", "release_pos_y", "release_pos_z", "ax", "ay", "az"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    if "pitcher" in df.columns:
        df["pitcher"] = df["pitcher"].astype(int)
    df = df.rename(columns=RENAME)
    return df

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_paths = []

    for i, (start, end) in enumerate(MONTHS):
        chunk_path = OUTPUT_DIR / f"_chunk_{YEAR}_{i:02d}.csv"
        if chunk_path.exists():
            print(f"Chunk already exists: {chunk_path}")
            chunk_paths.append(chunk_path)
            continue
        df = process_chunk(start, end)
        if df is not None and not df.empty:
            df.to_csv(chunk_path, index=False)
            chunk_paths.append(chunk_path)
            print(f"Wrote {len(df):,} pitches to {chunk_path}")
        else:
            print(f"No data for {start} to {end}")

    print("Concatenating chunks...")
    frames = []
    for p in chunk_paths:
        frames.append(pd.read_csv(p))
    if not frames:
        print("No chunks produced.")
        return
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(FINAL_PATH, index=False)
    print(f"Wrote {len(full):,} pitches to {FINAL_PATH}")

if __name__ == "__main__":
    main()
