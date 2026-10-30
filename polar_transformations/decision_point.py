"""
Estimate a pitcher's decision point in time-space from their average fastball.

Decision time = estimated fastball flight time − 300 ms
(150 ms swing + 150 ms reaction buffer).

Usage:
    python decision_point.py "Kyle Bradish"
    python decision_point.py 680694
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Allow running from repo root or from this folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polar_transformations.reconstruct_trajectory import reconstruct_trajectory

SWING_MS = 150
REACTION_MS = 150
TOTAL_OFFSET_MS = SWING_MS + REACTION_MS
FASTBALL_TYPES = {"FF", "SI", "FA", "FT"}

AVERAGES_PATH = Path(__file__).resolve().parent.parent / "datamining" / "outputs" / "pitch_averages_2025.csv"


def load_averages() -> pd.DataFrame:
    if not AVERAGES_PATH.exists():
        raise FileNotFoundError(f"Averages file not found: {AVERAGES_PATH}")
    return pd.read_csv(AVERAGES_PATH)


def find_pitcher(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return rows for the pitcher matching name or MLBAM ID."""
    query = query.strip()
    # Try as MLBAM ID first
    if query.isdigit():
        rows = df[df["pitcher"] == int(query)]
        if len(rows):
            return rows
    # Fall back to name search if a name column exists; otherwise require ID
    # Current averages file is keyed by pitcher (MLBAM). For name lookup we
    # would need a crosswalk. For now support ID and a simple lookup table.
    # If the user passes a name we try a hard-coded map for common cases.
    name_map = {
        "kyle bradish": 680694,
        "bradish": 680694,
    }
    key = query.lower()
    if key in name_map:
        return df[df["pitcher"] == name_map[key]]
    raise ValueError(
        f"Could not resolve pitcher '{query}'. "
        "Pass an MLBAM ID or add the name to the lookup in this script."
    )


def primary_fastball(rows: pd.DataFrame) -> pd.Series:
    fb = rows[rows["pitch_type"].isin(FASTBALL_TYPES)]
    if fb.empty:
        raise ValueError("No fastball types found for this pitcher")
    # Prefer the type with the most pitches
    return fb.loc[fb["n_pitches"].idxmax()]


def decision_point(pitcher_query: str) -> dict:
    averages = load_averages()
    rows = find_pitcher(averages, pitcher_query)
    fb = primary_fastball(rows)

    traj = reconstruct_trajectory(
        fb["release_pos_x_ft"],
        fb["release_pos_y_ft"],
        fb["release_pos_z_ft"],
        fb["ax_ft_s2"],
        fb["ay_ft_s2"],
        fb["az_ft_s2"],
        fb["plate_x_ft"],
        fb["plate_z_ft"],
        fb["release_speed_mph"],
    )

    flight_time_s = traj["t"].iloc[-1]
    decision_time_s = flight_time_s - (TOTAL_OFFSET_MS / 1000.0)
    decision_time_s = max(decision_time_s, 0.05)  # clamp

    # Interpolate position at decision time
    idx = np.searchsorted(traj["t"].to_numpy(), decision_time_s)
    idx = min(idx, len(traj) - 1)
    pt = traj.iloc[idx]

    return {
        "pitcher": int(fb["pitcher"]),
        "pitch_type": fb["pitch_type"],
        "n_pitches": int(fb["n_pitches"]),
        "release_speed_mph": float(fb["release_speed_mph"]),
        "flight_time_s": float(flight_time_s),
        "decision_time_s": float(decision_time_s),
        "decision_time_ms": float(decision_time_s * 1000),
        "decision_x_ft": float(pt["x"]),
        "decision_y_ft": float(pt["y"]),
        "decision_z_ft": float(pt["z"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate pitcher decision point")
    parser.add_argument("pitcher", help="Pitcher name or MLBAM ID")
    args = parser.parse_args()

    result = decision_point(args.pitcher)
    print(f"Pitcher MLBAM: {result['pitcher']}")
    print(f"Primary fastball: {result['pitch_type']} ({result['n_pitches']} pitches)")
    print(f"Avg velocity: {result['release_speed_mph']:.1f} mph")
    print(f"Flight time: {result['flight_time_s']*1000:.0f} ms")
    print(f"Decision time: {result['decision_time_ms']:.0f} ms after release")
    print(f"Decision point (Statcast ft):")
    print(f"  x = {result['decision_x_ft']:.2f}")
    print(f"  y = {result['decision_y_ft']:.2f}")
    print(f"  z = {result['decision_z_ft']:.2f}")


if __name__ == "__main__":
    main()
