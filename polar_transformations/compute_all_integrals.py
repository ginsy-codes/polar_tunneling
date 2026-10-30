"""
Wrapper: compute path integrals for every pitcher and every non-primary pitch.

For each pitcher in pitch_averages_2025.csv:
  1. Identify primary fastball (most-thrown among FF/SI/FA/FT)
  2. Compute decision time from that fastball
  3. For every other pitch type the pitcher throws:
       compute pre- and post-decision polar path integrals vs the fastball
  4. Write one row per (pitcher, target pitch type, handedness)

Output: datamining/outputs/path_integrals_2025.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polar_transformations.path_integrals import path_integrals, AVERAGES_PATH, FASTBALL_TYPES

OUT_PATH = AVERAGES_PATH.parent / "path_integrals_2025.csv"


def main():
    averages = pd.read_csv(AVERAGES_PATH)
    pitchers = averages["pitcher"].unique()
    rows = []

    for i, pid in enumerate(pitchers):
        pitcher_rows = averages[averages["pitcher"] == pid]
        fb_rows = pitcher_rows[pitcher_rows["pitch_type"].isin(FASTBALL_TYPES)]
        if fb_rows.empty:
            continue
        primary = fb_rows.loc[fb_rows["n_pitches"].idxmax(), "pitch_type"]
        targets = pitcher_rows[pitcher_rows["pitch_type"] != primary]["pitch_type"].unique()

        for ptype in targets:
            for hand in ("R", "L"):
                try:
                    result = path_integrals(int(pid), ptype, hand)
                    rows.append(result)
                except Exception as e:
                    print(f"skip {pid} {ptype} {hand}: {e}")

        if (i + 1) % 50 == 0:
            print(f"processed {i+1}/{len(pitchers)} pitchers, {len(rows)} rows so far")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
