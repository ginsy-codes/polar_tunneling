"""
Compute correlations between polar path integrals and pitch performance metrics.
Generates scatter plots with r^2 for each integral x metric combination.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PITCH_FILE = "datamining/outputs/pitches_2025.csv"
INTEGRALS_PATH = Path("datamining/outputs/path_integrals_2025.csv")
OUT_DIR = Path("report_generation/correlations/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = Path("report_generation/correlations/correlation_summary.csv")

SWING_DESCS = {
    "foul", "foul_tip", "swinging_strike", "swinging_strike_blocked",
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
    "hit_into_play_line_drive", "hit_into_play_ground_ball", "hit_into_play_fly_ball",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked"}
CALLED_STRIKE = {"called_strike"}


def compute_pitch_metrics() -> pd.DataFrame:
    """Aggregate whiff%, swing%, CSW% per pitcher + pitch_type + stand (handedness)."""
    pitches = pd.read_csv(PITCH_FILE, usecols=["pitcher", "pitch_type", "description", "stand"])
    pitches = pitches.dropna(subset=["pitch_type", "stand"])

    pitches["is_swing"] = pitches["description"].isin(SWING_DESCS)
    pitches["is_whiff"] = pitches["description"].isin(WHIFF_DESCS)
    pitches["is_called_strike"] = pitches["description"].isin(CALLED_STRIKE)

    agg = pitches.groupby(["pitcher", "pitch_type", "stand"]).agg(
        n_pitches=("description", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        called_strikes=("is_called_strike", "sum"),
    ).reset_index()

    agg["whiff_rate"] = np.where(agg["swings"] > 0, agg["whiffs"] / agg["swings"], np.nan)
    agg["swing_rate"] = agg["swings"] / agg["n_pitches"]
    agg["csw_rate"] = (agg["whiffs"] + agg["called_strikes"]) / agg["n_pitches"]
    return agg


def main():
    print("Computing pitch metrics...")
    metrics = compute_pitch_metrics()
    print(f"Metrics for {len(metrics)} pitcher-pitch combos")

    integrals = pd.read_csv(INTEGRALS_PATH)
    # Join on pitcher and target pitch type
    merged = integrals.merge(
        metrics,
        left_on=["pitcher", "target", "handedness"],
        right_on=["pitcher", "pitch_type", "stand"],
        how="inner",
    )
    # Filter to reasonable sample size
    merged = merged[merged["n_pitches"] >= 30].copy()  # slightly lower threshold since split by handedness
    print(f"Merged rows with >=50 pitches: {len(merged)}")

    integral_cols = [
        "pre_net_r", "pre_tv_r", "pre_net_theta", "pre_tv_theta",
        "pre_net_phi", "pre_tv_phi",
        "post_net_r", "post_tv_r", "post_net_theta", "post_tv_theta",
        "post_net_phi", "post_tv_phi",
    ]
    metric_cols = ["whiff_rate", "swing_rate", "csw_rate"]

    summary_rows = []

    for integ in integral_cols:
        for met in metric_cols:
            sub = merged[[integ, met]].dropna()
            if len(sub) < 20:
                continue
            x = sub[integ].to_numpy()
            y = sub[met].to_numpy()
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r2 = r_value ** 2

            summary_rows.append({
                "integral": integ,
                "metric": met,
                "n": len(sub),
                "r": r_value,
                "r2": r2,
                "slope": slope,
                "p_value": p_value,
            })

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x, y, alpha=0.4, s=12, edgecolors="none")
            # Regression line
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=2)
            ax.set_xlabel(integ)
            ax.set_ylabel(met)
            ax.set_title(f"{integ} vs {met}\nr² = {r2:.3f}  (n={len(sub)}, p={p_value:.3g})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_file = OUT_DIR / f"{integ}_vs_{met}.png"
            plt.savefig(out_file, dpi=120)
            plt.close()
            print(f"Saved {out_file.name}  r²={r2:.3f}")

    summary = pd.DataFrame(summary_rows).sort_values("r2", ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSummary written to {SUMMARY_PATH}")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
