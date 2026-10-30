"""
Compute polar-path divergence integrals versus a reference fastball
up to a decision time.

For a target pitch and a reference (primary fastball) trajectory,
both converted to polar (r, theta, phi) in the batter frame:

    delta_r(t)     = r_target(t) - r_ref(t)
    delta_theta(t) = theta_target(t) - theta_ref(t)
    delta_phi(t)   = phi_target(t) - phi_ref(t)

We report two integrals for each coordinate up to decision_time:
  - net change:  integral of d(delta)/dt  = delta(T) - delta(0)
  - total variation: integral of |d(delta)/dt|

Usage:
    python path_integrals.py --pitcher 680694 --target-type SL --handedness R
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polar_transformations.reconstruct_trajectory import reconstruct_trajectory
from polar_transformations.to_polar import to_polar_both
from polar_transformations.decision_point import decision_point, AVERAGES_PATH, FASTBALL_TYPES

def _traj_from_row(row: pd.Series) -> pd.DataFrame:
    return reconstruct_trajectory(
        row["release_pos_x_ft"],
        row["release_pos_y_ft"],
        row["release_pos_z_ft"],
        row["ax_ft_s2"],
        row["ay_ft_s2"],
        row["az_ft_s2"],
        row["plate_x_ft"],
        row["plate_z_ft"],
        row["release_speed_mph"],
    )

def _align_and_diff(target_polar: pd.DataFrame, ref_polar: pd.DataFrame, t_max: float):
    """Resample both polar series onto a common time grid up to t_max and return deltas."""
    t = np.linspace(0, t_max, 200)
    def interp(df, col):
        return np.interp(t, df["t"].to_numpy(), df[col].to_numpy())
    dr = interp(target_polar, "r") - interp(ref_polar, "r")
    dth = interp(target_polar, "theta") - interp(ref_polar, "theta")
    dph = interp(target_polar, "phi") - interp(ref_polar, "phi")
    # unwrap angles so differences are continuous
    dth = np.unwrap(dth)
    dph = np.unwrap(dph)
    return t, dr, dth, dph

def _integrals(t, delta):
    """Net change and total variation of delta over t."""
    d_dt = np.gradient(delta, t)
    net = delta[-1] - delta[0]
    total_var = np.trapezoid(np.abs(d_dt), t)
    return net, total_var

POST_DECISION_MS = 100

def path_integrals(pitcher_id: int, target_type: str, handedness: str = "R") -> dict:
    averages = pd.read_csv(AVERAGES_PATH)
    rows = averages[averages["pitcher"] == pitcher_id]
    if rows.empty:
        raise ValueError(f"No averages for pitcher {pitcher_id}")

    # Reference = primary fastball
    fb_rows = rows[rows["pitch_type"].isin(FASTBALL_TYPES)]
    if fb_rows.empty:
        raise ValueError("No fastball found")
    ref_row = fb_rows.loc[fb_rows["n_pitches"].idxmax()]

    # Target
    tgt_rows = rows[rows["pitch_type"] == target_type]
    if tgt_rows.empty:
        raise ValueError(f"No {target_type} found for pitcher {pitcher_id}")
    tgt_row = tgt_rows.iloc[0]

    # Decision time from the fastball
    dp = decision_point(str(pitcher_id))
    t_dec = dp["decision_time_s"]
    t_post = t_dec + POST_DECISION_MS / 1000.0

    ref_cart = _traj_from_row(ref_row)
    tgt_cart = _traj_from_row(tgt_row)

    ref_polar = to_polar_both(ref_cart)[0 if handedness.upper()=="R" else 1]
    tgt_polar = to_polar_both(tgt_cart)[0 if handedness.upper()=="R" else 1]

    # Pre-decision window: 0 → t_dec
    t1, dr1, dth1, dph1 = _align_and_diff(tgt_polar, ref_polar, t_dec)
    net_r1, tv_r1 = _integrals(t1, dr1)
    net_th1, tv_th1 = _integrals(t1, dth1)
    net_ph1, tv_ph1 = _integrals(t1, dph1)

    # Post-decision window: t_dec → t_dec + 100 ms
    # Align full series then slice
    t_full = np.linspace(0, max(ref_polar["t"].max(), tgt_polar["t"].max()), 500)
    def interp(df, col):
        return np.interp(t_full, df["t"].to_numpy(), df[col].to_numpy())
    dr_full = interp(tgt_polar, "r") - interp(ref_polar, "r")
    dth_full = np.unwrap(interp(tgt_polar, "theta") - interp(ref_polar, "theta"))
    dph_full = np.unwrap(interp(tgt_polar, "phi") - interp(ref_polar, "phi"))

    mask = (t_full >= t_dec) & (t_full <= t_post)
    t2 = t_full[mask]
    if len(t2) < 2:
        net_r2 = tv_r2 = net_th2 = tv_th2 = net_ph2 = tv_ph2 = float("nan")
    else:
        net_r2, tv_r2 = _integrals(t2, dr_full[mask])
        net_th2, tv_th2 = _integrals(t2, dth_full[mask])
        net_ph2, tv_ph2 = _integrals(t2, dph_full[mask])

    return {
        "pitcher": pitcher_id,
        "reference": ref_row["pitch_type"],
        "target": target_type,
        "handedness": handedness.upper(),
        "decision_time_ms": t_dec * 1000,
        "pre_dp_net_r": net_r1, "pre_dp_tv_r": tv_r1,
        "pre_dp_net_theta": net_th1, "pre_dp_tv_theta": tv_th1,
        "pre_dp_net_phi": net_ph1, "pre_dp_tv_phi": tv_ph1,
        "post_dp_net_r": net_r2, "post_dp_tv_r": tv_r2,
        "post_dp_net_theta": net_th2, "post_dp_tv_theta": tv_th2,
        "post_dp_net_phi": net_ph2, "post_dp_tv_phi": tv_ph2,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitcher", type=int, required=True, help="MLBAM ID")
    parser.add_argument("--target-type", required=True, help="Pitch type to compare (e.g. SL)")
    parser.add_argument("--handedness", default="R", choices=["R", "L"])
    args = parser.parse_args()

    result = path_integrals(args.pitcher, args.target_type, args.handedness)
    print(f"Pitcher {result['pitcher']} | {result['target']} vs {result['reference']} ({result['handedness']}HH)")
    print(f"Decision time: {result['decision_time_ms']:.0f} ms")
    print("Pre-decision (0 → decision):")
    print(f"  r:     net={result['pre_dp_net_r']:.4f} ft   total_var={result['pre_dp_tv_r']:.4f} ft")
    print(f"  theta: net={result['pre_dp_net_theta']:.4f} rad  total_var={result['pre_dp_tv_theta']:.4f} rad")
    print(f"  phi:   net={result['pre_dp_net_phi']:.4f} rad  total_var={result['pre_dp_tv_phi']:.4f} rad")
    print("Post-decision (decision → +100 ms):")
    print(f"  r:     net={result['post_dp_net_r']:.4f} ft   total_var={result['post_dp_tv_r']:.4f} ft")
    print(f"  theta: net={result['post_dp_net_theta']:.4f} rad  total_var={result['post_dp_tv_theta']:.4f} rad")
    print(f"  phi:   net={result['post_dp_net_phi']:.4f} rad  total_var={result['post_dp_tv_phi']:.4f} rad")

if __name__ == "__main__":
    main()
