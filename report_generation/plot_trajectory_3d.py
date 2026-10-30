"""
3D visualization of pitch trajectories from a batter's viewpoint.

Input: a DataFrame (or list of DataFrames) with columns t, x, y, z
        as produced by polar_transformations.reconstruct_trajectory.

The plot origin (0, 0, 0) is exactly the batter's eye position defined in
polar_transformations.to_polar for the chosen handedness.

View settings (approved):
  - elev = 2
  - azim = 85.3 for RHH, 94.7 for LHH
  - equal pixel-per-foot scale (box aspect matches data ranges)
  - thin trajectory lines
  - strike zone box at shifted y=0.5, Statcast z=1.5–3.04 ft, 17" wide
  - axes inverted so +x is right and +y is into the page
  - tight crop to remove whitespace

Usage (as library):
    from report_generation.plot_trajectory_3d import plot_trajectories
    plot_trajectories(traj, handedness="R", labels=["SL"])

Usage (CLI):
    python plot_trajectory_3d.py trajectory.csv --handedness R --output plot.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

# Origins in feet (matching to_polar.py)
IN_TO_FT = 1.0 / 12.0
ORIGIN_Z = 65.0 * IN_TO_FT
ORIGIN_Y = -28.3 * IN_TO_FT
ORIGIN_X_RHH = -35.8 * IN_TO_FT
ORIGIN_X_LHH = 35.8 * IN_TO_FT

# Approved view angles
ELEV = 2
AZIM = {"R": 85.3, "L": 94.7}


def _shift_to_viewpoint(traj: pd.DataFrame, handedness: str) -> pd.DataFrame:
    """Return a copy of traj with coordinates relative to the batter origin."""
    ox = ORIGIN_X_RHH if handedness.upper() == "R" else ORIGIN_X_LHH
    shifted = traj.copy()
    shifted["x"] = traj["x"] - ox
    shifted["y"] = traj["y"] - ORIGIN_Y
    shifted["z"] = traj["z"] - ORIGIN_Z
    return shifted


def _strike_zone(handedness: str, zone_y: float = 0.5):
    """Return (x, y, z) arrays for a strike-zone rectangle in the batter frame."""
    ox = ORIGIN_X_RHH if handedness.upper() == "R" else ORIGIN_X_LHH
    half_w = 17.0 * IN_TO_FT / 2.0
    zone_x_stat = np.array([-half_w, half_w, half_w, -half_w, -half_w])
    zone_x = zone_x_stat - ox
    zone_z_stat = np.array([1.5, 1.5, 3.04, 3.04, 1.5])
    zone_z = zone_z_stat - ORIGIN_Z
    zone_y_arr = np.full_like(zone_x, zone_y)
    return zone_x, zone_y_arr, zone_z


def plot_trajectories(
    trajectories,
    handedness: str = "R",
    labels=None,
    colors=None,
    output_path: str | None = None,
    show_zone: bool = True,
    linewidth: float = 1.2,
):
    """
    Plot one or more trajectories from the chosen batter viewpoint.

    Parameters
    ----------
    trajectories : DataFrame or list of DataFrame
        Each must have columns x, y, z (and optionally t).
    handedness : 'R' or 'L'
    labels : list of str, optional
    colors : list of colors, optional
    output_path : str, optional
        If given, save a tightly cropped PNG here.
    show_zone : bool
        Draw the strike-zone box.
    linewidth : float
        Line width for trajectories.
    """
    if isinstance(trajectories, pd.DataFrame):
        trajectories = [trajectories]
    n = len(trajectories)
    if labels is None:
        labels = [None] * n
    if colors is None:
        colors = [None] * n

    h = handedness.upper()
    azim = AZIM.get(h, 88)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")

    shifted_list = []
    for traj, label, color in zip(trajectories, labels, colors):
        shifted = _shift_to_viewpoint(traj, h)
        shifted_list.append(shifted)
        kw = {"linewidth": linewidth}
        if label:
            kw["label"] = label
        if color:
            kw["color"] = color
        ax.plot(shifted["x"], shifted["y"], shifted["z"], **kw)

    if show_zone:
        zx, zy, zz = _strike_zone(h)
        ax.plot(zx, zy, zz, color="black", linewidth=2)

    ax.view_init(elev=ELEV, azim=azim)

    # Limits from all trajectories + zone
    all_x = pd.concat([s["x"] for s in shifted_list])
    all_y = pd.concat([s["y"] for s in shifted_list])
    all_z = pd.concat([s["z"] for s in shifted_list])
    if show_zone:
        zx, zy, zz = _strike_zone(h)
        all_x = pd.concat([all_x, pd.Series(zx)])
        all_y = pd.concat([all_y, pd.Series(zy)])
        all_z = pd.concat([all_z, pd.Series(zz)])

    pad = 0.5
    x_min, x_max = all_x.min() - pad, all_x.max() + pad
    y_min, y_max = all_y.min() - pad, all_y.max() + pad
    z_min, z_max = all_z.min() - pad, all_z.max() + pad

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if any(labels):
        ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    if output_path:
        tmp = "/tmp/_plot_raw.png"
        plt.savefig(tmp, dpi=250, facecolor="white", bbox_inches="tight", pad_inches=0.05)
        plt.close()
        im = Image.open(tmp).convert("RGB")
        arr = np.array(im)
        # More aggressive mask: anything not near-white
        mask = np.any(arr < 250, axis=2)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Small margin only
            margin = 8
            rmin = max(0, rmin - margin)
            rmax = min(arr.shape[0] - 1, rmax + margin)
            cmin = max(0, cmin - margin)
            cmax = min(arr.shape[1] - 1, cmax + margin)
            im = im.crop((cmin, rmin, cmax + 1, rmax + 1))
        # Force square and large
        w, h_px = im.size
        side = max(w, h_px, 1000)
        canvas = Image.new("RGB", (side, side), "white")
        # Paste centered
        x_off = (side - w) // 2
        y_off = (side - h_px) // 2
        canvas.paste(im, (x_off, y_off))
        # Final resize to consistent large size
        canvas = canvas.resize((1200, 1200), Image.LANCZOS)
        canvas.save(output_path)
        print(f"Saved {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot pitch trajectories from batter viewpoint")
    parser.add_argument("csv", help="CSV with columns x, y, z")
    parser.add_argument("--handedness", default="R", choices=["R", "L"])
    parser.add_argument("--output", default="plot.png")
    parser.add_argument("--no-zone", action="store_true")
    args = parser.parse_args()

    traj = pd.read_csv(args.csv)
    plot_trajectories(
        traj,
        handedness=args.handedness,
        output_path=args.output,
        show_zone=not args.no_zone,
    )


if __name__ == "__main__":
    main()
