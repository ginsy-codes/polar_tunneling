"""
3D visualization of pitch trajectories from a batter's viewpoint.

Input: a DataFrame (or list of DataFrames) with columns t, x, y, z
        as produced by polar_transformations.reconstruct_trajectory.

The plot origin (0, 0, 0) is exactly the batter's eye position defined in
polar_transformations.to_polar for the chosen handedness.

The coordinate system is rotated so the vector from the batter's eye to the
pitcher release point points directly into the page (positive depth).
Vertical is constrained so the original +z (up) remains positive vertical.
The horizontal axis is justified to the most negative of:
  - the batter eye (origin)
  - the release point
  - the most-negative point on the trajectory

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

# Origins in feet (matching to_polar.py)
# RHH stands on the negative-x side in Statcast coordinates
IN_TO_FT = 1.0 / 12.0
ORIGIN_Z = 65.0 * IN_TO_FT
ORIGIN_Y = -28.3 * IN_TO_FT
ORIGIN_X_RHH = -35.8 * IN_TO_FT
ORIGIN_X_LHH = 35.8 * IN_TO_FT

def _shift_to_viewpoint(traj: pd.DataFrame, handedness: str) -> pd.DataFrame:
    """Return a copy of traj with coordinates relative to the batter origin."""
    ox = ORIGIN_X_RHH if handedness.upper() == "R" else ORIGIN_X_LHH
    shifted = traj.copy()
    shifted["x"] = traj["x"] - ox
    shifted["y"] = traj["y"] - ORIGIN_Y
    shifted["z"] = traj["z"] - ORIGIN_Z
    return shifted

def _rotate_so_release_is_depth(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Rotate so the release point lies on the positive depth axis, while keeping
    the new vertical axis oriented with the original +z (up).
    """
    rx, ry, rz = traj.iloc[0][["x", "y", "z"]].to_numpy()
    v = np.array([rx, ry, rz])
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return traj.copy()
    depth = v / norm  # positive depth toward release

    # Prefer original +z as the up reference
    world_up = np.array([0.0, 0.0, 1.0])
    # Horizontal = depth × up, then flip if needed so it is consistent
    horiz = np.cross(depth, world_up)
    if np.linalg.norm(horiz) < 1e-6:
        # Depth nearly parallel to z; fall back to world x
        horiz = np.cross(depth, np.array([1.0, 0.0, 0.0]))
    horiz = horiz / np.linalg.norm(horiz)

    # Vertical = horiz × depth, then ensure it has positive component along original +z
    vert = np.cross(horiz, depth)
    vert = vert / np.linalg.norm(vert)
    if vert[2] < 0:
        vert = -vert
        horiz = -horiz  # keep right-handed

    pts = traj[["x", "y", "z"]].to_numpy()
    new_x = pts @ horiz
    new_y = pts @ depth
    new_z = pts @ vert

    out = traj.copy()
    out["x"] = new_x
    out["y"] = new_y
    out["z"] = new_z
    return out

def plot_trajectories(
    trajectories,
    handedness: str = "R",
    labels=None,
    title=None,
    output_path=None,
    show=True,
):
    """
    Plot one or more trajectories in 3D from the batter's viewpoint.
    The origin (0, 0, 0) is exactly the batter's eye.
    The view is rotated so the eye-to-release line points into the page,
    and original +z remains upward.
    """
    if isinstance(trajectories, pd.DataFrame):
        trajectories = [trajectories]

    handedness = handedness.upper()
    if handedness not in ("R", "L"):
        raise ValueError("handedness must be 'R' or 'L'")

    if title is None:
        title = f"Pitch Trajectories ({handedness}HH viewpoint)"

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.90], projection="3d")

    all_x, all_y, all_z = [0.0], [0.0], [0.0]
    release_xs = [0.0]

    for i, traj in enumerate(trajectories):
        shifted = _shift_to_viewpoint(traj, handedness)
        rotated = _rotate_so_release_is_depth(shifted)
        label = labels[i] if labels and i < len(labels) else f"Pitch {i+1}"
        ax.plot(rotated["x"], rotated["y"], rotated["z"], label=label, linewidth=2)
        all_x.extend(rotated["x"])
        all_y.extend(rotated["y"])
        all_z.extend(rotated["z"])
        release_xs.append(rotated.iloc[0]["x"])

    # Most-negative horizontal reference among origin, release, and trajectory
    candidates = [0.0, min(release_xs), min(all_x)]
    x_left = min(candidates)
    pad = 1.0
    x_min = x_left
    x_max = max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_zlim(z_min - pad, z_max + pad)

    # Mark the origin (batter's eye)
    ax.scatter([0], [0], [0], color="red", s=80, label="Batter eye")

    ax.set_xlabel("horizontal (ft)", fontsize=11, labelpad=6)
    ax.set_ylabel("")  # depth labels are illegible when looking down the tunnel
    ax.set_yticklabels([])
    ax.set_zlabel("vertical (ft)", fontsize=11, labelpad=6)
    ax.set_title(title, fontsize=13, pad=8)
    ax.legend(loc="upper right", fontsize=10)

    # Look along positive depth (eye -> release points into the page)
    ax.view_init(elev=0, azim=-90)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot a 3D pitch trajectory from batter viewpoint")
    parser.add_argument("csv", type=str, help="Path to trajectory CSV (columns: t, x, y, z)")
    parser.add_argument("--handedness", type=str, default="R", choices=["R", "L", "r", "l"],
                        help="Batter handedness for viewpoint origin")
    parser.add_argument("--output", type=str, default=None, help="Save figure to this path")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    traj = pd.read_csv(args.csv)
    required = {"x", "y", "z"}
    if not required.issubset(traj.columns):
        raise ValueError(f"CSV must contain columns {required}")

    plot_trajectories(
        traj,
        handedness=args.handedness,
        title=args.title,
        output_path=args.output,
        show=not args.no_show,
    )

if __name__ == "__main__":
    main()
