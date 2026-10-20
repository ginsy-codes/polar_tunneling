"""
3D visualization of pitch trajectories.

Input: a DataFrame (or list of DataFrames) with columns t, x, y, z
        as produced by polar_transformations.reconstruct_trajectory.

Usage (as library):
    from report_generation.plot_trajectory_3d import plot_trajectories
    plot_trajectories([traj_df1, traj_df2], labels=["FF", "SL"])

Usage (CLI — expects a CSV with columns t,x,y,z):
    python plot_trajectory_3d.py path/to/trajectory.csv --output plot.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_trajectories(
    trajectories,
    labels=None,
    title="Pitch Trajectories",
    output_path=None,
    show=True,
):
    """
    Plot one or more trajectories in 3D.

    Parameters
    ----------
    trajectories : list of pd.DataFrame or single pd.DataFrame
        Each DataFrame must contain columns: x, y, z (t is optional).
    labels : list of str, optional
        Legend labels, one per trajectory.
    title : str
    output_path : str or Path, optional
        If given, save the figure to this path.
    show : bool
        Whether to call plt.show().
    """
    if isinstance(trajectories, pd.DataFrame):
        trajectories = [trajectories]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, traj in enumerate(trajectories):
        label = labels[i] if labels and i < len(labels) else f"Pitch {i+1}"
        ax.plot(traj["x"], traj["y"], traj["z"], label=label, linewidth=2)

    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    ax.set_zlabel("z (ft)")
    ax.set_title(title)
    ax.legend()

    # Reasonable default view (catcher perspective-ish)
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot a 3D pitch trajectory from CSV")
    parser.add_argument("csv", type=str, help="Path to trajectory CSV (columns: t, x, y, z)")
    parser.add_argument("--output", type=str, default=None, help="Save figure to this path")
    parser.add_argument("--title", type=str, default="Pitch Trajectory")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    args = parser.parse_args()

    traj = pd.read_csv(args.csv)
    required = {"x", "y", "z"}
    if not required.issubset(traj.columns):
        raise ValueError(f"CSV must contain columns {required}")

    plot_trajectories(
        traj,
        title=args.title,
        output_path=args.output,
        show=not args.no_show,
    )

if __name__ == "__main__":
    main()
