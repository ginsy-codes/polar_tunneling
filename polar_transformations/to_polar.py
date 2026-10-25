"""
Convert a Cartesian pitch trajectory into polar coordinates relative to
batter-centric origins for both right-handed (RHH) and left-handed (LHH) hitters.

Origins (in inches, then converted to feet to match trajectory units):
- z = 65"
- y = -28.3"
- x = +35.8" (RHH) or -35.8" (LHH)

Polar definitions (relative to the origin):
- r     : Euclidean distance from origin
- theta : rotation about the z-axis (azimuth in the xy-plane), atan2(dy, dx)
- phi   : rotation about the x-axis (angle in the yz-plane), atan2(dz, dy)

Input trajectory DataFrame must contain columns: t, x, y, z (feet).
Output: two DataFrames (rhh, lhh) with columns t, r, theta, phi.
Angles are in radians.
"""

import numpy as np
import pandas as pd

# Origins in inches
ORIGIN_Z_IN = 65.0
ORIGIN_Y_IN = -28.3
ORIGIN_X_RHH_IN = 35.8
ORIGIN_X_LHH_IN = -35.8

# Convert to feet (Statcast trajectory units)
IN_TO_FT = 1.0 / 12.0
ORIGIN_Z = ORIGIN_Z_IN * IN_TO_FT
ORIGIN_Y = ORIGIN_Y_IN * IN_TO_FT
ORIGIN_X_RHH = ORIGIN_X_RHH_IN * IN_TO_FT
ORIGIN_X_LHH = ORIGIN_X_LHH_IN * IN_TO_FT

def _cartesian_to_polar(traj: pd.DataFrame, ox: float, oy: float, oz: float) -> pd.DataFrame:
    """
    Convert trajectory points to polar coordinates relative to (ox, oy, oz).
    """
    dx = traj["x"].to_numpy() - ox
    dy = traj["y"].to_numpy() - oy
    dz = traj["z"].to_numpy() - oz

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    # theta: rotation about z-axis (azimuth in xy-plane)
    theta = np.arctan2(dy, dx)
    # phi: rotation about x-axis (angle in yz-plane)
    phi = np.arctan2(dz, dy)

    out = pd.DataFrame({
        "t": traj["t"].to_numpy(),
        "r": r,
        "theta": theta,
        "phi": phi,
    })
    return out

def to_polar_both(traj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (rhh_polar, lhh_polar) DataFrames for the given Cartesian trajectory.

    Parameters
    ----------
    traj : pd.DataFrame
        Must contain columns t, x, y, z (positions in feet).

    Returns
    -------
    rhh : pd.DataFrame
        Polar coordinates relative to right-handed hitter origin.
    lhh : pd.DataFrame
        Polar coordinates relative to left-handed hitter origin.
    """
    required = {"t", "x", "y", "z"}
    if not required.issubset(traj.columns):
        raise ValueError(f"Trajectory must contain columns {required}")

    rhh = _cartesian_to_polar(traj, ORIGIN_X_RHH, ORIGIN_Y, ORIGIN_Z)
    lhh = _cartesian_to_polar(traj, ORIGIN_X_LHH, ORIGIN_Y, ORIGIN_Z)
    return rhh, lhh

def main():
    """Simple CLI demo using a synthetic straight-line trajectory."""
    # Example: straight line from release toward plate
    t = np.linspace(0, 0.4, 20)
    x = np.full_like(t, -1.5)
    y = 54.0 - (54.0 / 0.4) * t
    z = 6.0 - (6.0 / 0.4) * t
    traj = pd.DataFrame({"t": t, "x": x, "y": y, "z": z})

    rhh, lhh = to_polar_both(traj)
    print("RHH polar (first 5 rows):")
    print(rhh.head())
    print("\nLHH polar (first 5 rows):")
    print(lhh.head())

if __name__ == "__main__":
    main()
