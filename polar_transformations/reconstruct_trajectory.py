"""
Reconstruct a pitch trajectory under the constant-acceleration model.

Assumptions:
- Constant acceleration (ax, ay, az) for the full flight.
- Release position: release_pos_x/y/z
- Plate crossing: y = 0, x = plate_x, z = plate_z
  (using pfx as a proxy for final location for simplicity)
- t = 0 at release

The script solves for flight time and initial velocity components, then
samples the trajectory at a fixed time interval.

Can be used as a library or run as a CLI on a single pitch row.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def solve_initial_velocity(x0, y0, z0, ax, ay, az, xf, zf, t_flight):
    """
    Given constant acceleration and known start/end positions,
    recover initial velocity components for a given flight time.

    x(t) = x0 + vx0*t + 0.5*ax*t**2
    => vx0 = (xf - x0 - 0.5*ax*t**2) / t
    """
    vx0 = (xf - x0 - 0.5 * ax * t_flight**2) / t_flight
    vy0 = (0.0 - y0 - 0.5 * ay * t_flight**2) / t_flight
    vz0 = (zf - z0 - 0.5 * az * t_flight**2) / t_flight
    return vx0, vy0, vz0

def estimate_flight_time(release_speed_mph, y0):
    """
    Rough flight-time estimate from release speed and y-distance.
    release_speed is mph; convert to ft/s.
    """
    speed_ft_s = release_speed_mph * 1.46667  # mph to ft/s
    # Approximate average speed ~ 0.95 * release speed
    avg_speed = 0.95 * speed_ft_s
    return abs(y0) / avg_speed

def reconstruct_trajectory(
    release_pos_x, release_pos_y, release_pos_z,
    ax, ay, az,
    plate_x, plate_z,
    release_speed_mph,
    dt=0.01,
):
    """
    Return a DataFrame with columns: t, x, y, z
    t=0 at release, sampled every dt seconds until plate.
    """
    t_flight = estimate_flight_time(release_speed_mph, release_pos_y)
    vx0, vy0, vz0 = solve_initial_velocity(
        release_pos_x, release_pos_y, release_pos_z,
        ax, ay, az,
        plate_x, plate_z,
        t_flight,
    )

    times = np.arange(0, t_flight + dt/2, dt)
    xs = release_pos_x + vx0 * times + 0.5 * ax * times**2
    ys = release_pos_y + vy0 * times + 0.5 * ay * times**2
    zs = release_pos_z + vz0 * times + 0.5 * az * times**2

    # Ensure last point is exactly at plate if close
    if ys[-1] > 0:
        times = np.append(times, t_flight)
        xs = np.append(xs, plate_x)
        ys = np.append(ys, 0.0)
        zs = np.append(zs, plate_z)

    return pd.DataFrame({"t": times, "x": xs, "y": ys, "z": zs})

def main():
    parser = argparse.ArgumentParser(description="Reconstruct a single pitch trajectory")
    parser.add_argument("--release_pos_x", type=float, required=True)
    parser.add_argument("--release_pos_y", type=float, required=True)
    parser.add_argument("--release_pos_z", type=float, required=True)
    parser.add_argument("--ax", type=float, required=True)
    parser.add_argument("--ay", type=float, required=True)
    parser.add_argument("--az", type=float, required=True)
    parser.add_argument("--plate_x", type=float, required=True)
    parser.add_argument("--plate_z", type=float, required=True)
    parser.add_argument("--release_speed", type=float, required=True)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    traj = reconstruct_trajectory(
        args.release_pos_x, args.release_pos_y, args.release_pos_z,
        args.ax, args.ay, args.az,
        args.plate_x, args.plate_z,
        args.release_speed,
        dt=args.dt,
    )

    if args.output:
        traj.to_csv(args.output, index=False)
        print(f"Wrote {len(traj)} points to {args.output}")
    else:
        print(traj.to_string(index=False))

if __name__ == "__main__":
    main()
