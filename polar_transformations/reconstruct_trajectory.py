"""
Reconstruct a pitch trajectory under the constant-acceleration model.

Preferred path (when vx0/vy0/vz0 are available):
- Use Statcast's 9-parameter fit.
- Velocities are reported at y = 50 ft.
- Plate location (plate_x, plate_z) is at the front of the plate, y = 17/12 ft.
- Extrapolate back to release_pos_y for the release state if needed.
- Sample from release (t=0) to the front of the plate.

Fallback (when velocities are missing):
- Estimate flight time from release_speed and y-distance.
- Back-solve initial velocities so the path ends at plate_x/plate_z at y=0.

Output: DataFrame with columns t, x, y, z (feet), t=0 at release.
"""

import argparse
import numpy as np
import pandas as pd

# Front of home plate in Statcast coordinates
PLATE_FRONT_Y = 17.0 / 12.0  # ft

def _time_to_y(y_start, vy, ay, y_target):
    """
    Solve y(t) = y_start + vy*t + 0.5*ay*t**2 = y_target for t >= 0.
    Returns the smallest non-negative root, or None.
    """
    # 0.5*ay*t**2 + vy*t + (y_start - y_target) = 0
    a = 0.5 * ay
    b = vy
    c = y_start - y_target
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return None
        t = -c / b
        return t if t >= 0 else None
    disc = b**2 - 4*a*c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)
    candidates = [t for t in (t1, t2) if t >= 0]
    return min(candidates) if candidates else None

def reconstruct_trajectory(
    release_pos_x, release_pos_y, release_pos_z,
    ax, ay, az,
    plate_x, plate_z,
    release_speed_mph,
    vx0=None, vy0=None, vz0=None,
    dt=0.01,
):
    """
    Return a DataFrame with columns: t, x, y, z
    t=0 at release, sampled every dt seconds until the plate.

    If vx0/vy0/vz0 are provided, uses the Statcast 9-parameter fit
    and ends at the front of the plate (y = 17/12). Otherwise falls
    back to the speed-based estimate ending at y=0.
    """
    use_fit = all(v is not None and not (isinstance(v, float) and np.isnan(v))
                  for v in (vx0, vy0, vz0))

    if use_fit:
        # Velocities are given at y=50. Recover state at release by going backward.
        t_release_from_50 = _time_to_y(50.0, vy0, ay, release_pos_y)
        if t_release_from_50 is None:
            # Degenerate; fall back
            use_fit = False
        else:
            # State at release (going backward in time is negative dt)
            t_back = -t_release_from_50
            rx = 50.0 + vx0 * t_back + 0.5 * ax * t_back**2  # wait, x at y=50 is not 50
            # Actually we don't have x/z at y=50 directly. Recover from plate.
            # Better approach: start from known plate state and go backward, or
            # integrate forward from release using recovered release velocities.

            # Recover position at y=50 from plate (plate is at y=PLATE_FRONT_Y)
            t_50_to_plate = _time_to_y(50.0, vy0, ay, PLATE_FRONT_Y)
            if t_50_to_plate is None:
                use_fit = False
            else:
                x_at_50 = plate_x - (vx0 * t_50_to_plate + 0.5 * ax * t_50_to_plate**2)
                z_at_50 = plate_z - (vz0 * t_50_to_plate + 0.5 * az * t_50_to_plate**2)

                # Now go from y=50 back to release_pos_y
                t_back = -t_release_from_50
                vx_release = vx0 + ax * t_back
                vy_release = vy0 + ay * t_back
                vz_release = vz0 + az * t_back
                x_release = x_at_50 + vx0 * t_back + 0.5 * ax * t_back**2
                z_release = z_at_50 + vz0 * t_back + 0.5 * az * t_back**2
                # y_release should match release_pos_y by construction

                # Prefer the reported release positions for x/z if they are close;
                # otherwise use the extrapolated ones. For now use extrapolated for consistency
                # with the velocity field, but blend with reported if desired.
                # Use reported release positions as the starting point and the recovered velocities.
                x0, y0, z0 = release_pos_x, release_pos_y, release_pos_z
                vx0_use, vy0_use, vz0_use = vx_release, vy_release, vz_release

                t_flight = _time_to_y(y0, vy0_use, ay, PLATE_FRONT_Y)
                if t_flight is None:
                    use_fit = False

    if not use_fit:
        # Fallback
        t_flight = estimate_flight_time(release_speed_mph, release_pos_y)
        vx0_use, vy0_use, vz0_use = solve_initial_velocity(
            release_pos_x, release_pos_y, release_pos_z,
            ax, ay, az,
            plate_x, plate_z,
            t_flight,
        )
        y_end = 0.0
        x_end, z_end = plate_x, plate_z
        x0, y0, z0 = release_pos_x, release_pos_y, release_pos_z
    else:
        y_end = PLATE_FRONT_Y
        # End position from integration
        x_end = x0 + vx0_use * t_flight + 0.5 * ax * t_flight**2
        z_end = z0 + vz0_use * t_flight + 0.5 * az * t_flight**2

    times = np.arange(0, t_flight + dt/2, dt)
    xs = x0 + vx0_use * times + 0.5 * ax * times**2
    ys = y0 + vy0_use * times + 0.5 * ay * times**2
    zs = z0 + vz0_use * times + 0.5 * az * times**2

    # Snap last point to the intended end location
    if len(times) == 0 or abs(ys[-1] - y_end) > 0.05:
        times = np.append(times, t_flight)
        xs = np.append(xs, x_end)
        ys = np.append(ys, y_end)
        zs = np.append(zs, z_end)

    return pd.DataFrame({"t": times, "x": xs, "y": ys, "z": zs})

def estimate_flight_time(release_speed_mph, y0):
    speed_ft_s = release_speed_mph * 1.46667
    avg_speed = 0.95 * speed_ft_s
    return abs(y0) / avg_speed

def solve_initial_velocity(x0, y0, z0, ax, ay, az, xf, zf, t_flight):
    vx0 = (xf - x0 - 0.5 * ax * t_flight**2) / t_flight
    vy0 = (0.0 - y0 - 0.5 * ay * t_flight**2) / t_flight
    vz0 = (zf - z0 - 0.5 * az * t_flight**2) / t_flight
    return vx0, vy0, vz0

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
    parser.add_argument("--vx0", type=float, default=None)
    parser.add_argument("--vy0", type=float, default=None)
    parser.add_argument("--vz0", type=float, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    traj = reconstruct_trajectory(
        args.release_pos_x, args.release_pos_y, args.release_pos_z,
        args.ax, args.ay, args.az,
        args.plate_x, args.plate_z,
        args.release_speed,
        vx0=args.vx0, vy0=args.vy0, vz0=args.vz0,
        dt=args.dt,
    )

    if args.output:
        traj.to_csv(args.output, index=False)
        print(f"Wrote {len(traj)} points to {args.output}")
    else:
        print(traj.to_string(index=False))

if __name__ == "__main__":
    main()
