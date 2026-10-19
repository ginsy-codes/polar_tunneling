# Polar Tunneling

Repository for developing a pitch quality / efficacy model focused on trajectory path similarity as perceived by the hitter.

## Core Idea

Unlike traditional Stuff+ models (which primarily use static features like velocity, induced vertical/horizontal break, spin, release point, and differentials relative to a primary fastball), this approach centers on the **full pitch trajectory** expressed in **polar coordinates from the batter’s point of view**.

The goal is to quantify “path similarities” (how the angular path of a pitch appears to the hitter over time) and relate those similarities to outcomes (whiff, called strike, contact quality, run value, etc.). This is intended as a complementary signal to conventional Stuff models, isolating the perceptual / tunneling component of pitch effectiveness.

## Key Design Decisions

### Representation
- Reconstruct Cartesian trajectories from Statcast 9-parameter data (release position, velocity components, acceleration components).
- Transform into a batter-centric frame using a generic hitter eye height and lateral offset (for a global score). Later extensions can use batter-specific geometry.
- Convert to polar coordinates (angular position / velocity over time).
- **Absolute time parameterization** (not time-normalized to [0,1]). This preserves the real temporal scale the batter experiences.

### Decision Horizon / Truncation
- Hitters generally decide well before the ball reaches the plate.
- Use the pitcher’s primary fastball velocity (or time-to-plate) to define a decision-point cutoff (e.g., a fixed perceptual window or percentage of fastball flight time that safely precedes plate arrival).
- Secondary pitches are truncated or masked at this absolute-time horizon so comparisons stay within the relevant perceptual window.
- Velocity itself is **not** used as an explicit feature in the path model (to avoid redundancy with existing Stuff models); it is used only to set the horizon.

### Reference Trajectory
- The pitcher’s primary fastball (highest-usage four-seam / sinker / cutter, or highest average velocity) serves as the reference path.
- Similarity (e.g., via Dynamic Time Warping or other trajectory distances) is computed relative to this reference within the decision horizon.
- League-relative or other secondary references can be added as diagnostics.

### Data & Modeling Approach
- Source: Baseball Savant / Statcast pitch-level data (via `pybaseball` or equivalent).
- Expected volume: millions of pitches per season; ~15 key parameters stored per pitch (pandas / Parquet).
- Exploratory phase: correlations / mutual information between polar-derived path features (or similarity scores) and outcomes.
- Predictive phase: gradient-boosted models (LightGBM / XGBoost) or sequence models predicting expected run value or discrete outcomes, using path-similarity features. Standardized to a + scale (mean 100, SD ~10) for interpretability.
- Evaluation: out-of-sample predictive power, year-to-year stability, incremental lift over a classical Stuff baseline, pitcher-aware cross-validation to avoid leakage.

### Practical Notes
- Variable-length absolute-time series can be handled with a conservative max-time bound (tunable hyperparameter) if full variable-length arrays become computationally burdensome.
- Leakage control: reference fastballs should come from prior outings or held-out data, not the same at-bat/game when evaluating in-game.
- Parallel interpretable model (GAM or sparse linear) on key polar scalars recommended for communication.

## Status

Planning / early development. Direct operation on `main` until the project is more operational.

## Next Steps
- Define exact polar origin, angular conventions, and sampling rate.
- Prototype Cartesian → batter-frame → polar conversion.
- Engineer initial scalar path-similarity features and run exploratory correlations.
- Fit baseline models and measure lift vs. conventional features.
