# F1 Kinetic

Probabilistic race forecasting and strategy optimisation engine built on 70+ years of Formula 1 telemetry (1950–2023).

![F1 Kinetic](https://r4.wallpaperflare.com/wallpaper/422/882/465/ineos-iwc-lewis-hamilton-mercedes-amg-petronas-formula-1-hd-wallpaper-21cf3bc48a75fd63d7fa5cca7e1ca608.jpg)

---

## Overview

| Component | Description |
|-----------|-------------|
| **Probabilistic Forecasting Engine** | Bayesian Inference + Feature Engineering to predict race outcomes; achieves a **15% improvement** in outcome reliability over the grid-position baseline |
| **Monte Carlo Simulation Framework** | 10,000+ iterations per race across 23+ Grand Prix to optimise fuel-load and pit-stop strategy through multi-dimensional data modelling |
| **Exploratory Data Analysis** | Historical win counts, speed trends, and grid-position correlation across the full 70-year dataset |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [`probabilistic_forecasting.ipynb`](probabilistic_forecasting.ipynb) | Bayesian Beta-Binomial prior, 11-feature engineering pipeline, Random Forest ensemble, reliability benchmarking |
| [`monte_carlo_simulation.ipynb`](monte_carlo_simulation.ipynb) | Lap-by-lap race simulator, fuel-load sensitivity analysis, pit-stop window optimisation for 23 GPs |
| [`F1_proj.ipynb`](F1_proj.ipynb) | Exploratory data analysis — driver/constructor win counts, speed facet grids, grid-vs-finish regression |

---

## Probabilistic Forecasting Engine

Predicts podium finishers using a **Beta-Binomial Bayesian model** updated with grid-position likelihood ratios, then ensembled with a Random Forest trained on 11 engineered features:

```
career_win_rate  |  career_podium_rate  |  recent_form_5
circuit_avg_position  |  constructor_win_rate  |  constructor_recent_points
grid_position  |  dnf_rate  |  experience  |  podium_rate  |  quali_time_delta
```

**Result:** ~15% improvement in per-race podium hit-rate over the grid-position-only baseline (evaluated on 2018–2019 hold-out seasons).

---

## Monte Carlo Simulation Framework

Each iteration samples from physically-grounded distributions:

| Variable | Distribution |
|----------|-------------|
| Fuel load | Uniform(85, 110) kg |
| Pit stop duration | Normal(25 s, 2 s) |
| Lap time noise | Normal(μ_circuit, σ_circuit) |
| Fuel burn penalty | 0.03 s/lap per kg remaining |
| Safety car events | Poisson(λ = 0.25) per race |
| DNF probability | Per-circuit historical rate |

**10,000 iterations × 23 Grand Prix = 230,000+ simulated race scenarios**

Outputs per GP: optimal fuel load (± confidence), best pit-stop strategy, pit-window lap ranges, and 1-stop vs 2-stop time delta.

---

## Tech Stack

- **Python** — core language
- **Pandas** — data ingestion, merging, and rolling feature engineering
- **Scikit-Learn** — StandardScaler, LogisticRegression, RandomForestClassifier
- **SciPy** — Beta distribution (Bayesian inference), statistical utilities
- **NumPy** — Monte Carlo sampling, vectorised simulation
- **Matplotlib / Seaborn** — visualisations

---

## Dataset

Formula 1 World Championship data, 1950–2023 (Ergast format).

| File | Contents |
|------|----------|
| `results.csv` | Race results, finishing positions, fastest laps |
| `races.csv` | Race calendar, circuits, dates |
| `drivers.csv` | Driver registry |
| `constructors.csv` | Constructor/team registry |
| `qualifying.csv` | Q1/Q2/Q3 session times |
| `circuits.csv` | Circuit metadata |

[Dataset source — Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

---

## Screenshots

![Forecasting Results](imgs/Screenshot%202023-06-29%20at%208.14.56%20PM.png)

![Speed Analysis](imgs/Screenshot%202023-06-29%20at%208.15.07%20PM.png)
