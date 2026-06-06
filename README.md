# BESS Economic Dispatch Optimization Benchmark

A Battery Energy Storage System (BESS) dispatch optimization benchmark built on the
[BenchOpt](https://benchopt.github.io/) framework, targeting **German electricity markets**.

The goal is to structure optimization experiments in a modular and reproducible way, allowing
different solver algorithms to be benchmarked against the same problem formulations and datasets.

---

## Repository Structure

```
.
├── bench_objective.py          # Shared evaluation objective
├── bench_runner.py             # Custom runner (alternative to `benchopt run .`)
├── ESS_CONFIG.json             # Hardware config (inverter & battery parameters, PWL fits)
├── RUN_CONFIG.json             # Self-consumption scenario config (load, generation, prices)
├── DA_CONFIG.json              # Day-ahead trading scenario config (hourly prices)
├── ID_CONFIG.json              # Intra-day trading scenario config (15-min prices)
├── datasets/
│   ├── baseline_dataset.py
│   ├── linear_model_dataset.py
│   ├── pwl_milp_dataset.py
│   ├── da_trading_dataset.py
│   └── id_trading_dataset.py
└── solvers/
    ├── run_basline_dispatch.py
    ├── run_linear_milp_dispatch.py
    ├── run_pwl_milp_dispatch.py
    ├── run_da_milp_dispatch.py
    ├── run_da_pwl_milp_dispatch.py
    ├── run_id_milp_dispatch.py
    └── run_id_pwl_milp_dispatch.py
```

| Component | Purpose |
|---|---|
| `datasets/` | Loads hardware config, ESS initial state, and time-series forecasts / price signals |
| `bench_objective.py` | Defines the evaluation metric; branches on scenario type (self-consumption vs. trading) |
| `solvers/` | Implements dispatch algorithms; each solver is self-contained |

---

## Scenarios

### Self-consumption (prosumer)
Minimise net energy cost for a behind-the-meter BESS co-located with a load and renewable
generation. Grid sales are capped at renewable output and paid at the feed-in tariff.

**Config files:** `ESS_CONFIG.json` + `RUN_CONFIG.json` (15-min resolution, 96 time steps)

### Day-ahead (DA) trading — EPEX SPOT DE/AT
Pure price arbitrage against the hourly day-ahead auction. The BESS charges when prices are
low and discharges at price peaks; all grid exchanges are at the spot price.

**Config file:** `ESS_CONFIG.json` + `DA_CONFIG.json` (60-min resolution, 24 time steps)

### Intra-day (ID) trading — EPEX SPOT continuous
Same arbitrage logic on 15-minute continuous intra-day products. Higher temporal resolution
creates more arbitrage cycles but also a larger MILP.

**Config file:** `ESS_CONFIG.json` + `ID_CONFIG.json` (15-min resolution, 96 time steps)

---

## Solvers

| Solver | Efficiency model | Scenario |
|---|---|---|
| Baseline | Rule-based (no optimisation) | Self-consumption |
| Linear MILP | Constant charge/discharge efficiency | Self-consumption |
| PWL MILP | Piecewise-linear inverter + battery loss | Self-consumption |
| DA MILP | Constant efficiency | DA trading |
| DA PWL MILP | Piecewise-linear loss | DA trading |
| ID MILP | Constant efficiency | ID trading |
| ID PWL MILP | Piecewise-linear loss | ID trading |

The **PWL efficiency model** approximates quadratic inverter and battery loss polynomials as
piecewise-linear segments using SOS2-style binary encoding. A greedy price-threshold heuristic
provides HiGHS with a warm-start incumbent, reducing solve time significantly.

All MILP solvers use [HiGHS](https://highs.dev/) via Pyomo's `appsi_highs` interface and
terminate at a 1 % MIP gap.

---

## Execution Flow

```
Dataset.get_data()
        ↓
Objective.set_data()      ← scenario tag flows in here
        ↓
Objective.get_objective()
        ↓
Solver.set_objective(...)
        ↓
Solver.run()
        ↓
Solver.get_result()
        ↓
Objective.evaluate_result()   ← branches on scenario type
```

---

## Running the Benchmark

### Custom runner (recommended)

```bash
python bench_runner.py --config-path "path/to/config/dir" --solver all
```

Run a specific solver:

```bash
python bench_runner.py --config-path "path/to/config/dir" --solver da_pwl_milp
```

Available solver keys: `baseline`, `linear_milp`, `pwl_milp`, `da_milp`, `da_pwl_milp`,
`id_milp`, `id_pwl_milp`.

### BenchOpt native runner

```bash
benchopt run .
```

### Setup

```bash
pip install -r requirements.txt
```

---

## Dependencies

- [BenchOpt](https://benchopt.github.io/) — reproducible benchmarking framework
- [Pyomo](https://www.pyomo.org/) — algebraic modelling language for optimisation
- [HiGHS](https://highs.dev/) — open-source MILP solver (via `highspy`)

---

## Benchmark Results (teaser)

Results below are for the hardware configuration in `ESS_CONFIG.json` (2 × BESS units,
30 kW / 70 kWh and 20 kW / 100 kWh) and the sample price/forecast data shipped in the repo.
Metric is the solver objective value (negative = net revenue / cost saving).

### Self-consumption

| Solver | Metric | Solve time |
|---|---|---|
| Baseline | -4 523 | < 1 s |
| Linear MILP | -4 988 | < 1 s |
| **PWL MILP** | **-4 989** | ~38 min |

PWL recovers only marginal additional value over linear for self-consumption — the loss model
matters less when the objective is dominated by a fixed load profile.

### DA trading (24 h, hourly)

| Solver | Metric | Solve time |
|---|---|---|
| DA MILP | -1 986 | < 1 s |
| **DA PWL MILP** | **-2 336** | 3 s |

### ID trading (24 h, 15-min)

| Solver | Metric | Solve time |
|---|---|---|
| ID MILP | -2 007 | < 1 s |
| **ID PWL MILP** | **-2 361** | 12 s |

**Key observations:**

- The PWL solver finds **~17–18 % better solutions than linear** for trading scenarios. Accurate
  loss modelling prevents over-trading — the linear model underestimates round-trip losses and
  therefore schedules more cycles than are actually profitable.
- Trading PWL solvers are **dramatically faster** than the self-consumption PWL (seconds vs.
  minutes) because the symmetric SoC initialisation at 50 % gives the warm-start a high-quality
  incumbent immediately.
- ID slightly outperforms DA in absolute terms — finer granularity exposes more intra-day price
  spread for the optimiser to exploit.
