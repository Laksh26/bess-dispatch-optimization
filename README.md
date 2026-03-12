# BESS economic dispatch optimization framework

This repository implements a Battery Energy Storage System (BESS) dispatch optimization benchmark using the BenchOpt 
framework.

The goal is to structure optimization experiments in a modular and reproducible way, allowing different algorithms 
(solvers) to be tested on the same problem formulation and datasets.

## Repository Structure

The benchmark follows the standard **BenchOpt architecture**:

```
.
├── objective.py
├── datasets/
│   └── linear_model_dataset.py
└── solvers/
    └── linear_milp_solver.py
```

Each component has a specific responsibility.

| Component      | Purpose                                                                                                                             |
| -------------- |-------------------------------------------------------------------------------------------------------------------------------------|
| `datasets/`    | Defines input data and parameters. <br/> Here the connection to external source for forecast and price signals could be established |
| `objective.py` | Defines the mathematical optimization objective to be solved for                                                                    |
| `solvers/`     | Implements algorithms that solve the problem                                                                                        |

This separation allows one to easily:

* test multiple solvers
* compare algorithms
* run multiple datasets
* maintain reproducibility

---

# Execution Flow

When running

```
benchopt run .
```

BenchOpt executes the following pipeline:

```
Dataset.get_data()
        ↓
Objective.set_data()
        ↓
Objective.get_objective()
        ↓
Solver.set_objective(...)
        ↓
Solver.run()
        ↓
Solver.get_result()
        ↓
Objective.evaluate_result()
```

## Dependencies

This project uses the following main packages:

- [BenchOpt](https://benchopt.github.io/) – for running reproducible benchmarks of optimization algorithms.
- [Pyomo](https://www.pyomo.org/) – for modeling and solving the optimization problem.

BenchOpt is used to structure the benchmarking experiments, while the optimization model and solver interface are implemented using Pyomo.
