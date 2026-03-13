import argparse
import json
from pathlib import Path

from solvers.run_basline_dispatch import BaselineSolver
from solvers.run_linear_milp_dispatch import LinearMILPSolver
from bench_objective import Objective
from datasets.baseline_dataset import BaselineDataset
from datasets.linear_model_dataset import LinearMILPDataset

# ----------------------------------------------------
# Registry
# ----------------------------------------------------

DATASETS = {
    "baseline": BaselineDataset,
    "linear_milp": LinearMILPDataset,
}

SOLVERS = {
    "baseline": BaselineSolver,
    "linear_milp": LinearMILPSolver,
}


# ----------------------------------------------------
# BenchOpt execution pipeline
# ----------------------------------------------------
def run_one(dataset_cls, solver_cls):

    dataset = dataset_cls()
    objective = Objective()
    solver = solver_cls()

    # Dataset → data
    data = dataset.get_data()

    # Objective ← data
    objective.set_data(**data)

    # Solver ← objective
    solver.set_objective(**objective.get_objective())

    # Solve
    solver.run()

    # Get solver result
    result = solver.get_result()

    # Evaluate
    metrics = objective.evaluate_result(**result)
    print(f"Completed with metric: {metrics} cents/Wh")
    return {"solution": result, "metrics": metrics}


# ----------------------------------------------------
# Save results
# ----------------------------------------------------
def save_output(output, solver_name):
    output_serializable = {
        "solution": {k: v.tolist() for k, v in output["solution"].items()},
        "metrics": output["metrics"],
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"{solver_name}.json"

    with open(output_file, "w") as f:
        json.dump(output_serializable, f, indent=4)

    print(f"Saved result → {output_file}")


# ----------------------------------------------------
# CLI
# ----------------------------------------------------
def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--solver", default="all", choices=["all"] + list(SOLVERS.keys())
    )
    args = parser.parse_args()

    solvers = SOLVERS if args.solver == "all" else {args.solver: SOLVERS[args.solver]}

    for sname, scls in solvers.items():
        print(f"\nRunning solver={sname}")

        output = run_one(DATASETS[sname], scls)
        save_output(output, sname)


if __name__ == "__main__":
    run_main()
