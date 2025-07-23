import uuid
import numpy as np
import json
from pathlib import Path
import argparse
from typing import List, Dict, Any
from tinyphysics import run_rollout

class ParameterSet:
    """Represents a set of PID gains and its metadata in a tournament."""
    def __init__(self, gains: Dict[str, List[float]]):
        self.id = str(uuid.uuid4())
        self.gains = gains
        self.stats: Dict[str, Any] = {}
        self.rounds_survived: int = 0
        self.status: str = "active"

def initialize_population(n: int) -> List[ParameterSet]:
    """Generate an initial population of random PID gain ParameterSets."""
    population: List[ParameterSet] = []
    for _ in range(n):
        gains = {
            'low':  list(np.random.uniform([0.25, 0.01, -0.25], [0.6, 0.12, -0.05])),
            'high': list(np.random.uniform([0.15, 0.005, -0.15], [0.4, 0.08, -0.03])),
            'dyn':  list(np.random.uniform([0.3, 0.02, -0.3], [0.8, 0.15, -0.08])),
        }
        population.append(ParameterSet(gains))
    return population

def select_elites(population: List[ParameterSet], elite_pct: float) -> List[ParameterSet]:
    """Select top X% elites by avg_total_cost; eliminate others."""
    if not population:
        return []
    sorted_pop = sorted(population, key=lambda ps: ps.stats.get('avg_total_cost', float('inf')))
    elite_count = max(int(len(population) * elite_pct), 1)
    elites = sorted_pop[:elite_count]
    for ps in population:
        if ps in elites:
            ps.status = "active"
            ps.rounds_survived += 1
        else:
            ps.status = "eliminated"
    return elites

def revival_lottery(archive: List[ParameterSet], revive_pct: float, pop_size: int) -> List[ParameterSet]:
    """Revive eliminated ParameterSets weighted by rounds_survived."""
    eliminated = [ps for ps in archive if ps.status == "eliminated"]
    if not eliminated:
        return []
    count = max(min(int(pop_size * revive_pct), len(eliminated)), 1)
    weights = np.array([ps.rounds_survived for ps in eliminated], dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()
    chosen = np.random.choice(len(eliminated), size=count, replace=False, p=probs)
    revived: List[ParameterSet] = []
    for idx in chosen:
        ps = eliminated[idx]
        ps.status = "active"
        ps.rounds_survived += 1
        revived.append(ps)
    return revived

def generate_new(count: int, around: ParameterSet, perturb_scale: float = 0.05) -> List[ParameterSet]:
    """Generate new ParameterSets by mutating gains around a base ParameterSet."""
    new_sets: List[ParameterSet] = []
    for _ in range(count):
        from typing import Dict, List
        noisy_gains: Dict[str, List[float]] = {}
        for mode, gains in around.gains.items():
            noisy_gains[mode] = [float(g) + float(np.random.normal(0, perturb_scale)) for g in gains]
        new_sets.append(ParameterSet(noisy_gains))
    return new_sets

def _make_temp_controller(ps: ParameterSet) -> str:
    """Write a temporary controller file and return its module name."""
    gains = ps.gains
    code_lines = [
        "from . import BaseController",
        "",
        "class SpecializedPID:",
        "    def __init__(self, p, i, d):",
        "        self.p, self.i, self.d = p, i, d",
        "        self.error_integral = 0",
        "        self.prev_error = 0",
        "",
        "    def update(self, error):",
        "        self.error_integral += error",
        "        diff = error - self.prev_error",
        "        self.prev_error = error",
        "        return self.p * error + self.i * self.error_integral + self.d * diff",
        "",
        "class Controller(BaseController):",
        "    def __init__(self):",
        f"        self.low_speed_pid = SpecializedPID({gains['low'][0]}, {gains['low'][1]}, {gains['low'][2]})",
        f"        self.high_speed_pid = SpecializedPID({gains['high'][0]}, {gains['high'][1]}, {gains['high'][2]})",
        f"        self.dynamic_pid = SpecializedPID({gains['dyn'][0]}, {gains['dyn'][1]}, {gains['dyn'][2]})",
        "",
        "    def update(self, target_lataccel, current_lataccel, state, future_plan):",
        "        err = target_lataccel - current_lataccel",
        "        u1 = self.low_speed_pid.update(err)",
        "        u2 = self.high_speed_pid.update(err)",
        "        u3 = self.dynamic_pid.update(err)",
        "        return u1 + u2 + u3",
    ]
    controllers_dir = Path(__file__).parent.parent / "controllers"
    module_name = f"temp_{ps.id.replace('-', '')}"
    file_path = controllers_dir / f"{module_name}.py"
    with open(file_path, "w") as f:
        f.write("\n".join(code_lines))
    return module_name

def cleanup_controllers(prefix: str = "temp_") -> None:
    """Remove temporary controller files."""
    controllers_dir = Path(__file__).parent.parent / "controllers"
    for path in controllers_dir.glob(f"{prefix}*.py"):
        try:
            path.unlink()
        except:
            pass

def evaluate(ps: ParameterSet, data_files: List[str], model_path: str, max_files: int) -> None:
    """Evaluate a ParameterSet and fill its stats."""
    total_costs: List[float] = []
    mod = _make_temp_controller(ps)
    try:
        for file in data_files[:max_files]:
            cost, _, _ = run_rollout(file, mod, model_path, debug=False)
            total_costs.append(cost["total_cost"])
    finally:
        cleanup_controllers(prefix=f"temp_{ps.id.replace('-', '')}")
    if total_costs:
        arr = np.array(total_costs)
        ps.stats = {
            "avg_total_cost": float(arr.mean()),
            "std_cost": float(arr.std()),
            "min_cost": float(arr.min()),
            "max_cost": float(arr.max()),
            "num_files": len(arr),
        }
    else:
        ps.stats = {"avg_total_cost": float("inf"), "num_files": 0}

def run_tournament(data_files: List[str], model_path: str, rounds: int,
                   pop_size: int, elite_pct: float, revive_pct: float,
                   max_files: int, perturb_scale: float) -> None:
    """Execute the tournament optimization loop."""
    population = initialize_population(pop_size)
    archive: List[ParameterSet] = population.copy()
    summary: List[Dict[str, Any]] = []
    for r in range(1, rounds + 1):
        for ps in population:
            evaluate(ps, data_files, model_path, max_files)
        elites = select_elites(population, elite_pct)
        revived = revival_lottery(archive, revive_pct, pop_size)
        best = min(elites, key=lambda ps: ps.stats["avg_total_cost"])
        new_count = pop_size - len(elites) - len(revived)
        new_sets = generate_new(new_count, best, perturb_scale)
        archive.extend(new_sets + revived)
        population = elites + revived + new_sets
        summary.append({
            "round": r,
            "elites": [ps.id for ps in elites],
            "revived": [ps.id for ps in revived],
            "new": [ps.id for ps in new_sets],
            "best_cost": best.stats["avg_total_cost"],
        })
    # Ensure plans directory exists and save per-round summary
    plans_dir = Path(__file__).parent.parent / "plans"
    plans_dir.mkdir(exist_ok=True)
    (plans_dir / "tournament_progress.json").write_text(json.dumps({"tournament_summary": summary}, indent=2))
    # Save full archive of ParameterSets
    archive_list: List[Dict[str, Any]] = []
    for ps in archive:
        archive_list.append({
            "id": ps.id,
            "gains": ps.gains,
            "stats": ps.stats,
            "rounds_survived": ps.rounds_survived,
            "status": ps.status
        })
    (plans_dir / "tournament_archive.json").write_text(json.dumps({"archive": archive_list}, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--elite_pct", type=float, default=0.3)
    parser.add_argument("--revive_pct", type=float, default=0.2)
    parser.add_argument("--max_files", type=int, default=5)
    parser.add_argument("--perturb_scale", type=float, default=0.05)
    args = parser.parse_args()
    args_dict = vars(args)
    if "perturb_scale" not in args_dict:
        args.perturb_scale = 0.05

    files = [str(f) for f in sorted(Path(args.data_path).glob("*.csv"))]
    run_tournament(files, args.model_path, args.rounds, args.pop_size,
                   args.elite_pct, args.revive_pct, args.max_files, args.perturb_scale)
    print("Tournament completed. Progress saved to tournament_progress.json")

if __name__ == "__main__":
    main()
