"""GPU-optimized tournament optimizer using proven blended 2-PID architecture"""
import uuid
import numpy as np
import json
from pathlib import Path
import argparse
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)
from utils.logging import print_banner, print_params, print_summary, tqdm, EMOJI_PARTY, EMOJI_OK, EMOJI_TROPHY

# Define where temporary controllers should live
base_dir = Path(__file__).parent.parent
controllers_dir = base_dir / "controllers"
target_dir = controllers_dir
from typing import List, Dict, Any, Optional
import sys
import os
import logging
import tempfile


# Add the parent directory to path to find tinyphysics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tinyphysics_custom import run_rollout, TinyPhysicsModel
from optimization import generate_blended_controller


def cleanup_artifacts() -> None:
    """Remove leftover temporary controllers and blender models."""
    base_dir = Path(__file__).parent.parent
    controllers_dir = base_dir / "controllers"
    models_dir = base_dir / "models"

    for path in controllers_dir.glob("temp_*.py"):
        try:
            path.unlink()
        except OSError:
            pass

    for path in models_dir.glob("blender_*.onnx"):
        try:
            path.unlink()
        except OSError:
            pass


cleanup_artifacts()

class ParameterSet:
    """Represents a set of blended 2-PID gains and metadata in a tournament."""
    def __init__(self, low_gains: List[float], high_gains: List[float]):
        self.id = str(uuid.uuid4())
        self.low_gains = low_gains
        self.high_gains = high_gains
        self.stats: Dict[str, Any] = {}
        self.rounds_survived: int = 0
        self.status: str = "active"

def load_champions_from_file(file_path: str, n: int) -> List[Dict]:
    """Load champions from either Stage 1 or Tournament archive format"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    if 'all_results' in data:
        candidates = data['all_results']
        champions = sorted(
            [c for c in candidates if c.get('avg_total_cost', float('inf')) != float('inf')],
            key=lambda x: x.get('avg_total_cost', float('inf'))
        )[:n//2]
    elif 'archive' in data:
        candidates = data['archive']
        champions = sorted(
            [c for c in candidates if c.get('stats', {}).get('avg_total_cost', float('inf')) != float('inf')],
            key=lambda x: x.get('stats', {}).get('avg_total_cost', float('inf'))
        )[:n//2]
    else:
        raise ValueError(f"Unknown format in {file_path}. Expected 'all_results' or 'archive' key.")
    return champions

def extract_gains_from_champion(champion: Dict) -> tuple:
    """Extract low_gains and high_gains from either format"""
    return champion['low_gains'], champion['high_gains']

def _make_temp_controller(ps: ParameterSet) -> str:
    """Generate temporary controller file for evaluation.

    Raises:
        RuntimeError: If the controller file cannot be written.
    """
    content = generate_blended_controller(ps.low_gains, ps.high_gains)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    module_name = f"temp_{ps.id.replace('-', '')}"

    file_path = controllers_dir / f"{module_name}.py"
    try:
        with open(file_path, "w") as f:
            f.write(content)
    except OSError as e:
        raise RuntimeError(f"Failed to write temporary controller {file_path}: {e}") from e
    return module_name

def cleanup_controllers(prefix: str = "temp_") -> None:
    """Remove temporary controller files."""
    controllers_dir = Path(__file__).parent.parent / "controllers"
    for path in controllers_dir.glob(f"{prefix}*.py"):
        try:
            path.unlink()
        except:
            pass

def evaluate(ps: ParameterSet, data_files: List[str], model: TinyPhysicsModel, max_files: int,
             rng: Optional[np.random.Generator] = None) -> None:
    """Evaluate a ParameterSet and populate its stats."""
    import sys

    module_name: Optional[str] = None
    full_module: Optional[str] = None
    total_costs: List[float] = []
    try:
        module_name = _make_temp_controller(ps)
        full_module = f"controllers.{module_name}"
        if full_module in sys.modules:
            del sys.modules[full_module]

        # Sample a random subset of files for each evaluation
        if rng is not None:
            selected = rng.choice(data_files, size=min(max_files, len(data_files)), replace=False)
        else:
            selected = data_files[:max_files]

        for file in selected:
            cost, _, _ = run_rollout(file, module_name, model, debug=False)
            total_costs.append(cost["total_cost"])
    except Exception as e:
        logging.error("Evaluation failed for %s: %s", ps.id, e)
    finally:
        cleanup_controllers(prefix=f"temp_{ps.id.replace('-', '')}")
        if full_module and full_module in sys.modules:
            del sys.modules[full_module]

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

def initialize_population(
    n: int,
    seed_from_archive: Optional[str] = None,
    init_low_min: List[float] = [0.25, 0.01, -0.25],
    init_low_max: List[float] = [0.6, 0.12, -0.05],
    init_high_min: List[float] = [0.15, 0.005, -0.15],
    init_high_max: List[float] = [0.4, 0.08, -0.03],
    init_seed: Optional[int] = None
) -> List[ParameterSet]:
    population: List[ParameterSet] = []
    if seed_from_archive and Path(seed_from_archive).exists():
        try:
            champions = load_champions_from_file(seed_from_archive, n)
            for i, champ in enumerate(champions):
                low, high = extract_gains_from_champion(champ)
                ps = ParameterSet(low, high)
                ps.id = f"champion_{i}"
                population.append(ps)
        except:
            pass
    remaining = n - len(population)
    rng = np.random.default_rng(init_seed)
    for _ in range(remaining):
        low = list(rng.uniform(init_low_min, init_low_max))
        high = list(rng.uniform(init_high_min, init_high_max))
        low = [round(x, 3) for x in low]
        high = [round(x, 3) for x in high]
        population.append(ParameterSet(low, high))
    return population

def select_elites(population: List[ParameterSet], elite_pct: float) -> List[ParameterSet]:
    sorted_pop = sorted(population, key=lambda ps: ps.stats.get("avg_total_cost", float("inf")))
    count = max(int(len(population) * elite_pct), 1)
    elites = sorted_pop[:count]
    for ps in population:
        ps.status = "active" if ps in elites else "eliminated"
    return elites

def revival_lottery(archive: List[ParameterSet], revive_pct: float, pop_size: int) -> List[ParameterSet]:
    eliminated = [ps for ps in archive if ps.status == "eliminated"]
    if not eliminated:
        return []
    weights = np.array([ps.rounds_survived + 1 for ps in eliminated])
    count = max(min(int(pop_size * revive_pct), len(eliminated)), 1)
    idx = np.random.choice(len(eliminated), size=count, replace=False, p=weights/weights.sum())
    revived = [eliminated[i] for i in idx]
    for ps in revived:
        ps.status = "active"
    return revived

def generate_new(count: int, best: ParameterSet, perturb_scale: float) -> List[ParameterSet]:
    new_sets: List[ParameterSet] = []
    for _ in range(count):
        new_low = [
            np.clip(best.low_gains[0] + np.random.normal(0, perturb_scale), 0.25, 0.6),
            np.clip(best.low_gains[1] + np.random.normal(0, perturb_scale), 0.01, 0.12),
            np.clip(best.low_gains[2] + np.random.normal(0, perturb_scale), -0.25, -0.05)
        ]
        new_high = [
            np.clip(best.high_gains[0] + np.random.normal(0, perturb_scale), 0.15, 0.4),
            np.clip(best.high_gains[1] + np.random.normal(0, perturb_scale), 0.005, 0.08),
            np.clip(best.high_gains[2] + np.random.normal(0, perturb_scale), -0.15, -0.03)
        ]
        new_sets.append(ParameterSet([round(x,3) for x in new_low], [round(x,3) for x in new_high]))
    return new_sets

def run_tournament(
    data_files: List[str],
    model_path: str,
    rounds: int,
    pop_size: int,
    elite_pct: float,
    revive_pct: float,
    max_files: int,
    perturb_scale: float,
    seed_from_archive: Optional[str],
    init_low_min: List[float],
    init_low_max: List[float],
    init_high_min: List[float],
    init_high_max: List[float],
    init_seed: Optional[int],
    data_seed: Optional[int] = None
) -> None:
    model = TinyPhysicsModel(model_path, debug=False)
    
    # Create RNG for data sampling
    rng = np.random.default_rng(data_seed)
    
    # Shuffle data files once at the start
    data_files_copy = data_files.copy()
    rng.shuffle(data_files_copy)
    
    population = initialize_population(
        pop_size,
        seed_from_archive,
        init_low_min,
        init_low_max,
        init_high_min,
        init_high_max,
        init_seed
    )
    archive: List[ParameterSet] = population.copy()
    best_overall_cost = float("inf")
    best_overall_ps = None

    for r in range(1, rounds + 1):
        print(f"\n--- Round {r}/{rounds} ---", flush=True)
        
        # Evaluate population with a progress bar
        for ps in tqdm(population, desc=f"Round {r} Evaluation", unit="ps"):
            if ps.stats.get("avg_total_cost", float("inf")) == float("inf"):
                evaluate(ps, data_files_copy, model, max_files, rng)

        # Sort population by performance
        population.sort(key=lambda ps: ps.stats.get("avg_total_cost", float("inf")))
        
        round_best_ps = population[0]
        round_best_cost = round_best_ps.stats.get("avg_total_cost", float("inf"))

        if best_overall_ps is None or round_best_cost < best_overall_cost:
            best_overall_cost = round_best_cost
            best_overall_ps = round_best_ps
            print(f"{EMOJI_PARTY} New best cost in round {r}: {best_overall_cost:.2f} (ID: {best_overall_ps.id})", flush=True)

        print_summary(f"Round {r} Summary", {
            "Best Cost": f"{round_best_cost:.2f}",
            "Avg Cost": f"{np.mean([p.stats.get('avg_total_cost', 0) for p in population]):.2f}",
        })

        if r < rounds:
            elites = select_elites(population, elite_pct)
            revived = revival_lottery(archive, revive_pct, pop_size)
            
            best_for_mutation = elites[0] if elites else population[0]
            new_sets = generate_new(pop_size - len(elites) - len(revived), best_for_mutation, perturb_scale)
            
            archive.extend(new_sets)
            population = elites + revived + new_sets
            print(f"➡️  Next generation created with {len(elites)} elites, {len(revived)} revived, {len(new_sets)} new.", flush=True)

    if best_overall_ps:
        print(f"\n{EMOJI_TROPHY} Tournament Complete {EMOJI_TROPHY}")
        print_summary("Overall Best Performer", {
            "ID": best_overall_ps.id,
            "Cost": f"{best_overall_cost:.2f}",
            "Low Gains": best_overall_ps.low_gains,
            "High Gains": best_overall_ps.high_gains,
        })

    plans_dir = Path(__file__).parent.parent / "plans"
    plans_dir.mkdir(exist_ok=True)
    out = [{
        "id": ps.id,
        "low_gains": ps.low_gains,
        "high_gains": ps.high_gains,
        "stats": ps.stats
    } for ps in archive]
    (plans_dir / "tournament_archive.json").write_text(json.dumps({"archive": out, "best_cost": best_overall_cost if best_overall_ps else None}, indent=2))

def main():
    cleanup_artifacts()
    parser = argparse.ArgumentParser(description="Tournament optimizer for blended 2-PID controllers")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--elite_pct", type=float, default=0.2)
    parser.add_argument("--revive_pct", type=float, default=0.1)
    parser.add_argument("--max_files", type=int, default=25)
    parser.add_argument("--perturb_scale", type=float, default=0.05)
    parser.add_argument("--seed_from_archive", type=str, default=None)
    parser.add_argument("--data-seed", "--data_seed", dest="data_seed", type=int, default=None,
                        help="Seed for shuffling data files (omit for non-deterministic)")
    parser.add_argument(
        "--init_low_min", type=lambda s: list(map(float, s.split(","))),
        default=[0.25, 0.01, -0.25]
    )
    parser.add_argument(
        "--init_low_max", type=lambda s: list(map(float, s.split(","))),
        default=[0.6, 0.12, -0.05]
    )
    parser.add_argument(
        "--init_high_min", type=lambda s: list(map(float, s.split(","))),
        default=[0.15, 0.005, -0.15]
    )
    parser.add_argument(
        "--init_high_max", type=lambda s: list(map(float, s.split(","))),
        default=[0.4, 0.08, -0.03]
    )
    parser.add_argument("--init_seed", type=int, default=None)
    args = parser.parse_args()
    # Stage banner and key parameters
    stage_num = 3 if args.seed_from_archive and "plans/tournament_archive.json" in args.seed_from_archive else 2
    stage_name = "PID Tournament #2" if stage_num == 3 else "PID Tournament #1"
    print_banner(stage_num, stage_name)
    print_params({
        "rounds": args.rounds,
        "population size": args.pop_size,
        "elite %": args.elite_pct,
        "revive %": args.revive_pct,
        "max files": args.max_files,
        "perturb scale": args.perturb_scale
    })

    base_dir = Path(__file__).parent.parent
    data_files = [str(f) for f in sorted((base_dir / "data").glob("*.csv"))]
    run_tournament(
        data_files,
        str(base_dir / "models" / "tinyphysics.onnx"),
        args.rounds,
        args.pop_size,
        args.elite_pct,
        args.revive_pct,
        args.max_files,
        args.perturb_scale,
        args.seed_from_archive,
        args.init_low_min,
        args.init_low_max,
        args.init_high_min,
        args.init_high_max,
        args.init_seed,
        args.data_seed
    )

if __name__ == "__main__":
    main()
