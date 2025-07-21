"""
Tournament-based evolutionary optimizer for ensemble PID controllers.
"""
import argparse
import os
import csv
import uuid
from pathlib import Path
from datetime import datetime
import random
import numpy as np  # type: ignore

from optimization.comprehensive_optimizer import ComprehensiveOptimizer

# Compute base paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = str(BASE_DIR / "models" / "tinyphysics.onnx")


class ParameterSet:
    """Represents a candidate parameter configuration."""
    def __init__(self, gains: dict):
        self.id = str(uuid.uuid4())
        self.gains = gains  # {'low': [...], 'high': [...], 'dyn': [...]}
        self.stats = {}     # to be filled: avg_total_cost, std_cost, min_cost, max_cost
        self.rounds_survived = 0
        self.status = "active"


def initialize_population(pop_size: int) -> list:
    """Generate initial population via existing search-space logic."""
    optimizer = ComprehensiveOptimizer(MODEL_PATH)
    combos = optimizer.define_comprehensive_search_space(pop_size)
    population = []
    for low, high, dyn in combos:
        gains = {'low': low, 'high': high, 'dyn': dyn}
        population.append(ParameterSet(gains))
    return population


def evaluate(ps: ParameterSet, data_files: list, files_per_round: int):
    """Evaluate a single ParameterSet on a subset of data files."""
    optimizer = ComprehensiveOptimizer(MODEL_PATH)
    # test_controller_combination returns stats dict
    stats = optimizer.test_controller_combination(
        data_files, 
        ps.gains['low'], 
        ps.gains['high'], 
        ps.gains['dyn'], 
        max_files=files_per_round
    )
    ps.stats = stats
    return ps


def select_elites(population: list, elite_pct: float) -> list:
    """Select top fraction of population as elites."""
    sorted_pop = sorted(
        population, 
        key=lambda ps: ps.stats.get('avg_total_cost', float('inf'))
    )
    k = max(1, int(len(sorted_pop) * elite_pct))
    elites = sorted_pop[:k]
    for ps in elites:
        ps.rounds_survived += 1
        ps.status = "active"
    # Mark others eliminated
    for ps in sorted_pop[k:]:
        ps.status = "eliminated"
    return elites


def revival_lottery(archive: list, revive_pct: float, pop_size: int) -> list:
    """Randomly revive eliminated ParameterSets weighted by rounds_survived."""
    eliminated = [ps for ps in archive if ps.status == "eliminated"]
    if not eliminated:
        return []
    k = max(1, int(pop_size * revive_pct))
    weights = [ps.rounds_survived for ps in eliminated]
    # Avoid zero weights
    weights = [w if w > 0 else 1 for w in weights]
    revived = random.choices(eliminated, weights=weights, k=k)
    for ps in revived:
        ps.rounds_survived += 1
        ps.status = "active"
    return revived


def generate_new(m: int, around: ParameterSet) -> list:
    """Generate new ParameterSets via Gaussian perturbation around a given set."""
    new_sets = []
    for _ in range(m):
        gains = {}
        for key in ['low', 'high', 'dyn']:
            base = np.array(around.gains[key])
            # perturb with small Gaussian noise
            perturbed = base + np.random.normal(0, 0.05, size=base.shape)
            gains[key] = [float(round(x, 3)) for x in perturbed]
        new_sets.append(ParameterSet(gains))
    return new_sets


def run_tournament(rounds: int, pop_size: int, data_files: list,
                   files_per_round: int, elite_pct: float, revive_pct: float):
    """Execute the tournament optimization and record per-round CSV summary."""
    # Prepare results directory and CSV file
    results_dir = BASE_DIR / "optimization" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    csv_path = results_dir / f"{timestamp}_tournament_results.csv"
    
    # Initialize population and archive
    population = initialize_population(pop_size)
    archive = list(population)
    
    # Open CSV and write header
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "round", "best_avg_cost", "num_elites", "num_revived", "num_new"
        ])
        
        # Tournament rounds
        for r in range(1, rounds + 1):
            # Evaluate active ParameterSets
            active = [ps for ps in population if ps.status == "active"]
            for ps in active:
                evaluate(ps, data_files, files_per_round)
            
            # Select elites
            elites = select_elites(active, elite_pct)
            
            # Revival lottery
            revived = revival_lottery(archive, revive_pct, pop_size)
            
            # Generate new entrants to refill population
            num_new = pop_size - (len(elites) + len(revived))
            best = min(elites, key=lambda ps: ps.stats['avg_total_cost'])
            new_sets = generate_new(num_new, best) if num_new > 0 else []
            
            # Update population and archive
            population = elites + revived + new_sets
            archive.extend(new_sets)
            
            # Compute best cost
            best_cost = best.stats.get('avg_total_cost', float('inf'))
            
            # Write round summary
            writer.writerow([
                r, 
                round(best_cost, 3), 
                len(elites), 
                len(revived), 
                len(new_sets)
            ])
    
    print(f"Tournament complete. Results written to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tournament-based optimizer for ensemble PID controllers"
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size")
    parser.add_argument(
        "--files", type=str, required=True,
        help="Directory of data files or comma-separated list"
    )
    parser.add_argument(
        "--files_per_round", type=int, default=25,
        help="Number of files to evaluate per ParameterSet each round"
    )
    parser.add_argument(
        "--elite_pct", type=float, default=0.2,
        help="Fraction of population to select as elites"
    )
    parser.add_argument(
        "--revive_pct", type=float, default=0.1,
        help="Fraction of population to revive each round"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Resolve data files
    if os.path.isdir(args.files):
        data_files = [str(f) for f in sorted(Path(args.files).glob("*.csv"))]
    else:
        data_files = [f.strip() for f in args.files.split(",")]
    
    run_tournament(
        rounds=args.rounds,
        pop_size=args.pop_size,
        data_files=data_files,
        files_per_round=args.files_per_round,
        elite_pct=args.elite_pct,
        revive_pct=args.revive_pct
    )


if __name__ == "__main__":
    main()
