"""
GPU-optimized tournament optimizer using proven blended 2-PID architecture
"""
import uuid
import numpy as np
import json
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional
import sys
import os

# Add the parent directory to path to find tinyphysics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tinyphysics_custom import run_rollout
from optimization import generate_blended_controller

class ParameterSet:
    """Represents a set of blended 2-PID gains and metadata in a tournament."""
    def __init__(self, low_gains: List[float], high_gains: List[float]):
        self.id = str(uuid.uuid4())
        self.low_gains = low_gains    # [P, I, D] for low-speed PID
        self.high_gains = high_gains  # [P, I, D] for high-speed PID
        self.stats: Dict[str, Any] = {}
        self.rounds_survived: int = 0
        self.status: str = "active"

def initialize_population(n: int, seed_from_archive: Optional[str] = None) -> List[ParameterSet]:
    """Generate initial population, optionally seeding best performers from archive."""
    population: List[ParameterSet] = []
    
    # Seed with archive champions if provided
    if seed_from_archive and Path(seed_from_archive).exists():
        try:
            with open(seed_from_archive, 'r') as f:
                archive_data = json.load(f)
            
            # Extract archive list from the JSON structure
            archive_list = archive_data.get('archive', [])
            
            # Sort by avg_cost, take top half of population as champions
            champions = sorted(
                [ps for ps in archive_list if ps.get('stats', {}).get('avg_total_cost') != float('inf')],
                key=lambda x: x.get('stats', {}).get('avg_total_cost', float('inf'))
            )[:n//2]
            
            print(f"Seeding {len(champions)} champions from {seed_from_archive}")
            
            for i, champ in enumerate(champions):
                ps = ParameterSet(champ['low_gains'], champ['high_gains'])
                ps.id = f"champion_{i}"
                ps.rounds_survived = 0  # Reset for new tournament
                population.append(ps)
                
        except Exception as e:
            print(f"Failed to load archive {seed_from_archive}: {e}")
            print("Falling back to random initialization")
    
    # Fill remaining slots with random generation
    np.random.seed(42)  # For reproducible results
    remaining = n - len(population)
    
    for _ in range(remaining):
        # Use EXACT same ranges that achieved 76.81 cost in blended_2pid_optimizer
        low_gains = list(np.random.uniform([0.25, 0.01, -0.25], [0.6, 0.12, -0.05]))
        high_gains = list(np.random.uniform([0.15, 0.005, -0.15], [0.4, 0.08, -0.03]))
        
        # Round to reasonable precision
        low_gains = [round(g, 3) for g in low_gains]
        high_gains = [round(g, 3) for g in high_gains]
        
        population.append(ParameterSet(low_gains, high_gains))
    
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
    
    # Weight by rounds survived (more rounds = higher revival chance)
    weights = np.array([ps.rounds_survived + 1 for ps in eliminated])
    weights = weights / weights.sum()
    
    revived_indices = np.random.choice(len(eliminated), size=count, replace=False, p=weights)
    revived = [eliminated[i] for i in revived_indices]
    
    for ps in revived:
        ps.status = "active"
    return revived

def generate_new(count: int, best: ParameterSet, perturb_scale: float) -> List[ParameterSet]:
    """Generate new parameter sets via Gaussian perturbation around the best."""
    new_sets: List[ParameterSet] = []
    for _ in range(count):
        # Perturb low-speed gains with proper bounds
        new_low = [
            np.clip(best.low_gains[0] + np.random.normal(0, perturb_scale), 0.25, 0.6),  # P
            np.clip(best.low_gains[1] + np.random.normal(0, perturb_scale), 0.01, 0.12), # I
            np.clip(best.low_gains[2] + np.random.normal(0, perturb_scale), -0.25, -0.05) # D
        ]
        
        # Perturb high-speed gains with proper bounds
        new_high = [
            np.clip(best.high_gains[0] + np.random.normal(0, perturb_scale), 0.15, 0.4),  # P
            np.clip(best.high_gains[1] + np.random.normal(0, perturb_scale), 0.005, 0.08), # I
            np.clip(best.high_gains[2] + np.random.normal(0, perturb_scale), -0.15, -0.03) # D
        ]
        
        # Round to reasonable precision
        new_low = [round(g, 3) for g in new_low]
        new_high = [round(g, 3) for g in new_high]
        
        new_sets.append(ParameterSet(new_low, new_high))
    return new_sets

def _make_temp_controller(ps: ParameterSet) -> str:
    """Generate temporary controller using shared utility."""
    controller_content = generate_blended_controller(ps.low_gains, ps.high_gains)
    
    controllers_dir = Path(__file__).parent.parent / "controllers"
    module_name = f"temp_{ps.id.replace('-', '')}"
    file_path = controllers_dir / f"{module_name}.py"
    
    with open(file_path, "w") as f:
        f.write(controller_content)
    
    return module_name

def cleanup_controllers(prefix: str = "temp_") -> None:
    """Remove temporary controller files."""
    controllers_dir = Path(__file__).parent.parent / "controllers"
    for path in controllers_dir.glob(f"{prefix}*.py"):
        try:
            path.unlink()
        except:
            pass

def evaluate(ps: ParameterSet, data_files: List[str], model_path_or_instance, max_files: int) -> None:
    """Evaluate a ParameterSet and fill its stats. Accepts either model path or model instance for GPU optimization."""
    import sys
    
    total_costs: List[float] = []
    mod = _make_temp_controller(ps)
    
    # Clear module from cache if it exists to force fresh import
    full_module_name = f"controllers.{mod}"
    if full_module_name in sys.modules:
        del sys.modules[full_module_name]
    
    try:
        for file in data_files[:max_files]:
            cost, _, _ = run_rollout(file, mod, model_path_or_instance, debug=False)
            total_costs.append(cost["total_cost"])
    except Exception as e:
        # If evaluation fails, set infinite cost
        pass
    finally:
        cleanup_controllers(prefix=f"temp_{ps.id.replace('-', '')}")
        # Also clean up from sys.modules
        if full_module_name in sys.modules:
            del sys.modules[full_module_name]
    
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
                   max_files: int, perturb_scale: float, seed_from_archive: Optional[str] = None) -> None:
    """Execute the tournament optimization loop with GPU optimization."""
    # Create model instance once for GPU optimization
    from tinyphysics_custom import TinyPhysicsModel
    model = TinyPhysicsModel(model_path, debug=False)
    providers = model.ort_session.get_providers()
    gpu_status = 'GPU ENABLED' if 'CUDAExecutionProvider' in providers else 'CPU FALLBACK'
    print(f"Tournament optimizer: {gpu_status}")
    
    population = initialize_population(pop_size, seed_from_archive)
    archive: List[ParameterSet] = population.copy()
    summary: List[Dict[str, Any]] = []
    best_cost = float('inf')
    
    print(f"Starting GPU-accelerated tournament optimization:")
    print(f"  - {rounds} rounds")
    print(f"  - Population size: {pop_size}")
    print(f"  - {max_files} files per evaluation")
    print(f"  - {len(data_files)} total data files available")
    
    for r in range(1, rounds + 1):
        print(f"\n🏆 Tournament Round {r}/{rounds}")
        
        # Evaluate all population members
        for i, ps in enumerate(population):
            evaluate(ps, data_files, model, max_files)
            
            # Show progress every few evaluations
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1}/{len(population)} parameter sets...")
        
        # Selection and evolution
        elites = select_elites(population, elite_pct)
        revived = revival_lottery(archive, revive_pct, pop_size)
        best = min(elites, key=lambda ps: ps.stats["avg_total_cost"])
        
        # Track best cost and show improvements
        if best.stats["avg_total_cost"] < best_cost:
            best_cost = best.stats["avg_total_cost"]
            print(f"\n🎉 New tournament best: {best_cost:.2f}")
            print(f"   Low-speed:  P={best.low_gains[0]:.3f}, I={best.low_gains[1]:.3f}, D={best.low_gains[2]:.3f}")
            print(f"   High-speed: P={best.high_gains[0]:.3f}, I={best.high_gains[1]:.3f}, D={best.high_gains[2]:.3f}")
            
        # Generate new parameter sets
        new_count = pop_size - len(elites) - len(revived)
        new_sets = generate_new(new_count, best, perturb_scale)
        
        # Update population and archive
        archive.extend(new_sets + revived)
        population = elites + revived + new_sets
        
        # Save round summary
        summary.append({
            "round": r,
            "elites": [ps.id for ps in elites],
            "revived": [ps.id for ps in revived],
            "new": [ps.id for ps in new_sets],
            "best_cost": best.stats["avg_total_cost"],
        })
        
        print(f"  Round {r} complete: {len(elites)} elites, {len(revived)} revived, {len(new_sets)} new")
    
    # Ensure plans directory exists and save results
    plans_dir = Path(__file__).parent.parent / "plans"
    plans_dir.mkdir(exist_ok=True)
    
    # Save per-round summary
    (plans_dir / "tournament_progress.json").write_text(
        json.dumps({"tournament_summary": summary}, indent=2)
    )
    
    # Save full archive of ParameterSets
    archive_list: List[Dict[str, Any]] = []
    for ps in archive:
        archive_list.append({
            "id": ps.id,
            "low_gains": ps.low_gains,
            "high_gains": ps.high_gains,
            "stats": ps.stats,
            "rounds_survived": ps.rounds_survived,
            "status": ps.status
        })
    
    (plans_dir / "tournament_archive.json").write_text(
        json.dumps({"archive": archive_list}, indent=2)
    )
    
    # Find and report final best
    final_best = min([ps for ps in archive if ps.stats.get('avg_total_cost', float('inf')) != float('inf')], 
                    key=lambda ps: ps.stats["avg_total_cost"])
    
    print(f"\n🏆 Tournament Complete!")
    print(f"Final best cost: {final_best.stats['avg_total_cost']:.2f}")
    print(f"Low-speed gains:  P={final_best.low_gains[0]:.3f}, I={final_best.low_gains[1]:.3f}, D={final_best.low_gains[2]:.3f}")
    print(f"High-speed gains: P={final_best.high_gains[0]:.3f}, I={final_best.high_gains[1]:.3f}, D={final_best.high_gains[2]:.3f}")
    print(f"Results saved to plans/tournament_progress.json and plans/tournament_archive.json")

def main():
    """Main tournament optimization routine"""
    parser = argparse.ArgumentParser(description="Tournament optimizer for blended 2-PID controllers")
    parser.add_argument("--rounds", type=int, default=20, help="Number of tournament rounds")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size")
    parser.add_argument("--elite_pct", type=float, default=0.2, help="Elite percentage")
    parser.add_argument("--revive_pct", type=float, default=0.1, help="Revival percentage")
    parser.add_argument("--max_files", type=int, default=25, help="Max files per evaluation")
    parser.add_argument("--perturb_scale", type=float, default=0.05, help="Perturbation scale")
    parser.add_argument("--seed_from_archive", type=str, default=None, help="Archive file to seed champions from")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    model_path = str(base_dir / "models" / "tinyphysics.onnx")
    data_dir = base_dir / "data"
    
    # Get data files - use all available files if no archive seeding, limit for initial tournament
    if args.seed_from_archive:
        # Tournament #2: Use all available data files for expanded validation
        data_files = [str(f) for f in sorted(data_dir.glob("*.csv"))]
        print(f"Tournament #2: Found {len(data_files)} data files (expanded dataset)")
    else:
        # Tournament #1: Use subset for initial optimization
        data_files = [str(f) for f in sorted(data_dir.glob("*.csv"))[:50]]
        print(f"Tournament #1: Found {len(data_files)} data files (initial dataset)")
    
    if not data_files:
        print("No data files found!")
        return
    
    run_tournament(
        data_files=data_files,
        model_path=model_path,
        rounds=args.rounds,
        pop_size=args.pop_size,
        elite_pct=args.elite_pct,
        revive_pct=args.revive_pct,
        max_files=args.max_files,
        perturb_scale=args.perturb_scale,
        seed_from_archive=args.seed_from_archive
    )

if __name__ == "__main__":
    main()
