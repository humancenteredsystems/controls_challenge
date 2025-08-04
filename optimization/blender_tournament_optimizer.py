#!/usr/bin/env python3
"""
Blender Tournament Optimizer - Neural Architecture Search for PID Blending
Simplified approach: Fixed PID controllers from Stage 3, evolve only neural architecture.
"""

import sys
import os
import argparse
import json
import random
import numpy as np
import hashlib
from pathlib import Path
import logging

# Add parent directory to path to find tinyphysics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tinyphysics_custom import run_rollout, TinyPhysicsModel
from utils.logging import print_banner, print_params, print_summary, print_goal_progress, tqdm, EMOJI_PARTY, EMOJI_TROPHY, EMOJI_OK

def cleanup_artifacts() -> None:
    """Remove leftover temporary controllers and blender models."""
    base_dir = Path(__file__).parent.parent
    for path in (base_dir / "controllers").glob("temp_*.py"):
        try: path.unlink()
        except: pass
    for path in (base_dir / "models").glob("blender_*.onnx"):
        try: path.unlink()
        except: pass

def load_top_pid_pairs(archive_path, n=5):
    """Load top N PID pairs from tournament archive."""
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    entries = archive.get('archive', [])
    valid = [e for e in entries if 'stats' in e and 'avg_total_cost' in e['stats']]
    top = sorted(valid, key=lambda x: x['stats']['avg_total_cost'])[:n]
    return [(e['low_gains'], e['high_gains']) for e in top]

def create_comprehensive_training_data(pid_pairs, data_files, model, num_samples=1000):
    """Generate comprehensive training data using all top PID pairs."""
    print(f"\n🔬 Generating comprehensive training data from {len(pid_pairs)} PID pairs...")
    samples = []
    samples_per_pair = num_samples // len(pid_pairs)
    
    for idx, (low_gains, high_gains) in enumerate(tqdm(pid_pairs, desc="PID Pairs", unit="pair")):
        pair_samples = []
        for _ in tqdm(range(samples_per_pair), desc=f" Pair {idx+1}", leave=False, unit="sample"):
            data_file = random.choice(data_files)
            best_blend = find_optimal_blend_weight(low_gains, high_gains, data_file, model)
            state, error, future = extract_state_from_file(data_file)
            features = [
                state.v_ego, state.roll_lataccel, state.a_ego,
                error, state.error_integral, state.error_derivative,
                np.mean(future.lataccel), np.std(future.lataccel)
            ]
            pair_samples.append((features, best_blend))
        samples.extend(pair_samples)
        print(f"  ✅ Generated {len(pair_samples)} samples for PID pair {idx+1}/{len(pid_pairs)}")
    
    # Fill remainder
    while len(samples) < num_samples:
        low_gains, high_gains = random.choice(pid_pairs)
        data_file = random.choice(data_files)
        best_blend = find_optimal_blend_weight(low_gains, high_gains, data_file, model)
        state, error, future = extract_state_from_file(data_file)
        features = [
            state.v_ego, state.roll_lataccel, state.a_ego,
            error, state.error_integral, state.error_derivative,
            np.mean(future.lataccel), np.std(future.lataccel)
        ]
        samples.append((features, best_blend))
    
    print_summary("Comprehensive Training Data", {
        "total_samples": len(samples),
        "pid_pairs": len(pid_pairs),
        "samples_per_pair": samples_per_pair
    })
    return samples

def create_random_architecture():
    """Create random neural architecture."""
    hidden = []
    for _ in range(random.randint(2, 3)):
        hidden.append(random.choice([16, 24, 32, 48]))
    dropout = random.choice([0.05, 0.1, 0.15])
    return {
        'hidden_sizes': hidden,
        'dropout_rate': dropout,
        'id': hashlib.md5(str(hidden + [dropout]).encode()).hexdigest()[:8],
        'cost': float('inf')
    }

def mutate_architecture(arch, mutation_rate=0.3):
    """Mutate neural architecture."""
    child = arch.copy()
    child['cost'] = float('inf')
    
    if random.random() < mutation_rate:
        # Mutate hidden layer size
        if child['hidden_sizes']:
            i = random.randrange(len(child['hidden_sizes']))
            child['hidden_sizes'][i] = random.choice([16, 24, 32, 48])
    
    if random.random() < mutation_rate:
        # Mutate dropout rate
        child['dropout_rate'] = random.choice([0.05, 0.1, 0.15])
    
    child['id'] = hashlib.md5(str(child['hidden_sizes'] + [child['dropout_rate']]).encode()).hexdigest()[:8]
    return child

def train_architecture(architecture, training_data, epochs=100):
    """Train neural architecture on training data."""
    from neural_blender_net import BlenderNet, train_blender_net
    arch_id = architecture['id']
    path = Path("models") / f"blender_{arch_id}.onnx"
    path.parent.mkdir(exist_ok=True)
    print(f"    Training architecture {arch_id}... ", end="", flush=True)
    model = train_blender_net(training_data, epochs=epochs)
    model.export_to_onnx(str(path))
    print("✅ Complete")
    return str(path)

def evaluate_architecture_on_pid_pairs(architecture, training_data, pid_pairs, data_files, model, max_files=20):
    """Evaluate architecture with multiple PID pairs showing detailed total_cost tracking."""
    onnx_path = train_architecture(architecture, training_data)
    costs = []
    
    try:
        test_files = random.sample(data_files, min(max_files, len(data_files)))
        print(f"      Testing on {len(test_files)} files with {len(pid_pairs[:3])} PID pairs...")
        
        for pid_idx, (low_gains, high_gains) in enumerate(pid_pairs[:3]):  # Test on top 3 PID pairs
            pid_costs = []
            for file_idx, test_file in enumerate(test_files):
                controller_name = create_temp_neural_controller(low_gains, high_gains, onnx_path, architecture['id'])
                try:
                    cost_result, _, _ = run_rollout(test_file, controller_name, model)
                    total_cost = cost_result["total_cost"]
                    pid_costs.append(total_cost)
                    costs.append(total_cost)
                    print(f"        PID{pid_idx+1} File{file_idx+1}: total_cost={total_cost:.2f}")
                except Exception:
                    costs.append(1e3)  # Penalty for failed evaluation
                    print(f"        PID{pid_idx+1} File{file_idx+1}: FAILED (cost=1000.0)")
                finally:
                    cleanup_temp_controller(controller_name)
            
            if pid_costs:
                print(f"      → PID Pair {pid_idx+1} average: {np.mean(pid_costs):.2f}")
        
        avg_cost = float(np.mean(costs)) if costs else float('inf')
        print(f"    → Architecture {architecture['id']} overall average: {avg_cost:.2f}")
        print_goal_progress(avg_cost)
        return avg_cost
    finally:
        try:
            Path(onnx_path).unlink()
        except:
            pass

def tournament_selection_and_evolution(population, elite_pct=0.3):
    """Tournament selection and mutation for next generation."""
    population.sort(key=lambda x: x['cost'])
    n_elites = max(1, int(len(population) * elite_pct))
    elites = population[:n_elites]
    
    next_gen = elites.copy()
    while len(next_gen) < len(population):
        parent = random.choice(population[:len(population)//2])  # Select from top half
        child = mutate_architecture(parent)
        next_gen.append(child)
    
    return next_gen

def create_temp_neural_controller(low_gains, high_gains, onnx_path, arch_id):
    """Create temporary neural controller file."""
    from optimization import generate_neural_blended_controller
    code = generate_neural_blended_controller(low_gains, high_gains, onnx_path)
    name = f"temp_neural_{hashlib.md5((str(arch_id) + onnx_path).encode()).hexdigest()[:8]}"
    path = Path("controllers") / f"{name}.py"
    Path("controllers").mkdir(exist_ok=True)
    with open(path, "w") as f:
        f.write(code)
    return name

def cleanup_temp_controller(name):
    """Remove temporary controller file."""
    path = Path("controllers") / f"{name}.py"
    if path.exists():
        path.unlink()

def get_tournament_baseline(archive_path):
    """Get baseline cost from tournament archive."""
    try:
        with open(archive_path, 'r') as f:
            archive = json.load(f)
        costs = [e['stats']['avg_total_cost'] for e in archive.get('archive', []) 
                if 'stats' in e and 'avg_total_cost' in e['stats']]
        baseline = min(costs) if costs else 200.0
        print(f"📊 Tournament baseline to beat: {baseline:.2f}")
        return baseline
    except:
        print("⚠️ Could not load baseline, using 200.0")
        return 200.0

def create_champion_controller(best_arch, best_pid_pair, archive_path):
    """Create final champion controller."""
    print(f"{EMOJI_TROPHY} Creating champion controller...")
    
    # Train champion model with extended epochs
    champion_data_path = Path("plans/blender_training_data.json")
    if champion_data_path.exists():
        with open(champion_data_path) as f:
            data = json.load(f)
        training_samples = [(s["features"], s["blend_weight"]) for s in data["samples"]]
    else:
        training_samples = []
    
    champion_onnx = train_architecture(best_arch, training_samples, epochs=200)
    final_path = "models/neural_blender_champion.onnx"
    Path(champion_onnx).rename(final_path)
    
    # Create champion controller code
    low_gains, high_gains = best_pid_pair
    from optimization import generate_neural_blended_controller
    controller_code = generate_neural_blended_controller(low_gains, high_gains, final_path)
    
    with open("controllers/neural_blended_champion.py", "w") as f:
        f.write(controller_code)
    
    print(f"{EMOJI_OK} Champion controller ready: controllers/neural_blended_champion.py")

def find_optimal_blend_weight(low_gains, high_gains, data_file, model):
    """Find optimal blend weight through discrete search."""
    from optimization import generate_blended_controller
    best_cost, best_blend = float('inf'), 0.5
    
    for blend in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        content = generate_blended_controller(low_gains, high_gains)
        # Replace blend logic to use fixed weight
        content = content.replace('if v_ego < 40:', f'if False:').replace(
            'weights = [0.8, 0.2]', f'weights = [{blend}, {1-blend}]').replace(
            'weights = [0.2, 0.8]', f'weights = [{blend}, {1-blend}]')
        
        name = f"temp_blend_{hashlib.md5(str(blend).encode()).hexdigest()[:8]}"
        path = Path("controllers") / f"{name}.py"
        try:
            with open(path, 'w') as f:
                f.write(content)
            cost_result, _, _ = run_rollout(data_file, name, model)
            if cost_result["total_cost"] < best_cost:
                best_cost, best_blend = cost_result["total_cost"], blend
        except:
            pass
        finally:
            if path.exists():
                path.unlink()
    
    return best_blend

def extract_state_from_file(data_file):
    """Extract state information from CSV file."""
    import pandas as pd
    df = pd.read_csv(data_file)
    idx = random.randint(100, min(len(df)-50, 400))
    
    from collections import namedtuple
    State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego', 'error_integral', 'error_derivative'])
    FuturePlan = namedtuple('FuturePlan', ['lataccel'])
    
    ACC_G = 9.81
    roll = np.sin(df.iloc[idx]['roll']) * ACC_G
    v = df.iloc[idx]['vEgo']
    a = df.iloc[idx]['aEgo']
    error = random.uniform(-1, 1)
    
    state = State(roll, v, a, error*0.1, error*0.05)
    future = FuturePlan(lataccel=df['targetLateralAcceleration'].iloc[idx:idx+20].tolist())
    
    return state, error, future

def run_blender_tournament(archive_path, data_files, model_path, rounds=10, pop_size=15, max_files=20):
    """Run simplified blender tournament focusing only on neural architecture."""
    print_banner(5, "Neural Blender Tournament (Architecture Search)")
    print_params({
        "archive": archive_path,
        "rounds": rounds,
        "population": pop_size, 
        "max_files": max_files,
        "approach": "Fixed PID pairs, evolve neural architecture only"
    })
    
    # Load fixed PID pairs from Stage 3
    pid_pairs = load_top_pid_pairs(archive_path, n=5)
    best_pid_pair = pid_pairs[0]  # Best pair for champion
    baseline = get_tournament_baseline(archive_path)
    
    model = TinyPhysicsModel(model_path, debug=False)
    print("Blender Tournament: GPU Enabled")
    
    # Create comprehensive training data
    training_data_path = Path("plans/blender_training_data.json")
    if training_data_path.exists():
        print("📂 Loading existing training data")
        with open(training_data_path) as f:
            data = json.load(f)
        training_data = [(s["features"], s["blend_weight"]) for s in data["samples"]]
    else:
        training_data = create_comprehensive_training_data(pid_pairs, data_files, model)
        # Save for future use
        data = {
            "num_samples": len(training_data),
            "feature_names": ["v_ego", "roll_lataccel", "a_ego", "error", 
                            "error_integral", "error_derivative", 
                            "future_lataccel_mean", "future_lataccel_std"],
            "samples": [{"features": feat, "blend_weight": weight} for feat, weight in training_data]
        }
        with open(training_data_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Initialize population
    population = [create_random_architecture() for _ in range(pop_size)]
    best_overall = {'cost': float('inf')}
    
    # Tournament evolution
    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num}/{rounds} ---", flush=True)
        
        # Evaluate architectures with detailed reporting
        for arch_idx, arch in enumerate(population):
            if arch['cost'] == float('inf'):
                print(f"  Architecture {arch_idx+1}/{len(population)} ({arch['id']})")
                cost = evaluate_architecture_on_pid_pairs(arch, training_data, pid_pairs, data_files, model, max_files)
                arch['cost'] = cost
                if cost < best_overall['cost']:
                    best_overall = arch.copy()
                    print(f" {EMOJI_PARTY} New overall best: {cost:.2f} (arch id: {arch['id']})", flush=True)
        
        # Round summary
        population.sort(key=lambda x: x['cost'])
        round_best = population[0]
        
        print_summary(f"Round {round_num} Summary", {
            "Round Best": f"{round_best['cost']:.2f}",
            "Overall Best": f"{best_overall['cost']:.2f}",
            "Architecture": f"{best_overall['hidden_sizes']}, dropout={best_overall['dropout_rate']:.2f}"
        })
        
        # Evolution for next round
        if round_num < rounds:
            population = tournament_selection_and_evolution(population)
            print("➡️  Next generation created", flush=True)
    
    # Final results
    improvement = baseline - best_overall['cost']
    print(f"\n🎯 Stage 5 Results:")
    print(f"  Baseline: {baseline:.2f}")
    print(f"  Best: {best_overall['cost']:.2f}")
    print(f"  Improvement: {improvement:.2f}")
    print(f"  Architecture: {best_overall['hidden_sizes']}, dropout={best_overall['dropout_rate']}")
    
    # Create champion
    create_champion_controller(best_overall, best_pid_pair, archive_path)
    
    # Save results
    results = {
        "best_cost": best_overall["cost"],
        "baseline_cost": baseline,
        "improvement": improvement,
        "winning_architecture": {
            "id": best_overall["id"],
            "hidden_sizes": best_overall["hidden_sizes"],
            "dropout_rate": best_overall["dropout_rate"]
        }
    }
    
    results_path = Path("plans/blender_tournament_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_overall

def main():
    cleanup_artifacts()
    parser = argparse.ArgumentParser(description='Simplified Blender Tournament Optimizer')
    parser.add_argument('--archive', default='plans/tournament_archive.json')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--pop_size', type=int, default=15)
    parser.add_argument('--max_files', type=int, default=20)
    parser.add_argument('--model_path', default='models/tinyphysics.onnx')
    parser.add_argument('--data_seed', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.data_seed is not None:
        random.seed(args.data_seed)
        np.random.seed(args.data_seed)
    
    data_files = [str(f) for f in Path("data").glob("*.csv")]
    if not data_files:
        print("❌ No data files found")
        return 1
    
    random.shuffle(data_files)
    print(f"Found {len(data_files)} data files")
    
    run_blender_tournament(args.archive, data_files, args.model_path, 
                          args.rounds, args.pop_size, args.max_files)
    return 0

if __name__ == "__main__":
    sys.exit(main())
