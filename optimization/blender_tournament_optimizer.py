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

# Add parent directory to path to find tinyphysics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tinyphysics_custom import run_rollout, TinyPhysicsModel
from utils.logging import (
    print_banner,
    print_params,
    print_summary,
    print_goal_progress,
    tqdm,
    EMOJI_PARTY,
    EMOJI_TROPHY,
    EMOJI_OK,
)

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

def create_random_hyperparameters():
    """Create random hyperparameter set for the fixed-architecture BlenderNet."""
    # Architecture is fixed to [32, 16] as per Stage 4 pre-training
    hyperparams = {
        'hidden_sizes': [32, 16],
        'dropout_rate': random.choice([0.05, 0.1, 0.15, 0.2, 0.25]),
        'learning_rate': random.choice([0.0005, 0.001, 0.002, 0.005]),
        'epochs': random.choice([50, 100, 150, 200]),
        'batch_size': random.choice([16, 32, 64]),
        'cost': float('inf')
    }
    # Generate a unique ID based on the hyperparameters
    hyperparams['id'] = hashlib.md5(str(sorted(hyperparams.items())).encode()).hexdigest()[:8]
    return hyperparams

def mutate_hyperparameters(parent_hyperparams, mutation_rate=0.3):
    """Mutate hyperparameters."""
    child = parent_hyperparams.copy()
    child['cost'] = float('inf')

    if random.random() < mutation_rate:
        child['dropout_rate'] = random.choice([0.05, 0.1, 0.15, 0.2, 0.25])
    
    if random.random() < mutation_rate:
        child['learning_rate'] = random.choice([0.0005, 0.001, 0.002, 0.005])

    if random.random() < mutation_rate:
        child['epochs'] = random.choice([50, 100, 150, 200])

    if random.random() < mutation_rate:
        child['batch_size'] = random.choice([16, 32, 64])

    child['id'] = hashlib.md5(str(sorted(child.items())).encode()).hexdigest()[:8]
    return child

def train_model_with_hyperparameters(hyperparams, training_data_path, pretrained_path=None):
    """Train a BlenderNet model with a specific set of hyperparameters."""
    from neural_blender_net import BlenderNet, train_blender_net_from_json
    import torch

    model_id = hyperparams['id']
    model_output_path = Path("models") / f"blender_{model_id}.onnx"
    model_output_path.parent.mkdir(exist_ok=True)
    print(f"    Training model {model_id} with params: {hyperparams}... ", end="", flush=True)

    # Create a new model with the fixed architecture but specified hyperparameters
    model = BlenderNet(
        hidden_sizes=hyperparams['hidden_sizes'],
        dropout_rate=hyperparams['dropout_rate']
    )

    # Load the weights from the pre-trained model
    if pretrained_path:
        if not Path(pretrained_path).exists():
            raise FileNotFoundError(
                f"Pretrained BlenderNet weights not found at '{pretrained_path}'. "
                "Generate weights with `python train_blender.py` or supply a valid file."
            )
        try:
            model.load_state_dict(torch.load(pretrained_path, weights_only=False))
        except Exception as e:
            raise RuntimeError(
                "Failed to load pretrained BlenderNet weights from "
                f"'{pretrained_path}': {e}. "
                "Generate weights with `python train_blender.py` or supply a valid file."
            ) from e

    # Train the model
    train_blender_net_from_json(
        data_path=training_data_path,
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        lr=hyperparams['learning_rate'],
        model_output=str(model_output_path),
        hidden_sizes=hyperparams['hidden_sizes'],
        dropout_rate=hyperparams['dropout_rate'],
    )
    
    print("✅ Complete")
    return str(model_output_path)


def train_architecture(architecture, training_data, epochs=100, pretrained_path=None):
    """Placeholder to allow tests to monkeypatch training."""
    raise NotImplementedError("train_architecture should be monkeypatched in tests")

def evaluate_hyperparameters_on_pid_pairs(hyperparams, training_data_path, pid_pairs, data_files, model, max_files=20, pretrained_path=None):
    """Evaluate a hyperparameter set with multiple PID pairs."""
    onnx_path = train_model_with_hyperparameters(hyperparams, training_data_path, pretrained_path)
    costs = []
    
    # Load normalization stats from training data
    with open(training_data_path, 'r') as f:
        norm_stats = json.load(f).get('feature_stats')

    try:
        test_files = random.sample(data_files, min(max_files, len(data_files)))
        print(f"      Testing on {len(test_files)} files with {len(pid_pairs[:3])} PID pairs...")
        
        for pid_idx, (low_gains, high_gains) in enumerate(pid_pairs[:3]):
            pid_costs = []
            for file_idx, test_file in enumerate(test_files):
                controller_name = create_temp_neural_controller(low_gains, high_gains, onnx_path, hyperparams['id'], norm_stats)
                try:
                    cost_result, _, _ = run_rollout(test_file, controller_name, model)
                    total_cost = cost_result["total_cost"]
                    pid_costs.append(total_cost)
                    costs.append(total_cost)
                except Exception:
                    costs.append(1e3)
                finally:
                    cleanup_temp_controller(controller_name)
            
        avg_cost = float(np.mean(costs)) if costs else float('inf')
        print(f"    → Hyperparameters {hyperparams['id']} overall average: {avg_cost:.2f}")
        print_goal_progress(avg_cost)
        return avg_cost
    finally:
        try:
            Path(onnx_path).unlink()
        except:
            pass


def evaluate_architecture_on_pid_pairs(architecture, training_data_path, pid_pairs, data_files, model, max_files=20):
    """Evaluate a neural architecture on PID gain pairs.

    This is a thin wrapper around :func:`evaluate_hyperparameters_on_pid_pairs`
    that exists for backward compatibility with tests expecting this API.
    The implementation delegates the training step to ``train_architecture``
    so that tests can monkeypatch it.
    """
    onnx_path = train_architecture(architecture, training_data_path)
    costs = []
    try:
        test_files = random.sample(data_files, min(max_files, len(data_files)))

        for pid_idx, (low_gains, high_gains) in enumerate(pid_pairs[:3]):
            pid_costs = []
            for test_file in test_files:
                controller_name = create_temp_neural_controller(low_gains, high_gains, onnx_path, architecture["id"])
                try:
                    cost_result, _, _ = run_rollout(test_file, controller_name, model)
                    total_cost = cost_result["total_cost"]
                    pid_costs.append(total_cost)
                    costs.append(total_cost)
                except Exception:
                    costs.append(1e3)
                finally:
                    cleanup_temp_controller(controller_name)

        return float(np.mean(costs)) if costs else float("inf")
    finally:
        try:
            Path(onnx_path).unlink()
        except Exception:
            pass



def tournament_selection_and_evolution(population, elite_pct=0.3):
    """Tournament selection and mutation for next generation."""
    population.sort(key=lambda x: x['cost'])
    n_elites = max(1, int(len(population) * elite_pct))
    elites = population[:n_elites]
    
    next_gen = elites.copy()
    while len(next_gen) < len(population):
        parent = random.choice(population[:len(population)//2])  # Select from top half
        child = mutate_hyperparameters(parent)
        next_gen.append(child)
    
    return next_gen

def create_temp_neural_controller(low_gains, high_gains, onnx_path, arch_id, norm_stats=None):
    """Create temporary neural controller file."""
    from optimization import generate_neural_blended_controller
    code = generate_neural_blended_controller(low_gains, high_gains, onnx_path, norm_stats=norm_stats)
    name = f"temp_neural_{hashlib.md5((str(arch_id) + onnx_path).encode()).hexdigest()[:8]}"
    path = Path("controllers") / f"{name}.py"
    Path("controllers").mkdir(exist_ok=True)
    with open(path, "w") as f:
        f.write(code)
    return name

# Backwards compatibility for tests expecting a private helper
_make_temp_neural_controller = create_temp_neural_controller

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

def create_champion_controller(best_hyperparams, best_pid_pair, archive_path):
    """Create final champion controller."""
    print(f"{EMOJI_TROPHY} Creating champion controller...")
    
    # Train champion model with extended epochs
    champion_data_path = "plans/blender_training_data.json"
    champion_onnx = train_model_with_hyperparameters(
        best_hyperparams,
        champion_data_path,
        pretrained_path="models/neural_blender_pretrained.pth"
    )
    final_path = "models/neural_blender_champion.onnx"
    Path(champion_onnx).rename(final_path)
    
    # Load normalization stats to embed in the controller
    with open(champion_data_path, 'r') as f:
        training_data = json.load(f)
    norm_stats = training_data.get('feature_stats')

    # Create champion controller code
    low_gains, high_gains = best_pid_pair
    from optimization import generate_neural_blended_controller
    controller_code = generate_neural_blended_controller(low_gains, high_gains, final_path, norm_stats=norm_stats)
    
    with open("controllers/neural_blended_champion.py", "w") as f:
        f.write(controller_code)
    
    print(f"{EMOJI_OK} Champion controller ready: controllers/neural_blended_champion.py")

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
    """Run simplified blender tournament focusing only on hyperparameter optimization."""
    print_banner(5, "Neural Blender Tournament (Hyperparameter Search)")
    print_params({
        "archive": archive_path,
        "rounds": rounds,
        "population": pop_size,
        "max_files": max_files,
        "approach": "Fixed architecture, evolve hyperparameters"
    })

    # Load fixed PID pairs from Stage 3
    pid_pairs = load_top_pid_pairs(archive_path, n=5)
    best_pid_pair = pid_pairs[0]  # Best pair for champion
    baseline = get_tournament_baseline(archive_path)

    model = TinyPhysicsModel(model_path, debug=False)
    print("Blender Tournament: GPU Enabled")

    # Ensure training data exists from Stage 4
    training_data_path = "plans/blender_training_data.json"
    if not Path(training_data_path).exists():
        print(f"❌ Training data not found at {training_data_path}. Run Stage 4 first.")
        sys.exit(1)
    print(f"📂 Using training data from {training_data_path}")

    # Initialize population
    population = [create_random_hyperparameters() for _ in range(pop_size)]
    best_overall = {'cost': float('inf')}
    
    # Tournament evolution
    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num}/{rounds} ---", flush=True)
        
        # Evaluate hyperparameter sets with detailed reporting
        for hyperparam_idx, hyperparams in enumerate(population):
            if hyperparams['cost'] == float('inf'):
                print(f"  Hyperparameters {hyperparam_idx+1}/{len(population)} ({hyperparams['id']})")
                cost = evaluate_hyperparameters_on_pid_pairs(
                    hyperparams,
                    training_data_path,
                    pid_pairs,
                    data_files,
                    model,
                    max_files,
                    pretrained_path="models/neural_blender_pretrained.pth"
                )
                hyperparams['cost'] = cost
                if cost < best_overall['cost']:
                    best_overall = hyperparams.copy()
                    print(f" {EMOJI_PARTY} New overall best: {cost:.2f} (hyperparams id: {hyperparams['id']})", flush=True)
        
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
        import torch
        torch.manual_seed(args.data_seed)
    
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
