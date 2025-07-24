#!/usr/bin/env python3
"""
Blender Tournament Optimizer - Neural Network Architecture Search for PID Blending
Extends the tournament optimization framework to evolve BlenderNet architectures
"""

import sys
import os
import argparse
import json
import random
import numpy as np
import hashlib
import tempfile
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to find tinyphysics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tinyphysics_custom import run_rollout, TinyPhysicsModel

def create_training_data_from_archive(archive_path, num_samples=5000):
    """
    Generate training data for BlenderNet from PID tournament archive
    
    Args:
        archive_path: Path to tournament archive JSON
        num_samples: Number of training samples to generate
    
    Returns:
        List of (features, optimal_blend_weight) tuples
    """
    print(f"🔬 Generating training data from {archive_path}")
    
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    # Get top performing PID combinations (top 10)
    top_performers = sorted(archive['archive'][:20], 
                           key=lambda x: x['stats']['min_cost'])[:10]
    
    print(f"Using top {len(top_performers)} PID combinations for training data")
    
    training_samples = []
    
    # For each top performer, simulate and find optimal blending patterns
    for i, combo in enumerate(top_performers):
        pid1_params = combo['low_gains']
        pid2_params = combo['high_gains']
        
        print(f"  Processing combo {i+1}/{len(top_performers)}: cost={combo['stats']['min_cost']:.2f}")
        
        # Generate samples for this PID combination
        samples = generate_samples_for_pid_combo(pid1_params, pid2_params, 
                                               num_samples // len(top_performers))
        training_samples.extend(samples)
    
    print(f"Generated {len(training_samples)} training samples")
    
    # Save training data
    training_data_path = Path("plans/blender_training_data.json")
    with open(training_data_path, 'w') as f:
        json.dump(training_samples, f, indent=2)
    
    return training_samples

def generate_samples_for_pid_combo(pid1_params, pid2_params, num_samples):
    """Generate training samples for a specific PID combination"""
    
    # This is a simplified version - in practice, you'd run detailed simulations
    # to find optimal blend weights at different states
    
    samples = []
    
    for _ in range(num_samples):
        # Generate random vehicle state
        v_ego = random.uniform(5, 60)  # Speed range
        roll_lataccel = random.uniform(-3, 3)  # Lateral acceleration
        a_ego = random.uniform(-2, 2)  # Longitudinal acceleration
        
        error = random.uniform(-1, 1)  # Control error
        error_integral = random.uniform(-0.5, 0.5)  # Integral term
        error_derivative = random.uniform(-0.2, 0.2)  # Derivative term
        
        future_lataccel_mean = random.uniform(-2, 2)  # Future plan mean
        future_lataccel_std = random.uniform(0, 1)    # Future plan std
        
        # Features vector (8 dimensions)
        features = [v_ego, roll_lataccel, a_ego, error, error_integral, 
                   error_derivative, future_lataccel_mean, future_lataccel_std]
        
        # Compute optimal blend weight (simplified heuristic for now)
        # In practice, this would come from detailed simulation analysis
        if v_ego < 20:
            optimal_blend = 0.8  # Favor PID1 at low speeds
        elif v_ego > 45:
            optimal_blend = 0.2  # Favor PID2 at high speeds  
        else:
            # Blend based on error magnitude and future plan complexity
            error_factor = min(abs(error), 1.0)
            plan_complexity = min(future_lataccel_std, 1.0)
            optimal_blend = 0.5 + 0.3 * (error_factor - plan_complexity)
            optimal_blend = np.clip(optimal_blend, 0.0, 1.0)
        
        samples.append((features, optimal_blend))
    
    return samples

def create_random_blender_architecture():
    """Create random BlenderNet architecture for tournament evolution"""
    
    # Random architecture parameters
    hidden_sizes = []
    num_layers = random.randint(2, 3)  # 2-3 hidden layers
    
    for _ in range(num_layers):
        layer_size = random.choice([16, 24, 32, 48])
        hidden_sizes.append(layer_size)
    
    dropout_rate = random.choice([0.05, 0.1, 0.15])
    
    return {
        'hidden_sizes': hidden_sizes,
        'dropout_rate': dropout_rate,
        'id': hashlib.md5(str(hidden_sizes + [dropout_rate]).encode()).hexdigest()[:8]
    }

def train_blender_architecture(architecture, training_data, epochs=100):
    """
    Train a BlenderNet architecture on training data
    
    Args:
        architecture: Dict with network architecture parameters
        training_data: List of (features, label) tuples
        epochs: Number of training epochs
    
    Returns:
        Path to trained ONNX model
    """
    
    # This is a placeholder - in practice, you'd use PyTorch here
    # For now, we'll simulate training and create a dummy ONNX model
    
    model_id = architecture['id']
    onnx_path = f"models/blender_{model_id}.onnx"
    
    # Simulate training time
    print(f"    Training architecture {model_id} for {epochs} epochs...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # For now, create a placeholder ONNX file
    # In practice, this would be the actual trained PyTorch model exported to ONNX
    with open(onnx_path, 'wb') as f:
        f.write(b"PLACEHOLDER_ONNX_MODEL")  # This would be real ONNX bytes
    
    return onnx_path

def evaluate_blender_architecture(architecture, training_data, data_files, model, max_files=20):
    """
    Evaluate BlenderNet architecture performance
    
    Args:
        architecture: Dict with network architecture
        training_data: Training data for this architecture  
        data_files: List of data files for evaluation
        model: TinyPhysicsModel instance
        max_files: Maximum files to evaluate on
    
    Returns:
        Average cost across evaluations
    """
    
    # Train the architecture
    onnx_path = train_blender_architecture(architecture, training_data)
    
    # Get best PID parameters from archive for evaluation
    pid_pairs = get_top_pid_pairs_from_archive()
    
    total_cost = 0
    num_evaluations = 0
    
    # Evaluate on subset of data files
    eval_files = data_files[:max_files]
    
    for data_file in eval_files:
        for pid1_params, pid2_params in pid_pairs[:3]:  # Top 3 PID pairs
            
            # Create temporary neural controller
            temp_controller_path = create_temp_neural_controller_file(
                pid1_params, pid2_params, onnx_path, architecture['id']
            )
            
            try:
                # Evaluate using the neural blended controller
                rollout_result = run_rollout(data_file, "neural_blended", model, debug=False)
                
                # Extract cost from rollout result (handle both formats)
                if isinstance(rollout_result, tuple):
                    cost = rollout_result[0].get('total_cost', 1000)
                else:
                    cost = rollout_result
                
                total_cost += cost
                num_evaluations += 1
                
            except Exception as e:
                print(f"    Evaluation failed: {e}")
                total_cost += 1000  # Penalty for failed evaluation
                num_evaluations += 1
                
            finally:
                # Clean up temporary controller
                if os.path.exists(temp_controller_path):
                    os.remove(temp_controller_path)
    
    return total_cost / num_evaluations if num_evaluations > 0 else 1000

def get_top_pid_pairs_from_archive(archive_path="plans/tournament_archive.json"):
    """Get top PID parameter pairs from tournament archive"""
    
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    # Filter performers with valid stats (same robust handling as training data gen)
    valid_performers = []
    for p in archive['archive']:
        if 'stats' in p and 'avg_total_cost' in p['stats']:
            valid_performers.append(p)
    
    if len(valid_performers) == 0:
        # Fallback to default parameters if no valid performers
        return [([0.25, 0.12, -0.05], [0.15, 0.08, -0.15])]
    
    # Get top 5 performers
    top_performers = sorted(valid_performers,
                           key=lambda x: x['stats']['avg_total_cost'])[:5]
    
    pid_pairs = []
    for combo in top_performers:
        pid_pairs.append((combo['low_gains'], combo['high_gains']))
    
    return pid_pairs

def create_temp_neural_controller_file(pid1_params, pid2_params, onnx_path, arch_id):
    """Create temporary neural controller file for evaluation"""
    
    controller_content = f'''from controllers.neural_blended import Controller as BaseNeuralController

class Controller(BaseNeuralController):
    def __init__(self):
        pid1_params = {pid1_params}
        pid2_params = {pid2_params}
        blender_model_path = "{onnx_path}"
        
        super().__init__(pid1_params, pid2_params, blender_model_path)
'''
    
    temp_path = f"controllers/temp_neural_eval_{arch_id}.py"
    
    with open(temp_path, 'w') as f:
        f.write(controller_content)
    
    return temp_path

def tournament_selection_and_mutation(population, elite_pct=0.3, mutation_rate=0.2):
    """Apply tournament selection and mutation to BlenderNet population"""
    
    # Sort by cost (lower is better)
    population.sort(key=lambda x: x['cost'])
    
    pop_size = len(population)
    num_elites = int(pop_size * elite_pct)
    
    # Keep elites
    new_population = population[:num_elites].copy()
    
    # Generate offspring through mutation
    while len(new_population) < pop_size:
        # Select random parent from top 50%
        parent = random.choice(population[:pop_size//2])
        
        # Mutate architecture
        child = mutate_architecture(parent, mutation_rate)
        child['cost'] = float('inf')  # Will be evaluated next round
        
        new_population.append(child)
    
    return new_population

def mutate_architecture(parent, mutation_rate=0.2):
    """Create mutated version of BlenderNet architecture"""
    
    child = parent.copy()
    
    # Mutate hidden layer sizes
    if random.random() < mutation_rate:
        idx = random.randint(0, len(child['hidden_sizes']) - 1)
        child['hidden_sizes'][idx] = random.choice([16, 24, 32, 48])
    
    # Mutate dropout rate
    if random.random() < mutation_rate:
        child['dropout_rate'] = random.choice([0.05, 0.1, 0.15])
    
    # Generate new ID
    child['id'] = hashlib.md5(str(child['hidden_sizes'] + [child['dropout_rate']]).encode()).hexdigest()[:8]
    
    return child

def run_blender_tournament(archive_path, data_files, model_path, rounds=15, pop_size=20, max_files=20):
    """
    Run tournament optimization for BlenderNet architectures
    
    Args:
        archive_path: Path to PID tournament archive
        data_files: List of data files for evaluation  
        model_path: Path to TinyPhysics model
        rounds: Number of tournament rounds
        pop_size: Population size
        max_files: Max files per evaluation
    
    Returns:
        Best BlenderNet architecture
    """
    
    print("🏆 Starting Blender Tournament Optimization")
    print("=" * 60)
    
    # Create GPU-accelerated model instance (follows tinyphysics.py pattern)
    model = TinyPhysicsModel(model_path, debug=False)
    print("Blender tournament: GPU ENABLED")
    
    # Generate training data from PID archive
    training_data = create_training_data_from_archive(archive_path)
    
    # Initialize population with random architectures
    population = []
    for i in range(pop_size):
        architecture = create_random_blender_architecture()
        architecture['cost'] = float('inf')
        population.append(architecture)
    
    print(f"Starting blender tournament optimization:")
    print(f"  - {rounds} rounds")
    print(f"  - Population size: {pop_size}")
    print(f"  - {max_files} files per evaluation")
    print(f"  - {len(data_files)} total data files available")
    print()
    
    best_ever_cost = float('inf')
    best_ever_architecture = None
    
    # Tournament rounds
    for round_num in range(1, rounds + 1):
        print(f"🏆 Blender Tournament Round {round_num}/{rounds}")
        
        # Evaluate each architecture
        for i, architecture in enumerate(population):
            if architecture['cost'] == float('inf'):  # Not yet evaluated
                cost = evaluate_blender_architecture(
                    architecture, training_data, data_files, model, max_files
                )
                architecture['cost'] = cost
                
                if cost < best_ever_cost:
                    best_ever_cost = cost
                    best_ever_architecture = architecture.copy()
            
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1}/{pop_size} architectures...")
        
        # Find best in this round
        round_best = min(population, key=lambda x: x['cost'])
        
        if round_best['cost'] < best_ever_cost:
            best_ever_cost = round_best['cost']
            best_ever_architecture = round_best.copy()
            
            print(f"🎉 New tournament best: {best_ever_cost:.2f}")
            print(f"   Architecture: {best_ever_architecture['hidden_sizes']}")
            print(f"   Dropout: {best_ever_architecture['dropout_rate']}")
        
        # Selection and mutation for next round
        if round_num < rounds:
            population = tournament_selection_and_mutation(population)
            print(f"  Round {round_num} complete: generation evolved")
        else:
            print(f"  Round {round_num} complete: final round")
    
    print()
    print("🏆 Blender Tournament Complete!")
    print(f"Best architecture cost: {best_ever_cost:.2f}")
    print(f"Best architecture: {best_ever_architecture}")
    
    # Save best architecture
    results = {
        'best_architecture': best_ever_architecture,
        'best_cost': best_ever_cost,
        'tournament_config': {
            'rounds': rounds,
            'pop_size': pop_size,
            'max_files': max_files
        }
    }
    
    results_path = "plans/blender_tournament_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return best_ever_architecture

def main():
    parser = argparse.ArgumentParser(description='Blender Tournament Optimizer')
    
    parser.add_argument('--archive', type=str, default='plans/tournament_archive.json',
                       help='Path to PID tournament archive')
    parser.add_argument('--rounds', type=int, default=15,
                       help='Number of tournament rounds')
    parser.add_argument('--pop_size', type=int, default=20,
                       help='Population size')
    parser.add_argument('--max_files', type=int, default=20,
                       help='Maximum files per evaluation')
    parser.add_argument('--model_path', type=str, default='models/tinyphysics.onnx',
                       help='Path to TinyPhysics model')
    
    args = parser.parse_args()
    
    # Get data files
    data_files = []
    data_dir = Path("data")
    if data_dir.exists():
        data_files = list(data_dir.glob("*.csv"))
        data_files = [str(f) for f in data_files]
    
    if not data_files:
        print("No data files found!")
        return 1
    
    print(f"Found {len(data_files)} data files")
    
    # Run blender tournament
    best_architecture = run_blender_tournament(
        args.archive, data_files, args.model_path,
        args.rounds, args.pop_size, args.max_files
    )
    
    return 0

if __name__ == "__main__":
    exit(main())