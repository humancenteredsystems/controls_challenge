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

def create_training_data_from_archive(archive_path, data_files, model, num_samples=5000):
    """
    Generate training data for BlenderNet from PID tournament archive
    
    Args:
        archive_path: Path to tournament archive JSON
        data_files: List of data files for training data generation
        model: TinyPhysicsModel instance for rollout evaluation
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
                                               num_samples // len(top_performers),
                                               data_files, model)
        training_samples.extend(samples)
    
    print(f"Generated {len(training_samples)} training samples")
    
    # Save training data
    training_data_path = Path("plans/blender_training_data.json")
    with open(training_data_path, 'w') as f:
        json.dump(training_samples, f, indent=2)
    
    return training_samples

def generate_samples_for_pid_combo(pid1_params, pid2_params, num_samples, data_files, model):
    """Generate training samples via optimal blend weight discovery using existing run_rollout()"""
    samples = []
    
    for _ in range(num_samples):
        data_file = random.choice(data_files)
        
        # Find optimal blend weight using existing infrastructure
        best_blend = find_optimal_blend_weight(pid1_params, pid2_params, data_file, model)
        
        # Extract features using existing state extraction
        state, error, future_plan = extract_state_from_file(data_file)
        features = [state.v_ego, state.roll_lataccel, state.a_ego, error, 0, 0,
                   np.mean(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0,
                   np.std(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0]
        
        samples.append((features, best_blend))
    
    return samples

def find_optimal_blend_weight(pid1_params, pid2_params, data_file, model):
    """Find optimal blend using existing run_rollout() infrastructure"""
    from optimization import generate_blended_controller
    import tempfile
    import os
    
    best_cost, best_blend = float('inf'), 0.5
    
    for blend in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:  # Simple discrete search
        # Create temporary controller with fixed blend weight
        controller_content = generate_blended_controller(pid1_params, pid2_params).replace(
            'if v_ego < 40:', f'if False:  # Fixed blend weight: {blend}').replace(
            'weights = [0.8, 0.2]', f'weights = [{blend}, {1-blend}]').replace(
            'weights = [0.2, 0.8]', f'weights = [{blend}, {1-blend}]')
        
        # Create temp file
        temp_controller_name = f"temp_blend_{hashlib.md5(str(blend).encode()).hexdigest()[:8]}"
        temp_path = f"controllers/{temp_controller_name}.py"
        
        try:
            with open(temp_path, 'w') as f:
                f.write(controller_content)
            
            cost, _, _ = run_rollout(data_file, temp_controller_name, model)
            
            if cost['total_cost'] < best_cost:
                best_cost, best_blend = cost['total_cost'], blend
                
        except Exception as e:
            print(f"Error testing blend {blend}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return best_blend

def extract_state_from_file(data_file):
    """Extract random state from data file for training"""
    import pandas as pd
    
    df = pd.read_csv(data_file)
    
    # Pick random time step
    idx = random.randint(100, min(len(df) - 50, 400))  # Ensure we have future plan data
    
    # Create state tuple (matches tinyphysics.py format)
    from collections import namedtuple
    State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
    FuturePlan = namedtuple('FuturePlan', ['lataccel'])
    
    ACC_G = 9.81
    state = State(
        roll_lataccel=np.sin(df.iloc[idx]['roll']) * ACC_G,
        v_ego=df.iloc[idx]['vEgo'],
        a_ego=df.iloc[idx]['aEgo']
    )
    
    error = random.uniform(-1, 1)  # Simulate control error
    
    future_plan = FuturePlan(
        lataccel=df['targetLateralAcceleration'].iloc[idx:idx+20].tolist()
    )
    
    return state, error, future_plan

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
    Train actual BlenderNet using PyTorch
    
    Args:
        architecture: Dict with network architecture parameters
        training_data: List of (features, label) tuples
        epochs: Number of training epochs
    
    Returns:
        Path to trained ONNX model
    """
    from neural_blender_net import BlenderNet, train_blender_net
    
    model_id = architecture['id']
    onnx_path = f"models/blender_{model_id}.onnx"
    
    print(f"    Training architecture {model_id} for {epochs} epochs...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train actual PyTorch model
    trained_model = train_blender_net(training_data, epochs=epochs)
    
    # Export to ONNX for inference
    trained_model.export_to_onnx(onnx_path)
    
    return onnx_path

def evaluate_blender_architecture(architecture, training_data, data_files, model, tournament_2_baseline, max_files=20):
    """
    Evaluate BlenderNet architecture performance - only reward improvements over Tournament #2
    
    Args:
        architecture: Dict with network architecture
        training_data: Training data for this architecture
        data_files: List of data files for evaluation
        model: TinyPhysicsModel instance
        tournament_2_baseline: Best cost from Tournament #2 to beat
        max_files: Maximum files to evaluate on
    
    Returns:
        Average cost (penalized if no improvement over baseline)
    """
    
    # Train the architecture
    onnx_path = train_blender_architecture(architecture, training_data)
    
    # Get best PID parameters from archive for evaluation
    pid_pairs = get_top_pid_pairs_from_archive()
    
    total_costs = []
    
    # Evaluate on subset of data files
    eval_files = data_files[:max_files]
    
    for data_file in eval_files:
        for pid1_params, pid2_params in pid_pairs[:3]:  # Top 3 PID pairs
            
            # Create temporary neural controller using new pattern
            controller_module = _make_temp_neural_controller(pid1_params, pid2_params, onnx_path, architecture['id'])
            
            try:
                # Evaluate using existing run_rollout pattern
                cost, _, _ = run_rollout(data_file, controller_module, model)
                total_costs.append(cost["total_cost"])
                
            except Exception as e:
                print(f"    Evaluation failed: {e}")
                total_costs.append(1000)  # Penalty for failed evaluation
                
            finally:
                # Clean up temporary controller
                cleanup_temp_controller(controller_module)
    
    neural_cost = np.mean(total_costs) if total_costs else 1000
    
    # Only reward if better than Tournament #2 best (need 2+ point improvement for leaderboard targeting)
    improvement = tournament_2_baseline - neural_cost
    if improvement > 2.0:
        return neural_cost
    else:
        return 1000  # Penalty for not improving enough

def _make_temp_neural_controller(pid1_params, pid2_params, onnx_path, arch_id):
    """Create temporary neural controller using new pattern"""
    from optimization import generate_neural_blended_controller
    
    controller_content = generate_neural_blended_controller(pid1_params, pid2_params, onnx_path)
    module_name = f"temp_neural_{hashlib.md5((str(arch_id) + onnx_path).encode()).hexdigest()[:8]}"
    
    with open(f"controllers/{module_name}.py", "w") as f:
        f.write(controller_content)
    
    return module_name

def cleanup_temp_controller(module_name):
    """Clean up temporary controller file"""
    temp_path = f"controllers/{module_name}.py"
    if os.path.exists(temp_path):
        os.remove(temp_path)

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

def get_tournament_2_baseline(archive_path="plans/tournament_archive.json"):
    """Get best cost from Tournament #2 to beat"""
    try:
        with open(archive_path, 'r') as f:
            archive = json.load(f)
        
        # Find absolute best performer from Tournament #2
        valid_performers = [p for p in archive['archive']
                          if 'stats' in p and 'avg_total_cost' in p['stats']]
        
        if valid_performers:
            best_performer = min(valid_performers, key=lambda x: x['stats']['avg_total_cost'])
            tournament_2_best = best_performer['stats']['avg_total_cost']
            print(f"📊 Tournament #2 baseline to beat: {tournament_2_best:.2f}")
            return tournament_2_best, best_performer
        else:
            print("⚠️  No valid Tournament #2 results found, using default baseline")
            return 76.81, None
            
    except Exception as e:
        print(f"⚠️  Could not load Tournament #2 baseline: {e}, using 76.81")
        return 76.81, None


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
    Run tournament optimization for BlenderNet architectures - targeting leaderboard performance
    
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
    
    print("🏆 Starting Blender Tournament Optimization (Target: <45 cost for leaderboard)")
    print("=" * 80)
    
    # Get Tournament #2 baseline to beat
    tournament_2_baseline, best_tournament_2 = get_tournament_2_baseline(archive_path)
    
    # Create GPU-accelerated model instance (follows tinyphysics.py pattern)
    model = TinyPhysicsModel(model_path, debug=False)
    print("Blender tournament: GPU ENABLED")
    
    # Generate training data from PID archive
    training_data = create_training_data_from_archive(archive_path, data_files, model)
    
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
    print(f"  - Tournament #2 baseline to beat: {tournament_2_baseline:.2f}")
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
                    architecture, training_data, data_files, model, tournament_2_baseline, max_files
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
            
            improvement = tournament_2_baseline - best_ever_cost
            print(f"🎉 New tournament best: {best_ever_cost:.2f} (improvement: +{improvement:.2f})")
            print(f"   Architecture: {best_ever_architecture['hidden_sizes']}")
            print(f"   Dropout: {best_ever_architecture['dropout_rate']}")
        
        # Selection and mutation for next round
        if round_num < rounds:
            population = tournament_selection_and_mutation(population)
            print(f"  Round {round_num} complete: generation evolved")
        else:
            print(f"  Round {round_num} complete: final round")
    
    print()
    print("🎯 Stage 2d Results:")
    print(f"  Tournament #2 best: {tournament_2_baseline:.2f}")
    print(f"  Neural blending best: {best_ever_cost:.2f}")
    improvement = tournament_2_baseline - best_ever_cost
    print(f"  Stage 2d improvement: {improvement:.2f} points")
    
    if best_ever_cost < tournament_2_baseline:
        print("✅ SUCCESS: Neural blending improved over Tournament #2!")
        # Create champion controller
        create_champion_controller(best_ever_architecture, best_tournament_2, best_ever_cost, archive_path)
    else:
        print("⚠️  WARNING: Neural blending did not improve over Tournament #2")
    
    # Save results
    results = {
        'champion_cost': best_ever_cost,
        'tournament_2_baseline': tournament_2_baseline,
        'improvement': improvement,
        'best_architecture': best_ever_architecture,
        'best_pid_params': {
            'low_gains': best_tournament_2['low_gains'] if best_tournament_2 else None,
            'high_gains': best_tournament_2['high_gains'] if best_tournament_2 else None
        },
        'tournament_config': {
            'rounds': rounds,
            'pop_size': pop_size,
            'max_files': max_files
        }
    }
    
    results_path = "plans/stage_2d_champion_results.json"
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

def create_champion_controller(best_architecture, best_tournament_2, champion_cost, archive_path):
    """Create final champion controller ready for eval.py"""
    
    print(f"🏆 Creating champion controller...")
    
    # Train the champion model
    training_data_path = "plans/blender_training_data.json"
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Train final champion model
    champion_onnx_path = train_blender_architecture(best_architecture, training_data)
    
    # Move to champion location
    import shutil
    final_onnx_path = "models/neural_blender_champion.onnx"
    shutil.copy2(champion_onnx_path, final_onnx_path)
    
    # Get best PID parameters
    if best_tournament_2:
        pid1_params = best_tournament_2['low_gains']
        pid2_params = best_tournament_2['high_gains']
    else:
        # Fallback to best from archive
        pid_pairs = get_top_pid_pairs_from_archive(archive_path)
        pid1_params, pid2_params = pid_pairs[0]
    
    # Create champion controller file
    from optimization import generate_neural_blended_controller
    
    champion_controller_content = f'''from controllers import BaseController
from controllers.shared_pid import SpecializedPID
import onnxruntime as ort
import numpy as np

class Controller(BaseController):
    """Champion Neural Blended Controller - Stage 2d Winner (Cost: {champion_cost:.2f})"""
    
    def __init__(self):
        # Best PID parameters from Tournament #2
        self.pid1 = SpecializedPID({pid1_params[0]}, {pid1_params[1]}, {pid1_params[2]}, "Champion_PID1")
        self.pid2 = SpecializedPID({pid2_params[0]}, {pid2_params[1]}, {pid2_params[2]}, "Champion_PID2")
        
        # Champion trained neural blender
        self.blender_session = ort.InferenceSession(
            "models/neural_blender_champion.onnx",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        print(f"🏆 Champion Neural Blended Controller loaded (cost: {champion_cost:.2f})")
        print(f"  PID1: P={pid1_params[0]:.3f}, I={pid1_params[1]:.3f}, D={pid1_params[2]:.3f}")
        print(f"  PID2: P={pid2_params[0]:.3f}, I={pid2_params[1]:.3f}, D={pid2_params[2]:.3f}")
        print(f"  Neural Architecture: {best_architecture['hidden_sizes']}")
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        pid1_output = self.pid1.update(error)
        pid2_output = self.pid2.update(error)
        
        # Neural blending using champion model
        features = np.array([[state.v_ego, state.roll_lataccel, state.a_ego, error,
                             self.pid1.error_integral, error - self.pid1.prev_error,
                             np.mean(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0,
                             np.std(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0]], dtype=np.float32)
        
        blend_weight = self.blender_session.run(None, {{'input': features}})[0][0]
        blend_weight = np.clip(float(blend_weight), 0.0, 1.0)
        
        return blend_weight * pid1_output + (1 - blend_weight) * pid2_output
    
    def __repr__(self):
        return f"ChampionNeuralBlendedController(cost={champion_cost:.2f})"
'''
    
    # Write champion controller
    champion_controller_path = "controllers/neural_blended_champion.py"
    with open(champion_controller_path, 'w') as f:
        f.write(champion_controller_content)
    
    print(f"✅ Champion controller created: {champion_controller_path}")
    print(f"✅ Champion model saved: {final_onnx_path}")
    print(f"🎯 Ready for eval.py: --test_controller neural_blended_champion")

if __name__ == "__main__":
    exit(main())