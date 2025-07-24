#!/usr/bin/env python3
"""
Simplified Blender Optimizer - Tests different blending strategies without neural networks
Focuses on optimizing static blend weights and speed-based thresholds
"""

import sys
import os
import argparse
import json
import random
import numpy as np
from pathlib import Path

# Add parent directory to path to find tinyphysics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tinyphysics import run_rollout, TinyPhysicsModel

def get_top_pid_pairs_from_archive(archive_path="plans/tournament_archive.json"):
    """Get top PID parameter pairs from tournament archive"""
    
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    # Filter performers with valid stats
    valid_performers = []
    for p in archive['archive']:
        if 'stats' in p and 'avg_total_cost' in p['stats']:
            valid_performers.append(p)
    
    if len(valid_performers) == 0:
        # Fallback to default parameters
        return [([0.25, 0.12, -0.05], [0.15, 0.08, -0.15])]
    
    # Get best performer (Tournament #2 winner)
    best_performer = min(valid_performers, key=lambda x: x['stats']['avg_total_cost'])
    
    return [(best_performer['low_gains'], best_performer['high_gains'])]

def create_blending_strategy(strategy_params):
    """
    Create a blending strategy function based on parameters
    
    Args:
        strategy_params: Dict with strategy configuration
        
    Returns:
        Function that takes vehicle state and returns blend weight [0,1]
    """
    
    strategy_type = strategy_params.get('type', 'speed_threshold')
    
    if strategy_type == 'speed_threshold':
        # Simple speed threshold blending
        threshold = strategy_params.get('speed_threshold', 40)
        low_weight = strategy_params.get('low_speed_weight', 0.8)
        high_weight = strategy_params.get('high_speed_weight', 0.2)
        
        def blend_func(v_ego, roll_lataccel, a_ego, error):
            return low_weight if v_ego < threshold else high_weight
            
    elif strategy_type == 'linear_speed':
        # Linear interpolation based on speed
        min_speed = strategy_params.get('min_speed', 10)
        max_speed = strategy_params.get('max_speed', 60)
        min_weight = strategy_params.get('min_weight', 0.9)
        max_weight = strategy_params.get('max_weight', 0.1)
        
        def blend_func(v_ego, roll_lataccel, a_ego, error):
            if v_ego <= min_speed:
                return min_weight
            elif v_ego >= max_speed:
                return max_weight
            else:
                # Linear interpolation
                t = (v_ego - min_speed) / (max_speed - min_speed)
                return min_weight + t * (max_weight - min_weight)
                
    elif strategy_type == 'error_adaptive':
        # Adaptive blending based on error magnitude
        base_threshold = strategy_params.get('speed_threshold', 35)
        error_sensitivity = strategy_params.get('error_sensitivity', 0.3)
        
        def blend_func(v_ego, roll_lataccel, a_ego, error):
            # Base blend from speed
            base_blend = 0.7 if v_ego < base_threshold else 0.3
            
            # Adjust based on error magnitude
            error_adjustment = error_sensitivity * min(abs(error), 1.0)
            
            # Higher error -> favor more responsive controller (lower blend weight)
            adjusted_blend = base_blend - error_adjustment
            
            return max(0.0, min(1.0, adjusted_blend))
    
    else:
        # Fallback to fixed blend
        fixed_weight = strategy_params.get('fixed_weight', 0.5)
        def blend_func(v_ego, roll_lataccel, a_ego, error):
            return fixed_weight
    
    return blend_func

def create_temp_blended_controller(pid1_params, pid2_params, blending_strategy, strategy_id):
    """Create temporary blended controller with custom blending strategy"""
    
    controller_content = f'''from controllers.blended_2pid import SpecializedPID
from controllers import BaseController

class Controller(BaseController):
    def __init__(self):
        # Initialize PID controllers
        self.pid1 = SpecializedPID({pid1_params[0]}, {pid1_params[1]}, {pid1_params[2]}, "PID1")
        self.pid2 = SpecializedPID({pid2_params[0]}, {pid2_params[1]}, {pid2_params[2]}, "PID2")
        
        # Blending strategy function
        self.get_blend_weight = self._create_blend_func()
        
        print(f"Custom Blended Controller initialized (strategy: {strategy_id})")
    
    def _create_blend_func(self):
        # Strategy implementation
        {blending_strategy}
        return blend_func
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        
        # Get PID outputs
        pid1_output = self.pid1.update(error)
        pid2_output = self.pid2.update(error)
        
        # Get blend weight from strategy
        blend_weight = self.get_blend_weight(state.v_ego, state.roll_lataccel, state.a_ego, error)
        
        # Blend outputs
        return blend_weight * pid1_output + (1 - blend_weight) * pid2_output
'''
    
    temp_path = f"controllers/temp_blend_strategy_{strategy_id}.py"
    
    with open(temp_path, 'w') as f:
        f.write(controller_content)
    
    return temp_path

def evaluate_blending_strategy(strategy_params, pid1_params, pid2_params, data_files, model, max_files=20):
    """Evaluate a blending strategy"""
    
    strategy_id = strategy_params.get('id', 'unknown')
    
    # Create blending function code
    blend_func = create_blending_strategy(strategy_params)
    
    # Generate function code for embedding in controller
    if strategy_params.get('type') == 'speed_threshold':
        blend_code = f'''
        def blend_func(v_ego, roll_lataccel, a_ego, error):
            return {strategy_params.get('low_speed_weight', 0.8)} if v_ego < {strategy_params.get('speed_threshold', 40)} else {strategy_params.get('high_speed_weight', 0.2)}
        '''
    elif strategy_params.get('type') == 'linear_speed':
        blend_code = f'''
        def blend_func(v_ego, roll_lataccel, a_ego, error):
            min_speed, max_speed = {strategy_params.get('min_speed', 10)}, {strategy_params.get('max_speed', 60)}
            min_weight, max_weight = {strategy_params.get('min_weight', 0.9)}, {strategy_params.get('max_weight', 0.1)}
            if v_ego <= min_speed:
                return min_weight
            elif v_ego >= max_speed:
                return max_weight  
            else:
                t = (v_ego - min_speed) / (max_speed - min_speed)
                return min_weight + t * (max_weight - min_weight)
        '''
    else:
        blend_code = f'''
        def blend_func(v_ego, roll_lataccel, a_ego, error):
            return {strategy_params.get('fixed_weight', 0.5)}
        '''
    
    # Create temporary controller
    temp_controller_path = create_temp_blended_controller(
        pid1_params, pid2_params, blend_code, strategy_id
    )
    
    total_cost = 0
    num_evaluations = 0
    
    try:
        # Evaluate on subset of data files
        eval_files = data_files[:max_files]
        
        for data_file in eval_files:
            try:
                # Extract cost from rollout result
                rollout_result = run_rollout(data_file, f"temp_blend_strategy_{strategy_id}", model, debug=False)
                
                if isinstance(rollout_result, tuple):
                    cost = rollout_result[0].get('total_cost', 1000)
                else:
                    cost = rollout_result
                
                total_cost += cost
                num_evaluations += 1
                
            except Exception as e:
                print(f"    Evaluation failed on {data_file}: {e}")
                total_cost += 1000  # Penalty
                num_evaluations += 1
        
    finally:
        # Clean up temporary controller
        if os.path.exists(temp_controller_path):
            os.remove(temp_controller_path)
    
    return total_cost / num_evaluations if num_evaluations > 0 else 1000

def generate_strategy_population(pop_size=20, generation=0):
    """Generate population of blending strategies"""
    
    population = []
    
    for i in range(pop_size):
        strategy_type = random.choice(['speed_threshold', 'linear_speed', 'error_adaptive'])
        
        if strategy_type == 'speed_threshold':
            strategy = {
                'type': 'speed_threshold',
                'speed_threshold': random.uniform(25, 50),
                'low_speed_weight': random.uniform(0.6, 0.9),
                'high_speed_weight': random.uniform(0.1, 0.4),
                'id': f'thresh_{i:02d}_g{generation}'
            }
        elif strategy_type == 'linear_speed':
            min_speed = random.uniform(5, 20)
            max_speed = random.uniform(45, 70)
            strategy = {
                'type': 'linear_speed',
                'min_speed': min_speed,
                'max_speed': max_speed,
                'min_weight': random.uniform(0.7, 0.95),
                'max_weight': random.uniform(0.05, 0.3),
                'id': f'linear_{i:02d}_g{generation}'
            }
        else:  # error_adaptive
            strategy = {
                'type': 'error_adaptive',
                'speed_threshold': random.uniform(30, 45),
                'error_sensitivity': random.uniform(0.1, 0.5),
                'id': f'adaptive_{i:02d}_g{generation}'
            }
        
        strategy['cost'] = float('inf')
        strategy['generation'] = generation
        population.append(strategy)
    
    return population

def run_simple_blender_optimization(archive_path, data_files, model_path, rounds=10, pop_size=20, max_files=20):
    """Run simplified blender optimization"""
    
    print("🏆 Starting Simple Blender Optimization")
    print("=" * 60)
    
    # Create GPU model instance
    model = TinyPhysicsModel(model_path, debug=False)
    print("Simple blender optimizer: GPU ENABLED")
    
    # Get best PID parameters from Tournament #2
    pid_pairs = get_top_pid_pairs_from_archive(archive_path)
    pid1_params, pid2_params = pid_pairs[0]
    
    print(f"Using Tournament #2 winner PID parameters:")
    print(f"  PID1 (low-speed):  P={pid1_params[0]:.3f}, I={pid1_params[1]:.3f}, D={pid1_params[2]:.3f}")
    print(f"  PID2 (high-speed): P={pid2_params[0]:.3f}, I={pid2_params[1]:.3f}, D={pid2_params[2]:.3f}")
    
    # Generate initial population of blending strategies
    population = generate_strategy_population(pop_size, generation=0)
    
    print(f"\nStarting blender optimization:")
    print(f"  - {rounds} rounds")
    print(f"  - Population size: {pop_size}")
    print(f"  - {max_files} files per evaluation")
    print(f"  - {len(data_files)} total data files available")
    print()
    
    best_ever_cost = float('inf')
    best_ever_strategy = None
    
    # Optimization rounds
    for round_num in range(1, rounds + 1):
        print(f"🏆 Blend Optimization Round {round_num}/{rounds}")
        
        # Evaluate each strategy
        for i, strategy in enumerate(population):
            if strategy['cost'] == float('inf'):  # Not yet evaluated
                cost = evaluate_blending_strategy(
                    strategy, pid1_params, pid2_params, data_files, model, max_files
                )
                strategy['cost'] = cost
                
                if cost < best_ever_cost:
                    best_ever_cost = cost
                    best_ever_strategy = strategy.copy()
            
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1}/{pop_size} strategies...")
        
        # Find best in this round
        round_best = min(population, key=lambda x: x['cost'])
        
        print(f"  Round {round_num} best cost: {round_best['cost']:.2f} (strategy: {round_best['id']})")
        
        if round_best['cost'] < best_ever_cost:
            best_ever_cost = round_best['cost']
            best_ever_strategy = round_best.copy()
            
            print(f"🎉 New overall best strategy: {best_ever_cost:.2f}")
            print(f"   Type: {best_ever_strategy['type']}")
            print(f"   ID: {best_ever_strategy['id']}")
        
        print(f"  Current total_cost: {best_ever_cost:.2f}")
        
        # Evolution for next round (simple mutation)
        if round_num < rounds:
            population = evolve_strategies(population, generation=round_num + 1)
            print(f"  Round {round_num} complete: strategies evolved")
        else:
            print(f"  Round {round_num} complete: final round")
        
        print("-" * 40)
    
    print()
    print("🏆 Simple Blender Optimization Complete!")
    print(f"Best strategy cost: {best_ever_cost:.2f}")
    print(f"Best strategy: {best_ever_strategy}")
    
    # Save results
    results = {
        'best_strategy': best_ever_strategy,
        'best_cost': best_ever_cost,
        'optimization_config': {
            'rounds': rounds,
            'pop_size': pop_size,
            'max_files': max_files
        },
        'pid_parameters': {
            'pid1': pid1_params,
            'pid2': pid2_params
        }
    }
    
    results_path = "plans/simple_blender_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return best_ever_strategy

def evolve_strategies(population, generation=1):
    """Evolve strategies for next generation"""
    
    # Sort by cost
    population.sort(key=lambda x: x['cost'])
    
    # Keep top 30% as elites
    elite_count = int(len(population) * 0.3)
    new_population = population[:elite_count].copy()
    
    # Generate offspring through mutation
    while len(new_population) < len(population):
        # Select parent from top 50%
        parent = random.choice(population[:len(population)//2])
        
        # Mutate
        child = mutate_strategy(parent, generation)
        child['cost'] = float('inf')  # Will be evaluated
        
        new_population.append(child)
    
    return new_population

def mutate_strategy(parent, generation=1):
    """Create mutated version of strategy"""
    
    child = parent.copy()
    # Generate new ID based on type and generation instead of appending '_m'
    strategy_type = parent['type']
    if strategy_type == 'speed_threshold':
        child['id'] = f'thresh_mut_{generation}_{random.randint(1000, 9999)}'
    elif strategy_type == 'linear_speed':
        child['id'] = f'linear_mut_{generation}_{random.randint(1000, 9999)}'
    else:  # error_adaptive
        child['id'] = f'adaptive_mut_{generation}_{random.randint(1000, 9999)}'
    
    child['generation'] = generation
    
    if parent['type'] == 'speed_threshold':
        # Mutate threshold and weights
        child['speed_threshold'] = max(20, min(60, parent['speed_threshold'] + random.gauss(0, 5)))
        child['low_speed_weight'] = max(0.5, min(1.0, parent['low_speed_weight'] + random.gauss(0, 0.1)))
        child['high_speed_weight'] = max(0.0, min(0.5, parent['high_speed_weight'] + random.gauss(0, 0.1)))
        
    elif parent['type'] == 'linear_speed':
        # Mutate speed range and weights
        child['min_speed'] = max(5, min(25, parent['min_speed'] + random.gauss(0, 3)))
        child['max_speed'] = max(40, min(80, parent['max_speed'] + random.gauss(0, 5)))
        child['min_weight'] = max(0.5, min(1.0, parent['min_weight'] + random.gauss(0, 0.1)))
        child['max_weight'] = max(0.0, min(0.5, parent['max_weight'] + random.gauss(0, 0.1)))
        
    elif parent['type'] == 'error_adaptive':
        # Mutate adaptive parameters
        child['speed_threshold'] = max(20, min(50, parent['speed_threshold'] + random.gauss(0, 3)))
        child['error_sensitivity'] = max(0.05, min(0.8, parent['error_sensitivity'] + random.gauss(0, 0.1)))
    
    return child

def main():
    parser = argparse.ArgumentParser(description='Simple Blender Optimizer')
    
    parser.add_argument('--archive', type=str, default='plans/tournament_archive.json',
                       help='Path to tournament archive')
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of optimization rounds')
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
    
    # Run optimization
    best_strategy = run_simple_blender_optimization(
        args.archive, data_files, args.model_path,
        args.rounds, args.pop_size, args.max_files
    )
    
    return 0

if __name__ == "__main__":
    exit(main())