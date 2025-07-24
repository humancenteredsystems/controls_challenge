#!/usr/bin/env python3

"""
Comprehensive Controller Performance Validation

Tests tournament-optimized controller against broad dataset to validate
performance beyond the 5 files used during tournament evolution.

Compares:
- Baseline (fallback controller)  
- Grid search winner (cost: 76.81)
- Tournament winner (cost: 58.95)

Uses GPU acceleration for efficient evaluation.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tinyphysics import TinyPhysicsModel
from optimization import generate_blended_controller

def load_tournament_winner():
    """Load tournament winner parameters"""
    with open('tournament_winner_params.json', 'r') as f:
        params = json.load(f)
    return params['low_gains'], params['high_gains'], params['cost']

def load_grid_search_winner():
    """Load grid search winner parameters"""
    with open('blended_2pid_comprehensive_results.json', 'r') as f:
        results = json.load(f)
    best_params = results['best_params']
    best_cost = results['best_cost']
    return best_params['low_gains'], best_params['high_gains'], best_cost

def evaluate_controller_performance(model, data_files: List[str], low_gains: List[float], 
                                  high_gains: List[float], controller_name: str, 
                                  max_files = None) -> Dict[str, Any]:
    """
    Evaluate controller performance across multiple data files.
    
    Args:
        model: TinyPhysicsModel instance (reused for efficiency)
        data_files: List of data file paths
        low_gains: Low-speed PID gains [P, I, D]
        high_gains: High-speed PID gains [P, I, D]  
        controller_name: Name for logging
        max_files: Maximum files to test (None = all files)
        
    Returns:
        Dictionary with performance statistics
    """
    if max_files:
        test_files = data_files[:max_files]
    else:
        test_files = data_files
        
    print(f"\n🧪 Testing {controller_name} on {len(test_files)} files...")
    
    # Generate controller code and write to temporary file
    controller_code = generate_blended_controller(low_gains, high_gains)
    
    # Create unique controller name to avoid Python import caching issues
    import time
    controller_id = f"temp_validation_{int(time.time() * 1000) % 100000}"
    temp_controller_file = f'controllers/{controller_id}.py'
    
    with open(temp_controller_file, 'w') as f:
        f.write(controller_code)
    
    # Use the unique controller module name
    controller = controller_id
    
    costs = []
    start_time = time.time()
    
    for i, data_file in enumerate(test_files):
        try:
            # Use the same data loading approach as the working optimizers
            from tinyphysics import run_rollout
            result = run_rollout(data_file, controller, model, debug=False)
            
            # run_rollout returns (rollout_dict, target_history, current_history)
            # Extract total_cost from the rollout dictionary
            if isinstance(result, tuple) and len(result) >= 1:
                rollout_dict = result[0]
                if isinstance(rollout_dict, dict) and 'total_cost' in rollout_dict:
                    cost_value = float(rollout_dict['total_cost'])
                else:
                    print(f"   ⚠️  Unexpected rollout format for {data_file}: {type(rollout_dict)}")
                    continue
            else:
                print(f"   ⚠️  Unexpected result format for {data_file}: {type(result)}")
                continue
            
            costs.append(cost_value)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_cost = np.mean(costs)
                print(f"   Progress: {i+1}/{len(test_files)} files, "
                      f"avg_cost={avg_cost:.2f}, elapsed={elapsed:.1f}s")
                
        except Exception as e:
            print(f"   ⚠️  Error processing {data_file}: {e}")
            continue
    
    # Calculate comprehensive statistics - now all elements are scalars
    if not costs:
        return {
            'controller_name': controller_name,
            'num_files': 0,
            'avg_total_cost': float('inf'),
            'std_cost': 0,
            'min_cost': float('inf'),
            'max_cost': float('inf'),
            'median_cost': float('inf'),
            'p75_cost': float('inf'),
            'p90_cost': float('inf'),
            'total_time': time.time() - start_time,
            'avg_time_per_file': 0,
            'low_gains': low_gains,
            'high_gains': high_gains
        }
    
    costs = np.array(costs)
    total_time = time.time() - start_time
    
    stats = {
        'controller_name': controller_name,
        'num_files': len(costs),
        'avg_total_cost': float(np.mean(costs)),
        'std_cost': float(np.std(costs)), 
        'min_cost': float(np.min(costs)),
        'max_cost': float(np.max(costs)),
        'median_cost': float(np.median(costs)),
        'p75_cost': float(np.percentile(costs, 75)),
        'p90_cost': float(np.percentile(costs, 90)),
        'total_time': total_time,
        'avg_time_per_file': total_time / len(costs),
        'low_gains': low_gains,
        'high_gains': high_gains
    }
    
    print(f"✅ {controller_name} Results:")
    print(f"   Files: {stats['num_files']}")
    print(f"   Avg Cost: {stats['avg_total_cost']:.2f} ± {stats['std_cost']:.2f}")
    print(f"   Range: [{stats['min_cost']:.2f}, {stats['max_cost']:.2f}]")
    print(f"   Median: {stats['median_cost']:.2f}")
    print(f"   Time: {stats['total_time']:.1f}s ({stats['avg_time_per_file']:.2f}s/file)")
    
    return stats

def run_comprehensive_validation(max_files: int = 50):
    """
    Run comprehensive validation across multiple controllers.
    
    Args:
        max_files: Maximum number of data files to test
    """
    print("🚀 COMPREHENSIVE CONTROLLER VALIDATION")
    print("="*60)
    
    # Initialize GPU-accelerated model (reused for efficiency)
    print("Initializing GPU-accelerated model...")
    model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=True)
    
    # Get data files
    data_dir = Path('./data')
    data_files = sorted([str(f) for f in data_dir.glob('*.csv')])
    
    if len(data_files) == 0:
        print("❌ No data files found in ./data directory")
        return
        
    print(f"Found {len(data_files)} data files")
    test_files = data_files[:max_files] if max_files else data_files
    print(f"Testing on {len(test_files)} files")
    
    # Load controller parameters
    try:
        tournament_low, tournament_high, tournament_cost = load_tournament_winner()
        print(f"Tournament winner loaded: cost={tournament_cost:.2f}")
    except Exception as e:
        print(f"❌ Failed to load tournament winner: {e}")
        return
        
    try:
        grid_low, grid_high, grid_cost = load_grid_search_winner()
        print(f"Grid search winner loaded: cost={grid_cost:.2f}")
    except Exception as e:
        print(f"❌ Failed to load grid search winner: {e}")
        return
    
    # Define baseline controller (fallback parameters)
    baseline_low = [0.3, 0.05, 0.1]   # Conservative fallback
    baseline_high = [0.15, 0.01, 0.05] # Conservative fallback
    
    results = []
    
    # Test 1: Baseline (Fallback) Controller
    print(f"\n" + "="*60)
    baseline_stats = evaluate_controller_performance(
        model, test_files, baseline_low, baseline_high, 
        "Baseline (Fallback)", max_files
    )
    results.append(baseline_stats)
    
    # Test 2: Grid Search Winner
    print(f"\n" + "="*60)
    grid_stats = evaluate_controller_performance(
        model, test_files, grid_low, grid_high,
        f"Grid Search Winner (cost={grid_cost:.2f})", max_files
    )
    results.append(grid_stats)
    
    # Test 3: Tournament Evolution Winner  
    print(f"\n" + "="*60)
    tournament_stats = evaluate_controller_performance(
        model, test_files, tournament_low, tournament_high,
        f"Tournament Winner (cost={tournament_cost:.2f})", max_files
    )
    results.append(tournament_stats)
    
    # Comparative Analysis
    print(f"\n" + "="*60)
    print("📊 COMPARATIVE ANALYSIS")
    print("="*60)
    
    baseline_cost = baseline_stats['avg_total_cost']
    grid_cost_actual = grid_stats['avg_total_cost'] 
    tournament_cost_actual = tournament_stats['avg_total_cost']
    
    grid_improvement = ((baseline_cost - grid_cost_actual) / baseline_cost) * 100
    tournament_improvement = ((baseline_cost - tournament_cost_actual) / baseline_cost) * 100
    tournament_vs_grid = ((grid_cost_actual - tournament_cost_actual) / grid_cost_actual) * 100
    
    print(f"Baseline (Fallback):     {baseline_cost:.2f}")
    print(f"Grid Search Winner:      {grid_cost_actual:.2f} ({grid_improvement:+.1f}%)")
    print(f"Tournament Winner:       {tournament_cost_actual:.2f} ({tournament_improvement:+.1f}%)")
    print(f"Tournament vs Grid:      {tournament_vs_grid:+.1f}% improvement")
    
    # Statistical significance
    print(f"\nStatistical Summary:")
    for stats in results:
        name = stats['controller_name']
        print(f"{name:25} | Cost: {stats['avg_total_cost']:.2f}±{stats['std_cost']:.2f} | "
              f"Range: [{stats['min_cost']:.2f}, {stats['max_cost']:.2f}]")
    
    # Save comprehensive results
    validation_results = {
        'validation_timestamp': time.time(),
        'num_files_tested': len(test_files),
        'max_files_limit': max_files,
        'controllers': results,
        'comparative_analysis': {
            'baseline_cost': baseline_cost,
            'grid_search_cost': grid_cost_actual,
            'tournament_cost': tournament_cost_actual,
            'grid_improvement_pct': grid_improvement,
            'tournament_improvement_pct': tournament_improvement,
            'tournament_vs_grid_pct': tournament_vs_grid
        }
    }
    
    results_file = 'comprehensive_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n✅ Comprehensive validation complete!")
    print(f"📄 Results saved to: {results_file}")
    print(f"🏆 Best Controller: Tournament Winner ({tournament_cost_actual:.2f} cost)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive Controller Validation')
    parser.add_argument('--max-files', type=int, default=50,
                       help='Maximum number of data files to test (default: 50)')
    args = parser.parse_args()
    
    run_comprehensive_validation(args.max_files)