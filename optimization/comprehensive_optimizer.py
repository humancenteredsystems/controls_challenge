"""
Comprehensive optimization engine for ensemble controllers
"""
import itertools
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import time
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyphysics import run_rollout

class ComprehensiveOptimizer:
    """Enhanced grid search optimization for ensemble controllers"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.results = []
        self.best_cost = float('inf')
        self.best_params = None
        
    def define_comprehensive_search_space(self, num_combinations: int = 250) -> List[Tuple[List[float], List[float], List[float]]]:
        """Define a comprehensive parameter search space with more granular ranges"""
        
        print(f"Generating comprehensive search space for {num_combinations} combinations...")
        
        # More granular parameter ranges for P, I, D
        low_p_values = np.linspace(0.25, 0.6, 8)    # 8 values from 0.25 to 0.6
        low_i_values = np.linspace(0.01, 0.12, 6)   # 6 values from 0.01 to 0.12
        low_d_values = np.linspace(-0.25, -0.05, 6) # 6 values from -0.25 to -0.05
        
        high_p_values = np.linspace(0.15, 0.4, 8)   # 8 values from 0.15 to 0.4
        high_i_values = np.linspace(0.005, 0.08, 6) # 6 values from 0.005 to 0.08
        high_d_values = np.linspace(-0.15, -0.03, 6)# 6 values from -0.15 to -0.03
        
        dyn_p_values = np.linspace(0.3, 0.8, 8)     # 8 values from 0.3 to 0.8
        dyn_i_values = np.linspace(0.02, 0.15, 6)   # 6 values from 0.02 to 0.15
        dyn_d_values = np.linspace(-0.3, -0.08, 6)  # 6 values from -0.3 to -0.08
        
        # Generate systematic combinations
        combinations = []
        
        # Strategy 1: Full factorial of reduced spaces (core search)
        low_core = [(0.3, 0.03, -0.1), (0.35, 0.05, -0.15), (0.4, 0.04, -0.12)]
        high_core = [(0.2, 0.01, -0.05), (0.25, 0.02, -0.08), (0.3, 0.015, -0.06)]
        dyn_core = [(0.4, 0.08, -0.1), (0.5, 0.1, -0.15), (0.6, 0.12, -0.2)]
        
        for low in low_core:
            for high in high_core:
                for dyn in dyn_core:
                    combinations.append((list(low), list(high), list(dyn)))
        
        # Strategy 2: Random sampling from refined ranges around best known values
        np.random.seed(42)  # For reproducible results
        
        # Best known ranges (from previous optimization)
        best_low = [0.3, 0.03, -0.1]
        best_high = [0.2, 0.01, -0.05] 
        best_dyn = [0.4, 0.1, -0.1]
        
        # Add variations around best known values
        for _ in range(num_combinations - len(combinations)):
            if len(combinations) >= num_combinations:
                break
                
            # Gaussian perturbation around best known values
            low_p = np.clip(np.random.normal(best_low[0], 0.08), 0.2, 0.6)
            low_i = np.clip(np.random.normal(best_low[1], 0.02), 0.01, 0.12)
            low_d = np.clip(np.random.normal(best_low[2], 0.04), -0.25, -0.05)
            
            high_p = np.clip(np.random.normal(best_high[0], 0.06), 0.15, 0.4)
            high_i = np.clip(np.random.normal(best_high[1], 0.015), 0.005, 0.08)
            high_d = np.clip(np.random.normal(best_high[2], 0.03), -0.15, -0.03)
            
            dyn_p = np.clip(np.random.normal(best_dyn[0], 0.1), 0.3, 0.8)
            dyn_i = np.clip(np.random.normal(best_dyn[1], 0.03), 0.02, 0.15)
            dyn_d = np.clip(np.random.normal(best_dyn[2], 0.05), -0.3, -0.08)
            
            low_gains = [round(low_p, 3), round(low_i, 3), round(low_d, 3)]
            high_gains = [round(high_p, 3), round(high_i, 3), round(high_d, 3)]
            dyn_gains = [round(dyn_p, 3), round(dyn_i, 3), round(dyn_d, 3)]
            
            combinations.append((low_gains, high_gains, dyn_gains))
        
        # Strategy 3: Add some extreme combinations to explore boundaries
        extreme_combinations = [
            ([0.6, 0.01, -0.05], [0.15, 0.005, -0.03], [0.8, 0.02, -0.08]),  # Very aggressive
            ([0.25, 0.12, -0.25], [0.4, 0.08, -0.15], [0.3, 0.15, -0.3]),   # Very conservative
            ([0.45, 0.06, -0.18], [0.25, 0.03, -0.09], [0.5, 0.09, -0.12]), # Balanced
        ]
        
        for combo in extreme_combinations:
            if len(combinations) < num_combinations:
                combinations.append(combo)
        
        print(f"Generated {len(combinations)} parameter combinations")
        return combinations[:num_combinations]
    
    def test_controller_combination(self, data_files: List[str], low_gains: List[float], 
                                   high_gains: List[float], dynamic_gains: List[float], 
                                   max_files: int = 25) -> Dict[str, float]:
        """Test a single controller combination with enhanced stability"""
        
        # Create temporary controller file
        controller_content = f'''from . import BaseController

class SpecializedPID:
    def __init__(self, p, i, d):
        self.p, self.i, self.d = p, i, d
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, error):
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff

class Controller(BaseController):
    def __init__(self):
        self.low_speed_pid = SpecializedPID({low_gains[0]}, {low_gains[1]}, {low_gains[2]})
        self.high_speed_pid = SpecializedPID({high_gains[0]}, {high_gains[1]}, {high_gains[2]})
        self.dynamic_pid = SpecializedPID({dynamic_gains[0]}, {dynamic_gains[1]}, {dynamic_gains[2]})
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        v_ego = state.v_ego
        
        low_output = self.low_speed_pid.update(error)
        high_output = self.high_speed_pid.update(error)
        dynamic_output = self.dynamic_pid.update(error)
        
        # Enhanced rule-based blending
        if v_ego < 20:  # Very low speed
            weights = [0.8, 0.1, 0.1]
        elif v_ego < 40:  # Low speed
            weights = [0.6, 0.3, 0.1]
        elif v_ego > 70:  # High speed
            weights = [0.05, 0.8, 0.15]
        elif v_ego > 50:  # Medium-high speed
            weights = [0.2, 0.6, 0.2]
        else:  # Medium speed
            weights = [0.4, 0.4, 0.2]
            
        # Adjust for dynamic scenarios
        if abs(target_lataccel) > 1.0:  # Very sharp turns
            weights = [0.1, 0.2, 0.7]
        elif abs(target_lataccel) > 0.5:  # Moderate turns
            weights[2] = min(weights[2] + 0.2, 0.6)
            weights[0] = max(weights[0] - 0.1, 0.1)
            weights[1] = 1.0 - weights[0] - weights[2]
        
        return sum(w * o for w, o in zip(weights, [low_output, high_output, dynamic_output]))
'''
        
        # Get the correct path to controllers directory
        base_dir = Path(__file__).parent.parent
        
        # Use unique filename to avoid Python module caching issues
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        controller_name = f'temp_test_{unique_id}'
        temp_controller_path = base_dir / 'controllers' / f'{controller_name}.py'
        
        # Write temporary controller
        with open(temp_controller_path, 'w') as f:
            f.write(controller_content)
        
        # Test on multiple files
        total_costs = []
        successful_tests = 0
        
        for data_file in data_files[:max_files]:
            try:
                cost, _, _ = run_rollout(data_file, controller_name, self.model_path, debug=False)
                total_costs.append(cost['total_cost'])
                successful_tests += 1
            except Exception as e:
                continue
        
        # Clean up
        if os.path.exists(temp_controller_path):
            os.remove(temp_controller_path)
        
        if successful_tests == 0:
            return {'avg_total_cost': float('inf'), 'num_files': 0}
            
        avg_cost = np.mean(total_costs)
        return {
            'avg_total_cost': avg_cost,
            'num_files': successful_tests,
            'std_cost': np.std(total_costs),
            'min_cost': np.min(total_costs),
            'max_cost': np.max(total_costs)
        }
    
    def optimize_comprehensive(self, data_files: List[str], num_combinations: int = 250, 
                             max_files_per_test: int = 25) -> Dict[str, Any]:
        """Run comprehensive optimization"""
        
        print(f"Starting comprehensive optimization:")
        print(f"  - {num_combinations} parameter combinations")
        print(f"  - {max_files_per_test} data files per test")
        print(f"  - {len(data_files)} total data files available")
        
        # Get search space
        combinations = self.define_comprehensive_search_space(num_combinations)
        
        # Test each combination
        results = []
        progress_file = 'optimization_progress.json'
        
        for i, (low_gains, high_gains, dynamic_gains) in enumerate(tqdm(combinations, desc="Testing combinations")):
            try:
                start_time = time.time()
                
                result = self.test_controller_combination(
                    data_files, low_gains, high_gains, dynamic_gains, max_files_per_test
                )
                
                test_time = time.time() - start_time
                
                if result['avg_total_cost'] != float('inf'):
                    cost = result['avg_total_cost']
                    
                    result_entry = {
                        'combination_id': i,
                        'low_gains': low_gains,
                        'high_gains': high_gains,
                        'dynamic_gains': dynamic_gains,
                        'avg_total_cost': cost,
                        'num_files': result['num_files'],
                        'std_cost': result.get('std_cost', 0),
                        'min_cost': result.get('min_cost', cost),
                        'max_cost': result.get('max_cost', cost),
                        'test_time': test_time
                    }
                    
                    results.append(result_entry)
                    
                    if cost < self.best_cost:
                        self.best_cost = cost
                        self.best_params = (low_gains, high_gains, dynamic_gains)
                        print(f"\n🎉 New best cost: {self.best_cost:.2f}")
                        print(f"   Low-speed:  P={low_gains[0]:.3f}, I={low_gains[1]:.3f}, D={low_gains[2]:.3f}")
                        print(f"   High-speed: P={high_gains[0]:.3f}, I={high_gains[1]:.3f}, D={high_gains[2]:.3f}")
                        print(f"   Dynamic:    P={dynamic_gains[0]:.3f}, I={dynamic_gains[1]:.3f}, D={dynamic_gains[2]:.3f}")
                        
                        # Save best params immediately
                        best_params = {
                            'low_speed_gains': low_gains,
                            'high_speed_gains': high_gains,
                            'dynamic_gains': dynamic_gains,
                            'best_cost': self.best_cost,
                            'combination_id': i,
                            'timestamp': time.time()
                        }
                        with open('best_params_temp.json', 'w') as f:
                            json.dump(best_params, f, indent=2)
                
                # Save progress every 25 combinations
                if (i + 1) % 25 == 0:
                    progress = {
                        'completed': i + 1,
                        'total': len(combinations),
                        'best_cost': self.best_cost,
                        'results_so_far': len(results)
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f, indent=2)
                        
            except Exception as e:
                print(f"Error testing combination {i}: {e}")
                continue
        
        # Sort results by cost
        results.sort(key=lambda x: x['avg_total_cost'])
        
        optimization_results = {
            'best_cost': self.best_cost,
            'best_params': {
                'low_gains': self.best_params[0] if self.best_params else None,
                'high_gains': self.best_params[1] if self.best_params else None,
                'dynamic_gains': self.best_params[2] if self.best_params else None
            },
            'all_results': results,
            'num_combinations_tested': len(results),
            'total_combinations_attempted': len(combinations),
            'success_rate': len(results) / len(combinations) if combinations else 0
        }
        
        return optimization_results
    
    def save_comprehensive_results(self, results: Dict[str, Any], 
                                 filename: str = "comprehensive_optimization_results.json"):
        """Save comprehensive optimization results"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Comprehensive results saved to {filename}")
    
    def print_top_results(self, results: Dict[str, Any], top_n: int = 10):
        """Print top N results with enhanced statistics"""
        print(f"\n🏆 Top {top_n} combinations:")
        print("=" * 100)
        
        for i, result in enumerate(results['all_results'][:top_n]):
            print(f"#{i+1} - Total Cost: {result['avg_total_cost']:.2f} (σ={result.get('std_cost', 0):.2f})")
            print(f"     Low-speed:  P={result['low_gains'][0]:.3f}, I={result['low_gains'][1]:.3f}, D={result['low_gains'][2]:.3f}")
            print(f"     High-speed: P={result['high_gains'][0]:.3f}, I={result['high_gains'][1]:.3f}, D={result['high_gains'][2]:.3f}")
            print(f"     Dynamic:    P={result['dynamic_gains'][0]:.3f}, I={result['dynamic_gains'][1]:.3f}, D={result['dynamic_gains'][2]:.3f}")
            print(f"     Files: {result['num_files']}, Range: [{result.get('min_cost', 0):.1f}, {result.get('max_cost', 0):.1f}]")
            print()

def main():
    """Main comprehensive optimization routine"""
    base_dir = Path(__file__).parent.parent
    model_path = str(base_dir / "models" / "tinyphysics.onnx")
    data_dir = base_dir / "data"
    
    # Get more data files for better testing
    data_files = [str(f) for f in sorted(data_dir.glob("*.csv"))[:50]]
    
    print(f"Found {len(data_files)} data files")
    
    # Create optimizer
    optimizer = ComprehensiveOptimizer(model_path)
    
    # Run comprehensive optimization
    print("Starting comprehensive parameter optimization...")
    results = optimizer.optimize_comprehensive(
        data_files, 
        num_combinations=300,  # Test 300 combinations
        max_files_per_test=25   # Use 25 files per test for better stability
    )
    
    # Print and save results
    optimizer.print_top_results(results, top_n=15)
    optimizer.save_comprehensive_results(results)
    
    # Update optimal parameters if we found better ones
    if results['best_params']['low_gains']:
        optimal_params = {
            'low_speed_gains': results['best_params']['low_gains'],
            'high_speed_gains': results['best_params']['high_gains'],
            'dynamic_gains': results['best_params']['dynamic_gains'],
            'best_cost': results['best_cost'],
            'baseline_cost': 106.9,
            'improvement': 106.9 - results['best_cost'],
            'optimization_type': 'comprehensive',
            'num_combinations_tested': results['num_combinations_tested']
        }
        
        with open('optimal_params.json', 'w') as f:
            json.dump(optimal_params, f, indent=2)
        
        print(f"\n✅ Updated optimal parameters!")
        print(f"   Best cost: {results['best_cost']:.2f}")
        print(f"   Target: <45.0")
        print(f"   Progress: {max(0, 119.46 - results['best_cost']):.2f} points improved from previous")
        
        if results['best_cost'] < 45:
            print("🎉 TARGET ACHIEVED! Cost < 45!")
        else:
            print(f"🎯 {results['best_cost'] - 45:.2f} points away from target")

if __name__ == "__main__":
    main()
