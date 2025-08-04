"""
Comprehensive optimization engine for blended 2-PID controllers
"""
import argparse
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

from tinyphysics_custom import run_rollout

class Blended2PIDOptimizer:
    """Enhanced grid search optimization for blended 2-PID controllers"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_instance = None            # Add model caching for GPU optimization
        self.results = []
        self.best_cost = float('inf')
        self.best_params = None
    
    def _get_model_instance(self):
        """Lazy model creation for GPU optimization - create once, reuse many times"""
        if self.model_instance is None:
            from tinyphysics_custom import TinyPhysicsModel
            self.model_instance = TinyPhysicsModel(self.model_path, debug=False)
            providers = self.model_instance.ort_session.get_providers()
            gpu_status = 'GPU ENABLED' if 'CUDAExecutionProvider' in providers else 'CPU FALLBACK'
            print(f"Blended 2-PID optimizer: {gpu_status}")
        return self.model_instance
        
    def define_comprehensive_search_space(self, num_combinations: int = 250) -> List[Tuple[List[float], List[float]]]:
        """Define a comprehensive parameter search space for 2-PID controllers"""
        
        print(f"Generating comprehensive search space for {num_combinations} combinations...")
        
        # More granular parameter ranges for P, I, D (6 parameters total)
        low_p_values = np.linspace(0.25, 0.6, 8)    # 8 values from 0.25 to 0.6
        low_i_values = np.linspace(0.01, 0.12, 6)   # 6 values from 0.01 to 0.12
        low_d_values = np.linspace(-0.25, -0.05, 6) # 6 values from -0.25 to -0.05
        
        high_p_values = np.linspace(0.15, 0.4, 8)   # 8 values from 0.15 to 0.4
        high_i_values = np.linspace(0.005, 0.08, 6) # 6 values from 0.005 to 0.08
        high_d_values = np.linspace(-0.15, -0.03, 6)# 6 values from -0.15 to -0.03
        
        # Generate systematic combinations
        combinations = []
        
        # Strategy 1: Full factorial of reduced spaces (core search)
        low_core = [(0.3, 0.03, -0.1), (0.35, 0.05, -0.15), (0.4, 0.04, -0.12)]
        high_core = [(0.2, 0.01, -0.05), (0.25, 0.02, -0.08), (0.3, 0.015, -0.06)]
        
        for low in low_core:
            for high in high_core:
                combinations.append((list(low), list(high)))
        
        # Strategy 2: Random sampling from refined ranges around best known values
        np.random.seed(42)  # For reproducible results
        
        best_low = [0.3, 0.03, -0.1]
        best_high = [0.2, 0.01, -0.05]
        
        for _ in range(num_combinations - len(combinations)):
            if len(combinations) >= num_combinations:
                break
            low_p = np.clip(np.random.normal(best_low[0], 0.08), 0.2, 0.6)
            low_i = np.clip(np.random.normal(best_low[1], 0.02), 0.01, 0.12)
            low_d = np.clip(np.random.normal(best_low[2], 0.04), -0.25, -0.05)
            high_p = np.clip(np.random.normal(best_high[0], 0.06), 0.15, 0.4)
            high_i = np.clip(np.random.normal(best_high[1], 0.015), 0.005, 0.08)
            high_d = np.clip(np.random.normal(best_high[2], 0.03), -0.15, -0.03)
            combinations.append(([round(low_p,3), round(low_i,3), round(low_d,3)],
                                 [round(high_p,3), round(high_i,3), round(high_d,3)]))
        
        # Strategy 3: Extreme boundary combinations
        extreme = [
            ([0.6, 0.01, -0.05], [0.15, 0.005, -0.03]),
            ([0.25, 0.12, -0.25], [0.4, 0.08, -0.15]),
            ([0.45, 0.06, -0.18], [0.25, 0.03, -0.09]),
        ]
        for combo in extreme:
            if len(combinations) < num_combinations:
                combinations.append(combo)
        
        print(f"Generated {len(combinations)} parameter combinations")
        return combinations[:num_combinations]
    
    def test_controller_combination(self, data_files: List[str], low_gains: List[float], 
                                   high_gains: List[float], max_files: int = 25) -> Dict[str, float]:
        """Test a single blended 2-PID controller combination"""
        controller_content = f'''from . import BaseController
from controllers.shared_pid import SpecializedPID
import math

class Controller(BaseController):
    def __init__(self):
        self.low_speed_pid = SpecializedPID({low_gains[0]}, {low_gains[1]}, {low_gains[2]})
        self.high_speed_pid = SpecializedPID({high_gains[0]}, {high_gains[1]}, {high_gains[2]})
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        v_ego = state.v_ego
        low = self.low_speed_pid.update(error)
        high = self.high_speed_pid.update(error)
        def smooth_blend_weight(v_ego, threshold=15.0, smoothness=1.5):
            return 1.0 / (1.0 + math.exp(-(v_ego - threshold) / smoothness))
        blend_weight = smooth_blend_weight(v_ego)
        return (1.0 - blend_weight) * low + blend_weight * high
'''
        base_dir = Path(__file__).parent.parent
        import uuid
        uid = str(uuid.uuid4())[:8]
        name = f"temp_test_{uid}"
        path = base_dir/"controllers"/f"{name}.py"
        with open(path, "w") as f:
            f.write(controller_content)
        costs, count = [], 0
        for fpath in data_files[:max_files]:
            try:
                model = self._get_model_instance()
                cost, _, _ = run_rollout(fpath, name, model, debug=False)
                costs.append(cost["total_cost"])
                count += 1
            except:
                pass
        if path.exists(): path.unlink()
        if count==0:
            return {"avg_total_cost": float("inf"), "num_files":0}
        arr = np.array(costs)
        return {"avg_total_cost": float(arr.mean()), "num_files":count,
                "std_cost": float(arr.std()), "min_cost": float(arr.min()),
                "max_cost": float(arr.max())}
    
    def optimize_comprehensive(self, data_files: List[str], num_combinations: int = 250, 
                               max_files_per_test: int = 25) -> Dict[str, Any]:
        """Run comprehensive optimization for blended 2-PID controllers"""
        print("Starting comprehensive blended 2-PID optimization:")
        print(f" - {num_combinations} combinations, {max_files_per_test} files/test, {len(data_files)} total")
        combos = self.define_comprehensive_search_space(num_combinations)
        results, progress_file = [], "blended_2pid_optimization_progress.json"
        for i, (low, high) in enumerate(tqdm(combos, desc="Testing combos")):
            try:
                start = time.time()
                res = self.test_controller_combination(data_files, low, high, max_files_per_test)
                duration = time.time()-start
                if res["avg_total_cost"]<float("inf"):
                    entry = {
                        "id":i, "low_gains":low, "high_gains":high,
                        "avg_total_cost":res["avg_total_cost"],
                        "num_files":res["num_files"],
                        "std_cost":res["std_cost"], "min_cost":res["min_cost"],
                        "max_cost":res["max_cost"], "test_time":duration
                    }
                    results.append(entry)
                    if entry["avg_total_cost"]<self.best_cost:
                        self.best_cost=entry["avg_total_cost"]
                        self.best_params=(low,high)
                        print(f"🎉 New best: {self.best_cost:.2f}")
                        with open("blended_2pid_best_params_temp.json","w") as pf:
                            json.dump({"low":low,"high":high,"cost":self.best_cost,"id":i},pf,indent=2)
                if (i+1)%25==0:
                    with open(progress_file,"w") as pf:
                        json.dump({"completed":i+1,"best":self.best_cost,"count":len(results)},pf,indent=2)
            except Exception as e:
                print(f"Error combo {i}: {e}")
        results.sort(key=lambda x: x["avg_total_cost"])
        return {"best_cost": self.best_cost,
                "best_params": {
                    "low": self.best_params[0] if self.best_params is not None else None,
                    "high": self.best_params[1] if self.best_params is not None else None
                },
                "all_results": results,
                "tested": len(results),
                "attempted": len(combos),
                "success_rate": len(results) / len(combos) if combos else 0}
    
    def save_comprehensive_results(self, results: Dict[str, Any], 
                                   filename: str = "blended_2pid_comprehensive_results.json"):
        """Save comprehensive optimization results"""
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def print_top_results(self, results: Dict[str, Any], top_n: int = 10):
        """Print top N results with enhanced statistics"""
        print(f"\n🏆 Top {top_n} combinations:")
        for r in results["all_results"][:top_n]:
            print(f"Cost {r['avg_total_cost']:.2f}, files {r['num_files']}, low {r['low_gains']}, high {r['high_gains']}")
    
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive optimization for blended 2-PID controllers"
    )
    parser.add_argument("--num_combinations", type=int, default=300,
                        help="Number of parameter sets to evaluate")
    parser.add_argument("--max_files_per_test", type=int, default=25,
                        help="Max CSV files used for each parameter set")
    parser.add_argument("--num_files", type=int, default=50,
                        help="Number of CSV files loaded from the data directory")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override path to tinyphysics model")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override path to directory containing *.csv data")
    parser.add_argument("--data-seed", "--data_seed", dest="data_seed", type=int, default=None,
                        help="Seed for shuffling data files (omit for non-deterministic)")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    model_path = args.model_path if args.model_path else str(base_dir/"models"/"tinyphysics.onnx")
    data_dir = Path(args.data_dir) if args.data_dir else base_dir/"data"
    data_files = [str(f) for f in data_dir.glob("*.csv")]
    data_files.sort()
    rng = np.random.default_rng(args.data_seed)
    rng.shuffle(data_files)
    data_files = data_files[:args.num_files]
    
    print(f"Found {len(data_files)} data files")
    optimizer = Blended2PIDOptimizer(model_path)
    results = optimizer.optimize_comprehensive(
        data_files,
        num_combinations=args.num_combinations,
        max_files_per_test=args.max_files_per_test
    )
    optimizer.print_top_results(results, top_n=15)
    optimizer.save_comprehensive_results(results)

if __name__ == "__main__":
    main()
