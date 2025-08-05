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

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

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
        """Define a comprehensive parameter search space using simple uniform sampling."""
        
        print(f"Generating comprehensive search space for {num_combinations} combinations using pure uniform random sampling...")
        
        combinations = []
        
        # Pure uniform random sampling (no core combinations)
        np.random.seed(42)  # For reproducible results
        
        bounds = {
            'low_p': (0.25, 0.6),
            'low_i': (0.01, 0.12),
            'low_d': (-0.25, -0.05),
            'high_p': (0.15, 0.4),
            'high_i': (0.005, 0.08),
            'high_d': (-0.15, -0.03)
        }
        
        for _ in range(num_combinations):
            low_gains = [
                round(np.random.uniform(*bounds['low_p']), 3),
                round(np.random.uniform(*bounds['low_i']), 3),
                round(np.random.uniform(*bounds['low_d']), 3)
            ]
            high_gains = [
                round(np.random.uniform(*bounds['high_p']), 3),
                round(np.random.uniform(*bounds['high_i']), 3),
                round(np.random.uniform(*bounds['high_d']), 3)
            ]
            combinations.append((low_gains, high_gains))
        
        print(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def test_controller_combination(self, data_files: List[str], low_gains: List[float], 
                                   high_gains: List[float], max_files: int = 25) -> Dict[str, float]:
        """Test a single blended 2-PID controller combination"""
        from optimization import generate_blended_controller
        controller_content = generate_blended_controller(low_gains, high_gains)
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
            except Exception as e:
                print(f"Error during rollout for {name}: {e}")
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
        results = []
        progress_file = BASE_DIR / "blended_2pid_optimization_progress.json"
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
                        self.best_cost = entry["avg_total_cost"]
                        self.best_params = (low, high)
                        print(f"🎉 New best: {self.best_cost:.2f}")
                        params_path = ROOT_DIR / "blended_2pid_params.json"
                        with open(params_path, "w") as pf:
                            json.dump(
                                {
                                    "low_gains": low,
                                    "high_gains": high,
                                    "best_cost": self.best_cost,
                                },
                                pf,
                                indent=2,
                            )
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

    if optimizer.best_params is None:
        raise RuntimeError("Optimization completed without finding valid parameters")

    params_path = base_dir / "blended_2pid_params.json"
    with open(params_path, "w") as f:
        json.dump({
            "low_gains": optimizer.best_params[0],
            "high_gains": optimizer.best_params[1],
            "best_cost": optimizer.best_cost,
        }, f, indent=2)
    print(f"Saved best parameters to {params_path}")

if __name__ == "__main__":
    main()
