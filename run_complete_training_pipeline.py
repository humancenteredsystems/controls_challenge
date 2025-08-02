#!/usr/bin/env python3
"""
Complete Training Pipeline for Neural Blender System

Pipeline stages:
1. Broad PID Parameter Space Exploration (Grid Search)
2. PID Tournament #1 (Initial Discovery)
3. PID Tournament #2 (Champion Validation)
4. PID Blender Tournament (Neural Architecture Search)

Goal: Drive total_cost as low as possible through progressive optimization
"""

import subprocess
import sys
import json
import time
import argparse
from pathlib import Path

def print_stage_header(stage_num, stage_name):
    """Print formatted stage header"""
    print("\n" + "=" * 80)
    print(f"STAGE {stage_num}: {stage_name}")
    print("=" * 80)

def print_results_summary(stage_name, results_file):
    """Print summary of stage results"""
    if Path(results_file).exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            if isinstance(results, dict) and 'best_cost' in results:
                print(f"✅ {stage_name} Complete - Best Cost: {results['best_cost']:.2f}")
            else:
                print(f"✅ {stage_name} Complete - Results saved to {results_file}")
        except:
            print(f"✅ {stage_name} Complete - Check {results_file} for results")
    else:
        print(f"⚠️  {stage_name} results file not found: {results_file}")

def run_stage_1_grid_search(args):
    """Stage 1: Broad PID Parameter Space Exploration"""
    print_stage_header(1, "Broad PID Parameter Space Exploration")
    cmd = [
        sys.executable, "optimization/blended_2pid_optimizer.py",
        "--num_combinations", str(args.stage1_combinations),
        "--max_files_per_test", str(args.stage1_max_files),
        "--num_files", str(args.stage1_num_files),
        "--model_path", args.model_path or "",
        "--data_dir", args.data_dir or ""
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print_results_summary("Grid Search", "blended_2pid_comprehensive_results.json")
        return True
    print("❌ Stage 1 failed!")
    return False

def run_stage_2_tournament_1(args):
    """Stage 2: PID Tournament #1 - Initial Discovery"""
    print_stage_header(2, "PID Tournament #1 - Initial Discovery")
    cmd = [
        sys.executable, "optimization/tournament_optimizer.py",
        "--rounds", str(args.t1_rounds),
        "--pop_size", str(args.t1_pop_size),
        "--elite_pct", str(args.t1_elite_pct),
        "--revive_pct", str(args.t1_revive_pct),
        "--max_files", str(args.t1_max_files),
        "--perturb_scale", str(args.perturb_scale)
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print_results_summary("Tournament #1", "plans/tournament_progress.json")
        return True
    print("❌ Stage 2 failed!")
    return False

def run_stage_3_tournament_2(args):
    """Stage 3: PID Tournament #2 - Champion Validation"""
    print_stage_header(3, "PID Tournament #2 - Champion Validation")
    archive = "plans/tournament_archive.json"
    if not Path(archive).exists():
        print(f"❌ Archive file {archive} not found! Tournament #1 must complete first.")
        return False
    cmd = [
        sys.executable, "optimization/tournament_optimizer.py",
        "--rounds", str(args.t2_rounds),
        "--pop_size", str(args.t2_pop_size),
        "--elite_pct", str(args.t2_elite_pct),
        "--revive_pct", str(args.t2_revive_pct),
        "--max_files", str(args.t2_max_files),
        "--perturb_scale", str(args.perturb_scale),
        "--seed_from_archive", archive
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print_results_summary("Tournament #2", "plans/tournament_progress.json")
        return True
    print("❌ Stage 3 failed!")
    return False

def run_stage_4_blender_tournament(args):
    """Stage 4: PID Blender Tournament - Neural Architecture Search"""
    print_stage_header(4, "PID Blender Tournament - Neural Architecture Search")
    archive = "plans/tournament_archive.json"
    if not Path(archive).exists():
        print(f"❌ Archive file {archive} not found! Previous tournaments must complete first.")
        return False
    cmd = [
        sys.executable, "optimization/blender_tournament_optimizer.py",
        "--archive", archive,
        "--rounds", str(args.blender_rounds),
        "--pop_size", str(args.blender_pop_size),
        "--max_files", str(args.blender_max_files)
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print_results_summary("Blender Tournament", "plans/blender_tournament_results.json")
        return True
    print("❌ Stage 4 failed!")
    return False

def print_pipeline_summary():
    """Print final pipeline summary"""
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    results_files = [
        ("Grid Search", "blended_2pid_comprehensive_results.json"),
        ("Tournament #1", "plans/tournament_progress.json"),
        ("Tournament #2", "plans/tournament_progress.json"),
        ("Blender Tournament", "plans/blender_tournament_results.json")
    ]
    for name, path in results_files:
        if Path(path).exists():
            try:
                data = json.load(open(path))
                cost = data.get('best_cost', None)
                if cost is not None:
                    print(f"{name:20}: {cost:.2f}")
            except:
                print(f"{name:20}: Error reading results")
        else:
            print(f"{name:20}: Results not found")
    print("\n🚀 Next Steps:")
    print("  1. Test the neural blended controller in production")
    print("  2. Monitor performance on new scenarios")
    print("  3. Run additional blender tournaments if needed")
    print("  4. Deploy best performing model")

def main():
    """Run complete training pipeline"""
    parser = argparse.ArgumentParser(description="Complete training pipeline flags")
    # Stage 1 flags
    parser.add_argument("--stage1-combinations", dest="stage1_combinations", type=int, default=300)
    parser.add_argument("--stage1-max-files", dest="stage1_max_files", type=int, default=25)
    parser.add_argument("--stage1-num-files", dest="stage1_num_files", type=int, default=50)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    # Tournament 1 flags
    parser.add_argument("--t1-rounds", dest="t1_rounds", type=int, default=12)
    parser.add_argument("--t1-pop-size", dest="t1_pop_size", type=int, default=25)
    parser.add_argument("--t1-max-files", dest="t1_max_files", type=int, default=30)
    parser.add_argument("--t1-elite-pct", dest="t1_elite_pct", type=float, default=0.3)
    parser.add_argument("--t1-revive-pct", dest="t1_revive_pct", type=float, default=0.2)
    # Tournament 2 flags
    parser.add_argument("--t2-rounds", dest="t2_rounds", type=int, default=12)
    parser.add_argument("--t2-pop-size", dest="t2_pop_size", type=int, default=25)
    parser.add_argument("--t2-max-files", dest="t2_max_files", type=int, default=50)
    parser.add_argument("--t2-elite-pct", dest="t2_elite_pct", type=float, default=0.3)
    parser.add_argument("--t2-revive-pct", dest="t2_revive_pct", type=float, default=0.2)
    parser.add_argument("--perturb-scale", dest="perturb_scale", type=float, default=0.05)
    # Blender tournament flags
    parser.add_argument("--blender-rounds", dest="blender_rounds", type=int, default=15)
    parser.add_argument("--blender-pop-size", dest="blender_pop_size", type=int, default=20)
    parser.add_argument("--blender-max-files", dest="blender_max_files", type=int, default=25)
    args = parser.parse_args()

    print("🚀 Starting Complete Training Pipeline")
    if not run_stage_1_grid_search(args):
        return 1
    if not run_stage_2_tournament_1(args):
        return 1
    if not run_stage_3_tournament_2(args):
        return 1
    if not run_stage_4_blender_tournament(args):
        return 1

    print_pipeline_summary()
    return 0

if __name__ == "__main__":
    sys.exit(main())
