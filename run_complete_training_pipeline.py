#!/usr/bin/env python3
"""
Complete Training Pipeline for Neural Blender System

Pipeline stages:
1. Broad PID Parameter Space Exploration (Grid Search)
2. PID Tournament #1 (Initial Discovery)
3. PID Tournament #2 (Champion Validation)
4. Data Generation & Pre-Training (Neural Blender)
5. PID Blender Tournament (Neural Architecture Search)

Goal: Drive total_cost as low as possible through progressive optimization
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
import shutil

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
            cost = results.get('best_cost')
            if cost is not None:
                print(f"✅ {stage_name} Complete - Best Cost: {cost:.2f}")
            else:
                print(f"✅ {stage_name} Complete - Results saved to {results_file}")
        except:
            print(f"✅ {stage_name} Complete - Check {results_file} for results")
    else:
        print(f"⚠️  {stage_name} results file not found: {results_file}")

def cleanup_previous_runs():
    """Remove artifacts from previous runs so pipeline starts fresh."""
    files_to_remove = [
        Path("blended_2pid_comprehensive_results.json"),
        Path("plans/tournament_archive.json"),
        Path("plans/blender_training_data.json"),
        Path("plans/blender_tournament_results.json"),
        Path("models/neural_blender_champion.onnx"),
        Path("controllers/neural_blended_champion.py"),
    ]
    for path in files_to_remove:
        if path.exists():
            try:
                path.unlink()
                print(f"🧹 Removed previous file: {path}")
            except Exception as e:
                print(f"⚠️ Could not remove {path}: {e}")

    # Remove temporary controllers directory
    temp_dir = Path("temp_controllers")
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 Removed temporary controllers directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Could not remove temporary controllers directory: {e}")

    # Remove any temporary controller scripts
    ctrl_dir = Path("controllers")
    for path in ctrl_dir.glob("temp_*.py"):
        try:
            path.unlink()
            print(f"🧹 Removed temporary controller: {path}")
        except Exception as e:
            print(f"⚠️ Could not remove temporary controller {path}: {e}")

def run_stage_1_grid_search(args):
    print_stage_header(1, "Broad PID Parameter Space Exploration")
    cmd = [
        sys.executable, "optimization/blended_2pid_optimizer.py",
        "--num_combinations", str(args.stage1_combinations),
        "--max_files_per_test", str(args.stage1_max_files),
        "--num_files", str(args.stage1_num_files),
        "--model_path", args.model_path or "",
        "--data_dir", args.data_dir or ""
    ]
    if args.data_seed is not None:
        cmd += ["--data-seed", str(args.data_seed)]
    subprocess.run(cmd, check=False)
    print_results_summary("Grid Search", "blended_2pid_comprehensive_results.json")
    return True

def run_stage_2_tournament_1(args):
    print_stage_header(2, "PID Tournament #1 - Initial Discovery")
    seed_archive = "blended_2pid_comprehensive_results.json"
    cmd = [
        sys.executable, "optimization/tournament_optimizer.py",
        "--rounds", str(args.t1_rounds),
        "--pop_size", str(args.t1_pop_size),
        "--elite_pct", str(args.t1_elite_pct),
        "--revive_pct", str(args.t1_revive_pct),
        "--max_files", str(args.t1_max_files),
        "--perturb_scale", str(args.perturb_scale),
        "--seed_from_archive", seed_archive
    ]
    if args.t1_init_seed is not None:
        cmd += ["--init_seed", str(args.t1_init_seed)]
    if args.data_seed is not None:
        cmd += ["--data-seed", str(args.data_seed)]
    subprocess.run(cmd, check=False)
    print_results_summary("Tournament #1", "plans/tournament_archive.json")
    return True

def run_stage_3_tournament_2(args):
    print_stage_header(3, "PID Tournament #2 - Champion Validation")
    archive = "plans/tournament_archive.json"
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
    if args.t2_init_seed is not None:
        cmd += ["--init_seed", str(args.t2_init_seed)]
    if args.data_seed is not None:
        cmd += ["--data-seed", str(args.data_seed)]
    subprocess.run(cmd, check=False)
    print_results_summary("Tournament #2", "plans/tournament_archive.json")
    return True

def run_stage_4_data_generation(args):
    print_stage_header(4, "Data Generation & Pre-Training (Neural Blender)")
    cmd = [
        sys.executable, "generate_blender_training_data.py",
        "--samples", str(args.stage4_samples),
        "--output-path", args.stage4_output_data
    ]
    seed = args.stage4_data_seed if args.stage4_data_seed is not None else args.data_seed
    if seed is not None:
        cmd += ["--data-seed", str(seed)]
    subprocess.run(cmd, check=False)

    # Report data generation results
    print_results_summary("Data Generation", args.stage4_output_data)

    # Train BlenderNet on generated data
    from neural_blender_net import train_blender_net_from_json

    final_loss = train_blender_net_from_json(
        args.stage4_output_data,
        epochs=args.stage4_epochs,
        batch_size=args.stage4_batch_size,
        model_output=args.stage4_model_output,
    )

    # Report training results including final loss and model path
    print_results_summary(
        f"Pre-Training - Loss {final_loss:.4f}", args.stage4_model_output
    )
    return True

def run_stage_5_blender_tournament(args):
    print_stage_header(5, "PID Blender Tournament - Neural Architecture Search")
    archive = "plans/tournament_archive.json"
    cmd = [
        sys.executable, "optimization/blender_tournament_optimizer.py",
        "--archive", archive,
        "--rounds", str(args.blender_rounds),
        "--pop_size", str(args.blender_pop_size),
        "--max_files", str(args.blender_max_files)
    ]
    if args.data_seed is not None:
        cmd += ["--data-seed", str(args.data_seed)]
    subprocess.run(cmd, check=False)
    print_results_summary("Blender Tournament", "plans/blender_tournament_results.json")
    return True

def print_pipeline_summary():
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    stages = [
        ("Grid Search", "blended_2pid_comprehensive_results.json"),
        ("Tournament #1", "plans/tournament_archive.json"),
        ("Tournament #2", "plans/tournament_archive.json"),
        ("Data Generation", "plans/blender_training_data.json"),
        ("Blender Tournament", "plans/blender_tournament_results.json")
    ]
    for name, path in stages:
        if Path(path).exists():
            try:
                data = json.load(open(path))
                cost = data.get('best_cost')
                if cost is not None:
                    print(f"{name:20}: {cost:.2f}")
            except:
                print(f"{name:20}: Error reading results")
        else:
            print(f"{name:20}: Results not found")

def main():
    parser = argparse.ArgumentParser(description="Complete training pipeline flags")
    # Stage 1
    parser.add_argument("--stage1-combinations", type=int, default=300)
    parser.add_argument("--stage1-max-files", type=int, default=25)
    parser.add_argument("--stage1-num-files", type=int, default=50)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    # Stage 2 #1
    parser.add_argument("--t1-rounds", type=int, default=12)
    parser.add_argument("--t1-pop-size", type=int, default=25)
    parser.add_argument("--t1-max-files", type=int, default=30)
    parser.add_argument("--t1-elite-pct", type=float, default=0.3)
    parser.add_argument("--t1-revive-pct", type=float, default=0.2)
    parser.add_argument("--t1-init-seed", type=int, default=None)
    parser.add_argument("--perturb-scale", type=float, default=0.1)
    # Stage 3 #2
    parser.add_argument("--t2-rounds", type=int, default=12)
    parser.add_argument("--t2-pop-size", type=int, default=25)
    parser.add_argument("--t2-max-files", type=int, default=50)
    parser.add_argument("--t2-elite-pct", type=float, default=0.3)
    parser.add_argument("--t2-revive-pct", type=float, default=0.2)
    parser.add_argument("--t2-init-seed", type=int, default=None)
    # Stage 4 data generation
    parser.add_argument("--stage4-output-data", dest="stage4_output_data", type=str,
                        default="plans/blender_training_data.json", help="Output path for generated JSON training data")
    parser.add_argument("--stage4-samples", dest="stage4_samples", type=int, default=50,
                        help="Number of CSV samples per PID combo for data generation")
    parser.add_argument("--stage4-epochs", dest="stage4_epochs", type=int, default=10,
                        help="Number of pre-training epochs for neural blender")
    parser.add_argument("--stage4-batch-size", dest="stage4_batch_size", type=int, default=32,
                        help="Batch size for neural blender pre-training")
    parser.add_argument("--stage4-model-output", dest="stage4_model_output", type=str,
                        default="models/neural_blender_pretrained.onnx", help="Output path for pre-trained neural blender ONNX model")
    parser.add_argument("--stage4-data-seed", dest="stage4_data_seed", type=int, default=None,
                        help="Seed for Stage 4 data generation and pre-training")
    # Stage 5 blender tournament
    parser.add_argument("--blender-rounds", type=int, default=15)
    parser.add_argument("--blender-pop-size", type=int, default=20)
    parser.add_argument("--blender-max-files", type=int, default=25)
    # Global
    parser.add_argument("--data-seed", type=int, default=None)
    args = parser.parse_args()

    print("🚀 Starting Complete Training Pipeline")
    cleanup_previous_runs()
    run_stage_1_grid_search(args)
    run_stage_2_tournament_1(args)
    run_stage_3_tournament_2(args)
    run_stage_4_data_generation(args)
    run_stage_5_blender_tournament(args)
    print_pipeline_summary()

if __name__ == "__main__":
    sys.exit(main())
