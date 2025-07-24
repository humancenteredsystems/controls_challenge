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

def run_stage_1_grid_search():
    """Stage 1: Broad PID Parameter Space Exploration"""
    print_stage_header(1, "Broad PID Parameter Space Exploration")
    print("Exploring comprehensive parameter space using grid search...")
    print("This establishes baseline performance and initial parameter ranges.")
    
    cmd = [
        sys.executable, "optimization/blended_2pid_optimizer.py",
        "--comprehensive", "true"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print_results_summary("Grid Search", "blended_2pid_comprehensive_results.json")
        return True
    else:
        print("❌ Stage 1 failed!")
        return False

def run_stage_2_tournament_1():
    """Stage 2: PID Tournament #1 - Initial Discovery"""
    print_stage_header(2, "PID Tournament #1 - Initial Discovery")
    print("Running initial tournament optimization on limited dataset...")
    print("This discovers promising parameter combinations efficiently.")
    
    cmd = [
        sys.executable, "optimization/tournament_optimizer.py",
        "--rounds", "12",
        "--pop_size", "25", 
        "--elite_pct", "0.3",
        "--revive_pct", "0.2",
        "--max_files", "30",
        "--perturb_scale", "0.05"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print_results_summary("Tournament #1", "plans/tournament_progress.json")
        return True
    else:
        print("❌ Stage 2 failed!")
        return False

def run_stage_3_tournament_2():
    """Stage 3: PID Tournament #2 - Champion Validation"""
    print_stage_header(3, "PID Tournament #2 - Champion Validation")
    print("Validating champions from Tournament #1 on expanded dataset...")
    print("This provides robust validation of top performers.")
    
    archive_path = "plans/tournament_archive.json"
    if not Path(archive_path).exists():
        print(f"❌ Archive file {archive_path} not found! Tournament #1 must complete first.")
        return False
    
    cmd = [
        sys.executable, "optimization/tournament_optimizer.py",
        "--rounds", "12",
        "--pop_size", "25",
        "--elite_pct", "0.3", 
        "--revive_pct", "0.2",
        "--max_files", "50",
        "--perturb_scale", "0.05",
        "--seed_from_archive", archive_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print_results_summary("Tournament #2", "plans/tournament_progress.json")
        return True
    else:
        print("❌ Stage 3 failed!")
        return False

def run_stage_4_blender_tournament():
    """Stage 4: PID Blender Tournament - Neural Architecture Search"""
    print_stage_header(4, "PID Blender Tournament - Neural Architecture Search")
    print("Evolving neural network architectures for optimal PID blending...")
    print("This learns intelligent blending strategies to minimize cost.")
    
    archive_path = "plans/tournament_archive.json"
    if not Path(archive_path).exists():
        print(f"❌ Archive file {archive_path} not found! Previous tournaments must complete first.")
        return False
    
    cmd = [
        sys.executable, "optimization/blender_tournament_optimizer.py",
        "--archive", archive_path,
        "--rounds", "15",
        "--pop_size", "20",
        "--max_files", "25"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print_results_summary("Blender Tournament", "plans/blender_tournament_results.json")
        return True
    else:
        print("❌ Stage 4 failed!")
        return False

def print_pipeline_summary():
    """Print final pipeline summary"""
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    
    # Collect results from all stages
    results_files = [
        ("Grid Search", "blended_2pid_comprehensive_results.json"),
        ("Tournament #1", "plans/tournament_progress.json"),
        ("Tournament #2", "plans/tournament_progress.json"),
        ("Blender Tournament", "plans/blender_tournament_results.json")
    ]
    
    print("\n📊 Final Results Summary:")
    print("-" * 50)
    
    best_costs = []
    
    for stage_name, results_file in results_files:
        if Path(results_file).exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                if stage_name == "Grid Search" and isinstance(results, dict):
                    if 'best_cost' in results:
                        cost = results['best_cost']
                        best_costs.append((stage_name, cost))
                        print(f"{stage_name:20}: {cost:.2f}")
                
                elif "Tournament" in stage_name and isinstance(results, dict):
                    if 'best_cost' in results:
                        cost = results['best_cost']
                        best_costs.append((stage_name, cost))
                        print(f"{stage_name:20}: {cost:.2f}")
                
                elif stage_name == "Blender Tournament" and isinstance(results, dict):
                    if 'best_cost' in results:
                        cost = results['best_cost']
                        best_costs.append((stage_name, cost))
                        print(f"{stage_name:20}: {cost:.2f}")
                        
            except Exception as e:
                print(f"{stage_name:20}: Error reading results - {e}")
        else:
            print(f"{stage_name:20}: Results file not found")
    
    if best_costs:
        print("\n🏆 Performance Progression:")
        print("-" * 40)
        for stage, cost in best_costs:
            print(f"  {stage}: {cost:.2f}")
        
        if len(best_costs) > 1:
            initial_cost = best_costs[0][1]
            final_cost = best_costs[-1][1]
            improvement = ((initial_cost - final_cost) / initial_cost) * 100
            print(f"\n🎯 Total Improvement: {improvement:.1f}% reduction in cost")
            print(f"   From {initial_cost:.2f} → {final_cost:.2f}")
    
    print("\n📁 Key Output Files:")
    print("-" * 30)
    print("  • plans/tournament_archive.json - Complete PID parameter database")
    print("  • plans/blender_tournament_results.json - Best neural architecture")
    print("  • models/blender_*.onnx - Trained neural blending models")
    print("  • controllers/neural_blended.py - Production controller")
    
    print("\n🚀 Next Steps:")
    print("-" * 20)
    print("  1. Test the neural blended controller in production")
    print("  2. Monitor performance on new scenarios")  
    print("  3. Run additional blender tournaments if needed")
    print("  4. Deploy best performing model")

def main():
    """Run complete training pipeline"""
    
    start_time = time.time()
    
    print("🚀 Starting Complete Training Pipeline")
    print("Goal: Drive total_cost as low as possible through progressive optimization")
    print(f"Pipeline: Grid Search → Tournament #1 → Tournament #2 → Blender Tournament")
    
    # Stage 1: Grid Search
    if not run_stage_1_grid_search():
        print("Pipeline failed at Stage 1")
        return 1
    
    # Stage 2: Tournament #1
    if not run_stage_2_tournament_1():
        print("Pipeline failed at Stage 2")
        return 1
    
    # Stage 3: Tournament #2  
    if not run_stage_3_tournament_2():
        print("Pipeline failed at Stage 3")
        return 1
    
    # Stage 4: Blender Tournament
    if not run_stage_4_blender_tournament():
        print("Pipeline failed at Stage 4")
        return 1
    
    # Summary
    end_time = time.time()
    duration = (end_time - start_time) / 60  # Convert to minutes
    
    print_pipeline_summary()
    print(f"\n⏱️  Total Pipeline Duration: {duration:.1f} minutes")
    print("\n🎉 Training Pipeline Complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())