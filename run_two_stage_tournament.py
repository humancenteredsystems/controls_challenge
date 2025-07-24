#!/usr/bin/env python3
"""
Two-stage tournament optimization example:
1. Tournament #1: Initial optimization on subset of data
2. Tournament #2: Champions from #1 compete on full dataset
"""

import subprocess
import sys
from pathlib import Path

def run_tournament_1():
    """Run Tournament #1 - initial optimization"""
    print("🏆 Starting Tournament #1: Initial Optimization")
    print("=" * 60)
    
    cmd = [
        sys.executable, "optimization/tournament_optimizer.py",
        "--rounds", "10",
        "--pop_size", "20", 
        "--elite_pct", "0.3",
        "--revive_pct", "0.2",
        "--max_files", "30",
        "--perturb_scale", "0.05"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Tournament #1 failed!")
        return False
    
    print("\n✅ Tournament #1 Complete!")
    print("Archive saved to plans/tournament_archive.json")
    return True

def run_tournament_2():
    """Run Tournament #2 - champions compete on full dataset"""
    print("\n🏆 Starting Tournament #2: Champion Validation")
    print("=" * 60)
    
    archive_path = "plans/tournament_archive.json"
    if not Path(archive_path).exists():
        print(f"Archive file {archive_path} not found! Run Tournament #1 first.")
        return False
    
    cmd = [
        sys.executable, "optimization/tournament_optimizer.py",
        "--rounds", "10",
        "--pop_size", "20",
        "--elite_pct", "0.3", 
        "--revive_pct", "0.2",
        "--max_files", "50",  # More files for robust validation
        "--perturb_scale", "0.05",
        "--seed_from_archive", archive_path
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Tournament #2 failed!")
        return False
        
    print("\n✅ Tournament #2 Complete!")
    print("Final results with expanded validation complete!")
    return True

def main():
    """Run two-stage tournament optimization"""
    print("Starting Two-Stage Tournament Optimization")
    print("Stage 1: Initial optimization on data subset")
    print("Stage 2: Champions compete on full dataset")
    print()
    
    # Stage 1: Tournament #1
    if not run_tournament_1():
        return 1
    
    # Stage 2: Tournament #2 
    if not run_tournament_2():
        return 1
    
    print("\n🎉 Two-Stage Tournament Complete!")
    print("Best performers from Tournament #1 have been validated on expanded dataset")
    print("Check plans/tournament_archive.json for final results")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())