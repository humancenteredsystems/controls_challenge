#!/usr/bin/env python3
"""
Standalone validation script for the tournament champion controller.

Samples 100 random data files, benchmarks optimized_ensemble against baseline PID,
and generates an HTML report via eval.py.
"""
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
import json
import sys

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    model_path = base_dir / "models" / "tinyphysics.onnx"
    optimal_file = base_dir / "optimal_params.json"

    # Ensure necessary files exist
    if not data_dir.is_dir():
        print(f"Error: data directory not found at {data_dir}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_file():
        print(f"Error: ONNX model not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    if not optimal_file.is_file():
        print(f"Error: optimal_params.json not found at {optimal_file}", file=sys.stderr)
        sys.exit(1)

    # Champion controller uses optimized_ensemble module
    test_controller = "optimized_ensemble"
    baseline_controller = "pid"

    # Sample 100 random CSV files
    all_files = list(data_dir.glob("*.csv"))
    sample_size = min(100, len(all_files))
    sampled = random.sample(all_files, sample_size)
    print(f"Selected {sample_size} random files for hold-out validation.")

    # Copy sampled files into temporary directory
    with tempfile.TemporaryDirectory(prefix="validation_data_") as tmpdir:
        tmp_path = Path(tmpdir)
        for f in sampled:
            shutil.copy(f, tmp_path / f.name)

        # Build eval.py command
        eval_script = base_dir / "eval.py"
        cmd = [
            sys.executable, str(eval_script),
            "--model_path", str(model_path),
            "--data_path", str(tmp_path),
            "--num_segs", str(sample_size),
            "--test_controller", test_controller,
            "--baseline_controller", baseline_controller
        ]
        print("Running validation command:")
        print("  " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Validation failed: {e}", file=sys.stderr)
            sys.exit(e.returncode)

if __name__ == "__main__":
    main()
