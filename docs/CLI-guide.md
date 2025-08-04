# CLI Guide

This document describes all available command-line flags for running the full training pipeline, individual stage scripts, and provides sample commands for common workflows.

---

## 1. Full Pipeline Script

**File:** `run_complete_training_pipeline.py`

Stage 1: Broad PID Parameter Space Exploration
- `--stage1-combinations INT`  
  Number of parameter sets in Stage 1 grid search (default: 300)  
- `--stage1-max-files INT`  
  Max CSV files per parameter set in Stage 1 (default: 25)  
- `--stage1-num-files INT`  
  Number of CSV files loaded from disk in Stage 1 (default: 50)  
- `--model_path PATH`  
  Override path to TinyPhysics ONNX model  
- `--data_dir PATH`  
  Override path to CSV data directory  

Stage 2: PID Tournament #1 (Initial Discovery)
- `--t1-rounds INT`  
- `--t1-pop-size INT`  
- `--t1-max-files INT`  
- `--t1-elite-pct FLOAT`  
- `--t1-revive-pct FLOAT`  
- `--t1-init-seed INT`  
- `--perturb-scale FLOAT`  

Stage 3: PID Tournament #2 (Champion Validation)
- `--t2-rounds INT`  
- `--t2-pop-size INT`  
- `--t2-max-files INT`  
- `--t2-elite-pct FLOAT`  
- `--t2-revive-pct FLOAT`  
- `--t2-init-seed INT`  
- `--perturb-scale FLOAT`  

Stage 4: Data Generation & Pre-Training (Neural Blender)
- `--stage4-samples INT`  
  CSV samples per PID combo for data generation (default: 50)  
- `--stage4-output-data PATH`  
  Output path for generated JSON training data (default: `plans/blender_training_data.json`)  
- `--stage4-epochs INT`  
  Number of pre-training epochs (default: 10)  
- `--stage4-batch-size INT`  
  Batch size for pre-training (default: 32)  
- `--stage4-model-output PATH`  
  Path to write the pre-trained ONNX model (default: `models/neural_blender_pretrained.onnx`)  
- `--stage4-data-seed INT`  
  Seed for Stage 4 randomness (default: use `--data-seed`)  

Stage 5: PID Blender Tournament (Neural Architecture Search)
- `--blender-rounds INT`  
- `--blender-pop-size INT`  
- `--blender-max-files INT`  

Global
- `--data-seed INT`  
  Seed for shuffling and sampling in all stages (omit for non-deterministic behavior)  

---

## 2. Individual Stage Scripts

### 2.1 Stage 1: Grid Search  
**File:** `optimization/blended_2pid_optimizer.py`  
- `--num-combinations INT`  
- `--max-files-per-test INT`  
- `--num-files INT`  
- `--model-path PATH`  
- `--data-dir PATH`  
- `--data-seed INT`  

### 2.2 Stage 2 & 3: Tournament Optimizer  
**File:** `optimization/tournament_optimizer.py`  
- `--rounds INT`  
- `--pop-size INT`  
- `--elite-pct FLOAT`  
- `--revive-pct FLOAT`  
- `--max-files INT`  
- `--perturb-scale FLOAT`  
- `--seed-from-archive PATH`  
- `--init-low-min “p,i,d”`  
- `--init-low-max “p,i,d”`  
- `--init-high-min “p,i,d”`  
- `--init-high-max “p,i,d”`  
- `--init-seed INT`  
- `--data-seed INT`  

### 2.3 Stage 4: Data Generation & Pre-Training  
**File:** `generate_blender_training_data.py`  
- `--archive PATH` (optional)  
- `--samples INT` (via `--stage4-samples` in full pipeline)  
- `--output-path PATH` (via `--stage4-output-data`)  
- `--data-seed INT`  

### 2.4 Stage 5: Blender Tournament (Simplified)  
**File:** `optimization/blender_tournament_optimizer.py`  
- `--archive PATH` (loads fixed PID pairs from Stage 3)  
- `--rounds INT` (default: 10)  
- `--pop_size INT` (default: 15)  
- `--max_files INT` (default: 20)  
- `--data_seed INT` (reproducible neural architecture search)  

**Note:** Stage 5 now uses fixed PID controllers from Stage 3 and only evolves neural architecture to avoid duplicating the PID optimization work already completed.

---

## 3. Sample Command Templates

### 3.1 Quick Validation (small sizes)
```bash
# Stage 1
python3 optimization/blended_2pid_optimizer.py --num-combinations 10 --max-files-per-test 5 --num-files 5 --data-seed 42
# Stage 2
python3 optimization/tournament_optimizer.py --rounds 1 --pop-size 5 --max-files 5 --seed-from-archive blended_2pid_comprehensive_results.json --data-seed 42
# Stage 3
python3 optimization/tournament_optimizer.py --rounds 1 --pop-size 5 --max-files 5 --seed-from-archive plans/tournament_archive.json --data-seed 42
# Stage 4
python3 generate_blender_training_data.py --data-seed 42
# Stage 5
python3 optimization/blender_tournament_optimizer.py --archive plans/tournament_archive.json --rounds 1 --pop-size 5 --max-files 5 --data-seed 42
```

### 3.2 Small Full Pipeline Run
```bash
python3 run_complete_training_pipeline.py \
  --stage1-combinations 50 --stage1-max-files 10 --stage1-num-files 20 \
  --t1-rounds 3 --t1-pop-size 10 --t1-max-files 10 \
  --t2-rounds 3 --t2-pop-size 10 --t2-max-files 20 \
  --stage4-samples 20 --stage4-epochs 5 --stage4-batch-size 16 \
  --stage4-output-data plans/blender_training_data.json \
  --blender-rounds 3 --blender-pop-size 10 --blender-max-files 10 \
  --data-seed 123
```

### 3.3 Medium Pipeline Run
```bash
python3 run_complete_training_pipeline.py \
  --stage1-combinations 150 --stage1-max-files 15 --stage1-num-files 40 \
  --t1-rounds 6 --t1-pop-size 15 --t1-max-files 20 \
  --t2-rounds 6 --t2-pop-size 15 --t2-max-files 40 \
  --stage4-samples 50 --stage4-epochs 10 --stage4-batch-size 32 \
  --stage4-output-data plans/blender_training_data.json \
  --blender-rounds 6 --blender-pop-size 15 --blender-max-files 20 \
  --data-seed 456
```

### 3.4 Full Pipeline Run
```bash
python3 run_complete_training_pipeline.py \
  --stage1-combinations 300 --stage1-max-files 25 --stage1-num-files 50 \
  --t1-rounds 12 --t1-pop-size 25 --t1-max-files 30 \
  --t2-rounds 12 --t2-pop-size 25 --t2-max-files 50 \
  --stage4-samples 100 --stage4-epochs 20 --stage4-batch-size 64 \
  --stage4-output-data plans/blender_training_data.json \
  --blender-rounds 15 --blender-pop-size 20 --blender-max-files 25 \
  --data-seed 789
```

---

## 4. Notes & Tips

- **Reproducibility:** Use `--data-seed` to ensure consistent behavior across all stages.  
- **Stage 4 Defaults:** Data generation writes to `plans/blender_training_data.json` and pre-training model to `models/neural_blender_pretrained.onnx` by default.  
- **Stage 5 Input:** Blender tournament reads the pre-trained model implicitly from the standard path or via `--archive`.  
- **Quick Test:** Override stage flags to small values for fast iterations.  
- **Logging:** Redirect output to a log file: `> pipeline.log 2>&1`.
