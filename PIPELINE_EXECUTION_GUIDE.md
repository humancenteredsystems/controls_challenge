# CLI-Driven Pipeline and Stage 1 Usage Guide

This guide describes the new command-line options for the Stage 1 optimizer and the complete training pipeline script.

---

## Stage 1: Blended 2-PID Optimizer

File: `optimization/blended_2pid_optimizer.py`

View help and available flags:
```bash
python optimization/blended_2pid_optimizer.py --help
```

Key flags:
- `--num_combinations`: Number of parameter sets to evaluate (default: 300)
- `--max_files_per_test`: Max CSV files used per parameter set (default: 25)
- `--num_files`: Number of CSV files to load (default: 50)
- `--model_path`: Override path to ONNX model (default: `models/tinyphysics.onnx`)
- `--data_dir`: Override data directory (default: `data/`)

Example:
```bash
python optimization/blended_2pid_optimizer.py \
  --num_combinations 10 \
  --max_files_per_test 5 \
  --num_files 20 \
  --model_path ./models/tinyphysics.onnx \
  --data_dir ./data
```

---

## Complete Training Pipeline

File: `run_complete_training_pipeline.py`

View help and available flags:
```bash
python run_complete_training_pipeline.py --help
```

### Stage 1 Flags
- `--stage1-combinations`: Grid search combinations (default: 300)
- `--stage1-max-files`: Files per test (default: 25)
- `--stage1-num-files`: Total files to load (default: 50)
- `--model_path`: Path to ONNX model
- `--data_dir`: Path to data directory

### Tournament #1 Flags
- `--t1-rounds`: Generations (default: 12)
- `--t1-pop-size`: Population size (default: 25)
- `--t1-max-files`: Files per test (default: 30)
- `--t1-elite-pct`: Elite preservation (%) (default: 0.3)
- `--t1-revive-pct`: Revival rate (%) (default: 0.2)

### Tournament #2 Flags
- `--t2-rounds`: Generations (default: 12)
- `--t2-pop-size`: Population size (default: 25)
- `--t2-max-files`: Files per test (default: 50)
- `--t2-elite-pct`: Elite preservation (%) (default: 0.3)
- `--t2-revive-pct`: Revival rate (%) (default: 0.2)
- `--perturb-scale`: Mutation scale (default: 0.05)

### Blender Tournament Flags
- `--blender-rounds`: Generations (default: 15)
- `--blender-pop-size`: Population size (default: 20)
- `--blender-max-files`: Files per test (default: 25)

---

## Complete Pipeline Example

```bash
python run_complete_training_pipeline.py \
  --stage1-combinations 100 \
  --stage1-max-files 10 \
  --t1-rounds 5 \
  --t1-pop-size 30 \
  --t2-rounds 8 \
  --t2-max-files 40 \
  --perturb-scale 0.1 \
  --blender-rounds 12 \
  --blender-pop-size 25 \
  --blender-max-files 30
```

This invocation will:
1. Run Stage 1 grid search with 100 combinations, 10 files/test.
2. Run Tournament #1 for 5 generations, population 30.
3. Run Tournament #2 for 8 generations, 40 files/test, perturb scale 0.1.
4. Run Blender tournament for 12 generations, population 25, 30 files/test.

---

End of Guide.
