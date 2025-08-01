# Time Step Corrected Optimization Pipeline Execution Guide

## 🚀 Ready to Execute - All Fixes Applied

### ✅ Infrastructure Fixes Complete:
- **Controller template fixed** in [`optimization/__init__.py`](optimization/__init__.py) - Added `dt = 0.1` scaling
- **Tournament controller fixed** in [`controllers/tournament_optimized.py`](controllers/tournament_optimized.py) - Changed to `dt = 0.1`

---

## 📋 STEP 1: Validate Fixes (5 minutes)

**Run the validation script first:**
```bash
python validate_timestep_fix.py
```

**Expected output:**
```
🚀 Time Step Fix Validation
==================================================
🔧 Testing controller template fix...
✅ Controller template fix VERIFIED - dt = 0.1 scaling found

🔧 Testing tournament controller fix...
✅ Tournament controller fix VERIFIED - loads without errors

🔧 Testing single evaluation with corrected time steps...
✅ Single evaluation SUCCESSFUL
   Test file: 00000.csv
   Total cost: XX.XX

==================================================
📊 VALIDATION RESULTS: 3/3 tests passed
🎉 ALL TESTS PASSED - Ready to proceed with optimization pipeline!
```

---

## 🏗️ STEP 2: Execute Re-optimization Pipeline

### **Phase 1: Broad PID Parameter Exploration (2-4 hours)**

```bash
# Conservative approach (recommended)
python optimization/blended_2pid_optimizer.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --num_combinations 250 \
    --max_files 15

# Aggressive approach (faster)
python optimization/blended_2pid_optimizer.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --num_combinations 150 \
    --max_files 10
```

**Expected output location:** `blended_2pid_best_params_temp.json`

---

### **Phase 2: PID Tournament #1 - Initial Discovery (4-8 hours)**

```bash
# Conservative approach
python optimization/tournament_optimizer.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --rounds 10 \
    --pop_size 20 \
    --elite_pct 20 \
    --revive_pct 10 \
    --max_files 30 \
    --perturb_scale 0.1

# Aggressive approach  
python optimization/tournament_optimizer.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --rounds 8 \
    --pop_size 15 \
    --elite_pct 25 \
    --revive_pct 15 \
    --max_files 20 \
    --perturb_scale 0.15
```

**Expected output location:** `plans/tournament_archive.json`

---

### **Phase 3: PID Tournament #2 - Champion Validation (6-12 hours)**

```bash
# Conservative approach  
python optimization/tournament_optimizer.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --rounds 10 \
    --pop_size 20 \
    --elite_pct 20 \
    --revive_pct 10 \
    --max_files 50 \
    --perturb_scale 0.05 \
    --seed_from_archive ./plans/tournament_archive.json

# Aggressive approach
python optimization/tournament_optimizer.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --rounds 8 \
    --pop_size 15 \
    --elite_pct 25 \
    --revive_pct 15 \
    --max_files 30 \
    --perturb_scale 0.08 \
    --seed_from_archive ./plans/tournament_archive.json
```

**Expected output:** Updated `plans/tournament_archive.json` with enhanced validation

---

### **Phase 4: PID Blender Tournament - Final Optimization (8-16 hours)**

```bash
# Conservative approach
python optimization/blender_tournament_optimizer.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --rounds 8 \
    --pop_size 15 \
    --max_files 15 \
    --tournament_archive ./plans/tournament_archive.json

# Aggressive approach  
python optimization/blender_tournament_optimizer.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --rounds 6 \
    --pop_size 12 \
    --max_files 10 \
    --tournament_archive ./plans/tournament_archive.json
```

**Expected output location:** `plans/blender_tournament_results.json`

---

## 🧪 STEP 3: Generate Final Controller

After blender tournament completes:

```bash
# Extract the best parameters and create new optimized controller
python -c "
import json
from pathlib import Path

# Load blender tournament results
with open('plans/blender_tournament_results.json', 'r') as f:
    results = json.load(f)

# Find best strategy
best_strategy = min(results['strategies'], key=lambda x: x['avg_cost'])

print(f'Best strategy cost: {best_strategy[\"avg_cost\"]:.2f}')
print(f'Best strategy config: {best_strategy}')

# TODO: Generate new optimized_blender_corrected.py with these parameters
"
```

---

## 🔍 STEP 4: Pre-Submission Validation with eval_custom.py

**Always validate with eval_custom.py before running official eval.py to catch compatibility issues:**

```bash
# Quick validation (recommended first step)
python eval_custom.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --controller neural_blended_champion \
    --mode quick \
    --validate-first

# Standard validation (full compatibility check)
python eval_custom.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --controller neural_blended_champion \
    --mode standard \
    --output pre_submission_results.json

# Minimal validation (compatibility only)
python eval_custom.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --controller neural_blended_champion \
    --mode validate-only
```

**Expected output for successful validation:**
```
🎯 eval_custom.py - Controller Pre-submission Validation
   Controller: neural_blended_champion
   Mode: quick
   
🔍 Validating controller compatibility: neural_blended_champion
✅ Controller 'neural_blended_champion' is loadable
🧪 Testing basic rollout with file: 00000.csv
✅ Basic rollout successful
   Sample cost: 45.23
   Rollout time: 0.125s
✅ Time step consistency verified (array lengths match)
✅ Controller compatibility validated successfully

🚀 Running quick evaluation: Fast validation with 10 segments for development testing
📈 Evaluation Results (quick mode):
   Test Controller (neural_blended_champion): 42.15
   Baseline Controller (pid): 58.33
   ✅ Improvement: 16.18 (27.7%)
   
🎉 SUCCESS: Controller ready for eval.py submission!
```

---

## 📊 STEP 5: Official Final Validation

**Only run eval.py after successful eval_custom.py validation:**

```bash
# Single file test (quick check)
python eval.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --num_segs 1 \
    --test_controller neural_blended_champion \
    --baseline_controller pid

# Full official evaluation (for leaderboard submission)
python eval.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --num_segs 100 \
    --test_controller neural_blended_champion \
    --baseline_controller pid
```

---

## ⏱️ Timeline Estimates

### **Conservative Approach (Recommended):**
- **Phase 1:** 2-4 hours (250 combinations, 15 files)
- **Phase 2:** 4-8 hours (10 rounds, 20 pop, 30 files)  
- **Phase 3:** 6-12 hours (10 rounds, 20 pop, 50 files)
- **Phase 4:** 8-16 hours (8 rounds, 15 pop, 15 files)
- **Total:** 20-40 hours

### **Aggressive Approach (Faster):**
- **Phase 1:** 1-2 hours (150 combinations, 10 files)
- **Phase 2:** 2-4 hours (8 rounds, 15 pop, 20 files)
- **Phase 3:** 3-6 hours (8 rounds, 15 pop, 30 files)  
- **Phase 4:** 4-8 hours (6 rounds, 12 pop, 10 files)
- **Total:** 10-20 hours

---

## 🔄 Progress Monitoring

Monitor these files for progress:
- `blended_2pid_optimization_progress.json` (Phase 1)
- `plans/tournament_archive.json` (Phases 2-3)
- `plans/blender_tournament_results.json` (Phase 4)

Each phase saves intermediate results, so you can stop/resume safely.

---

## ⚠️ Important Notes

1. **GPU Memory:** Ensure ~4-8GB GPU memory available
2. **Data Path:** Adjust `--data_path` if your data is elsewhere
3. **Model Path:** Adjust `--model_path` if model location differs
4. **Backup:** Current results are preserved - new files will have different names
5. **Resume:** Each phase can be resumed from saved state if interrupted

---

## 🎯 Expected Final Result

With corrected time steps, your new controller should:
- ✅ Achieve consistent performance between optimization and evaluation
- ✅ Beat baseline PID controller in official eval.py
- ✅ Have parameters properly tuned for actual 10 Hz physics simulation
- ✅ Eliminate the eval.py performance disconnect entirely

**Ready to execute!** Start with the validation script, then proceed through the phases.