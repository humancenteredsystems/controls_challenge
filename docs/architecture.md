# Comma.ai Controls Challenge Optimization System Architecture

**Document Version:** 1.1  
**Date:** August 2025  
**Status:** Updated Architecture

## 1. System Overview

Optimization Pipeline Stages  
1. Stage 1: Broad PID Parameter Search (Grid Search)  
2. Stage 2: Tournament A (Initial PID Tournament, varied gains)  
3. Stage 3: Tournament B (Champion Validation, expanded dataset)  
4. Stage 4: Neural Blender Training (Data generation + pre-training)  
5. Stage 5: Neural Blender Tournament (Architecture search)  
6. Stage 6: Lightweight Validation (eval_custom.py on Stage 5 winner)  
7. Stage 7: Official Evaluation (eval.py on Stage 5 winner)

## 2. Pipeline Flow

```
[Stage 1] → [Stage 2] → [Stage 3]
                           ↓
                       [Stage 5] ← [Stage 4]
                           ↓
                [Stage 6: eval_custom.py]
                           ↓
                  [Stage 7: eval.py]
```

## 3. Core Components

### 3.1 TinyPhysicsModel

- CPU-only version: `tinyphysics.py` (competition interface)  
- GPU-optimized version: `tinyphysics_custom.py` (CUDA + CPU fallback)  
- Unified rollout interface supports both model path and instance reuse.

### 3.2 Pipeline Orchestration

- Entry point: `run_complete_training_pipeline.py`  
- Configurable via CLI flags for each stage (see CLI guide).

### 3.3 Optimization Algorithms

- **Stage 1:** Grid search  
  - `optimization/blended_2pid_optimizer.py`

- **Stages 2 & 3:** Evolutionary Tournament  
  - `optimization/tournament_optimizer.py`  
  - Maintains archive: `plans/tournament_archive.json`

- **Stage 4:** Data generation & pre-training  
  - `generate_blender_training_data.py`  
  - `neural_blender_net.py`: BlenderNet model & training scripts

- **Stage 5:** Neural architecture search (simplified)  
  - `optimization/blender_tournament_optimizer.py`  
  - Uses fixed PID controllers from Stage 3, evolves only neural architecture  
  - Avoids duplication of PID optimization already completed in Stages 2/3

## 4. Controller Types

- **PID**: `controllers/pid.py`  
- **Blended 2-PID**: `controllers/blended_2pid.py`  
- **TournamentOptimized**: `controllers/tournament_optimized.py`  
- **NeuralBlended**: `controllers/neural_blended.py`  
- **ChampionNeuralBlended**: `controllers/neural_blended_champion.py`  
- **Zero**: `controllers/zero.py`

## 5. System Integration Flows

- **Stages 1–3:**  
  Model → Parameter generation → Tournament evolution → Archive update.

- **Stages 4–5:**  
  Archive → Generate training data → Pre-train BlenderNet → Architecture tournament → Champion ONNX.

- **Stages 6–7:**  
  eval_custom.py → quick validation → eval.py → official evaluation.

## 6. Technical Specifications

- **Language:** Python 3.8+  
- **ONNX Runtime:** 1.17.1  
- **CUDA:** 11.8 (optional)  
- **Dependencies:** see `requirements.txt`

**Model artifacts:**  
- `models/tinyphysics.onnx`  
- `models/neural_blender_pretrained.onnx`  
- `models/neural_blender_champion.onnx`  

**Data artifacts:**  
- CSV files in `data/`  
- JSON archives in `plans/`

## 7. Pipeline Execution

- **Full run:**  
  ```bash
  python3 run_complete_training_pipeline.py \
    --stage1-combinations 300 --stage1-max-files 25 --stage1-num-files 50 \
    --t1-rounds 12 --t1-pop-size 25 --t1-max-files 30 \
    --t2-rounds 12 --t2-pop-size 25 --t2-max-files 50 \
    --stage4-samples 100 --stage4-epochs 20 --stage4-batch-size 64 \
    --blender-rounds 15 --blender-pop-size 20 --blender-max-files 25 \
    --data-seed 789
  ```

- **Validation:**  
  ```bash
  python3 eval_custom.py --data-dir data/ --controller neural_blended_champion
  python3 eval.py        --data-dir data/ --controller neural_blended_champion
  ```

## 8. Conclusion

This architecture reflects the current 7-stage optimization pipeline, aligns with our CLI guide, and ensures consistency across code and documentation.
