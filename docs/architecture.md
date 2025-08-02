# Controller Optimization Pipeline Architecture

**Version:** 2.0  \
**Status:** Target architecture  \
**Date:** January 2025

## 1. Purpose and Scope

This document defines the architecture for the Comma.ai controller optimization
pipeline.  It captures the software components, data flow and operational
constraints required to evaluate steering controllers with GPU‑accelerated
physics simulation.  The intent is to provide a shared reference for engineers
extending or operating the system.

## 2. System Goals

* **Fast evaluation** – leverage a single cached ONNX Runtime session to drive
  3–5× more rollouts than CPU‑only execution.
* **Deterministic operation** – sequential execution avoids resource conflicts
  and ensures reproducible results.
* **Extensibility** – controller implementations and optimizers can be swapped
  without changing the surrounding pipeline.

## 3. Architectural Overview

The pipeline evaluates candidate controllers against recorded driving logs using
a GPU‑accelerated physics model.  Optimization drivers generate parameter sets,
run simulations through the model, score the results and feed new candidates
back into the search algorithm.

```
data generation ──▶ optimizer ──▶ controller ──▶ physics model ──▶ evaluation
```

### 3.1 Component Summary

**TinyPhysicsModel** – ONNX model wrapper providing CUDA execution with CPU
fallback.  It exposes a lightweight API used by all optimizers.

**Controller Library** – PID, blended 2‑PID and ensemble controllers stored in
`controllers/`.  Each exposes a unified interface for rollouts.

**Optimization Drivers** – scripts such as `run_complete_training_pipeline.py`
and `run_two_stage_tournament.py` that orchestrate parameter search strategies.

**Evaluation Tools** – `eval.py` supplies the official challenge validation
while `eval_custom.py` provides local pre‑submission checks.

### 3.2 Data Flow

1. Training data is generated or loaded from `plans/` and `validation/` logs.
2. An optimizer constructs parameter sets for a chosen controller.
3. For each parameter set the controller performs a rollout by invoking
   `run_rollout` in `tinyphysics.py`.
4. `TinyPhysicsModel` executes the ONNX graph on the GPU and returns vehicle
   dynamics and cost metrics.
5. The optimizer scores the rollout and decides on the next parameters to try.
6. Top performing configurations are serialized to `optimization_progress.json`
   and optionally validated with `eval.py`.

## 4. Technology Stack

* **Language:** Python 3.8+
* **Physics model:** ONNX Runtime 1.17.1 with CUDA 11.8 and cuDNN 8.9
* **Compute:** single NVIDIA GPU; automatic CPU fallback if unavailable
* **Data format:** JSON logs for optimization progress and tournament archives

## 5. Operational Considerations

### 5.1 Performance

The system reuses a single `TinyPhysicsModel` instance per run, eliminating the
costly model initialization that previously occurred on every evaluation.  This
reduces startup overhead by ~90 % and allows 50–250 evaluations per second on a
mid‑range GPU.

### 5.2 Monitoring and Debugging

* ONNX providers are logged during model creation to confirm GPU availability.
* Optimizers emit throughput statistics and cache progress to
  `optimization_progress.json` for post‑mortem analysis.
* Debug mode prints detailed provider status and timing information.

## 6. Future Enhancements

* **Multi‑GPU scaling** – distribute parameter sweeps across devices or nodes.
* **Batch rollouts** – run multiple trajectories per kernel launch for higher
  utilization.
* **Adaptive search** – incorporate variance analysis and scenario‑aware
  optimization algorithms.

---

The architecture above forms the basis for a reliable, extensible and
GPU‑accelerated controller optimization pipeline.

