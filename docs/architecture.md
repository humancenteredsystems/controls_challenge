# Optimization System Architecture

GPU-accelerated platform for tuning autonomous steering controllers.

## Core Concepts
- **TinyPhysicsModel** uses ONNX Runtime with CUDA and falls back to CPU when necessary.
- **run_rollout** accepts a model path or instance, enabling model reuse and eliminating repeated initialization.
- Sequential execution avoids resource conflicts while keeping backward compatibility.

## High-Level Stack
- **Optimizers:** blended 2-PID, tournament, comprehensive.
- **GPU Core:** cached TinyPhysicsModel with CUDAExecutionProvider.
- **Controllers:** PID variants including blended and ensemble designs.
- **Evaluation:** `eval_custom.py` for local checks and `eval.py` for challenge validation.
- **Simulation:** TinyPhysicsSimulator provides the physics environment.

## Observability
Debug output reports active providers and GPU status, helping track performance and memory use.

## Roadmap
Future improvements target multi-GPU support, distributed scaling, and advanced GPU memory optimization.
