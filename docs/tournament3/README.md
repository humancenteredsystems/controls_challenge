# Tournament #3 Neural Blending - Quick Start

## Overview
Tournament #3 implements neural-weighted PID blending for lateral vehicle control. Neural networks determine optimal blending weights between high/low speed PID controllers based on vehicle state.

## Quick Setup

### 1. Install Dependencies
```bash
pip install onnxruntime torch numpy
```

### 2. Train Neural Model
```bash
python simple_neural_trainer.py
# Creates models/blender.onnx in 10 seconds
```

### 3. Use Controller
```python
from controllers.tournament3_simple import Controller

controller = Controller()
output = controller.update(target_lataccel, current_lataccel, state, future_plan)
```

### 4. Validate Setup
```python
# Test neural model loading
controller = Controller()
print(f"Neural status: {'LOADED' if controller.blender_session else 'FALLBACK'}")

# Test functionality
from collections import namedtuple
State = namedtuple('State', ['v_ego', 'roll_lataccel', 'a_ego'])
state = State(v_ego=45, roll_lataccel=0.1, a_ego=0.0)
output = controller.update(1.0, 0.5, state, None)
print(f"Controller output: {output:.3f}")
```

## Architecture
- **Neural Pipeline**: PyTorch training → ONNX export → CUDA inference
- **Controller Extension**: Inherits from Tournament #1/#2 architecture  
- **Model Discovery**: Single `blender.onnx` (vs multi-model `blender_*.onnx`)
- **Fallback**: Velocity-based blending when neural models unavailable