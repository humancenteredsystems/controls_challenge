# Neural Blending User Guide: Tournament #3 System

**Version:** 1.0  
**Last Updated:** January 2025  
**Difficulty:** Intermediate to Advanced

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [System Requirements](#system-requirements)
4. [Usage Examples](#usage-examples)
5. [Configuration Options](#configuration-options)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)
9. [Integration Guide](#integration-guide)
10. [Best Practices](#best-practices)

## Overview

The Tournament #3 Neural Blending System combines traditional PID control with learned neural network weights to provide intelligent, velocity-specific control optimization. Unlike static blending approaches, neural blending adapts control weights based on real-time driving conditions using specialized neural networks.

### 🎯 **What Neural Blending Does**

**Traditional Approach** (Tournament #2):
```python
# Fixed velocity-based blending
weight = 0.8 if v_ego < 40 else 0.2
output = weight * low_pid + (1-weight) * high_pid
```

**Neural Blending Approach** (Tournament #3):
```python
# Dynamic neural weight calculation
features = [velocity, acceleration, error, integrals, future_plan_stats]
neural_weight = neural_model.predict(features)
output = neural_weight * low_pid + (1-neural_weight) * high_pid  
```

### 🏆 **Performance Characteristics**

| Metric | Value | Notes |
|--------|-------|-------|
| **Infrastructure Status** | ✅ Fully Operational | 43 working neural models |
| **GPU Acceleration** | ✅ Active | CUDAExecutionProvider enabled |
| **Current Performance** | ⚠️ Mixed Results | 566.33 avg, 289.89 median cost |
| **Success Rate** | 56% | Files performing better than baseline |
| **Fallback Reliability** | ✅ Robust | Automatic Tournament #2 fallback |

## Quick Start

### 1. **Basic Usage**

```python
from controllers.neural_blended import Controller
from collections import namedtuple

# Create data structures (as provided by framework)
State = namedtuple('State', ['v_ego', 'roll_lataccel', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel'])

# Initialize neural blending controller
controller = Controller()

# Use in control loop
state = State(v_ego=45, roll_lataccel=0.1, a_ego=0.0)
future_plan = FuturePlan(lataccel=[0.0, 0.5, 1.0])

output = controller.update(
    target_lataccel=1.5, 
    current_lataccel=1.0, 
    state=state, 
    future_plan=future_plan
)
```

### 2. **Integration with Evaluation Framework**

```python
# Compatible with existing evaluation pipeline
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
                      --data_path ./data/00000.csv \
                      --controller neural_blended

# Batch evaluation
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
                      --data_path ./data \
                      --num_segs 100 \
                      --controller neural_blended
```

### 3. **Verification Test**

```python
# Quick system check
python -c "
from controllers.neural_blended import Controller
controller = Controller()
print(f'Neural blending controller ready: {controller}')
"
```

## System Requirements

### 🔧 **Hardware Requirements**

**Minimum:**
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 8GB system memory
- GPU: NVIDIA GPU with CUDA 11.8+ support (optional but recommended)

**Recommended:**
- CPU: 8+ core processor  
- RAM: 16GB system memory
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: SSD for model loading performance

### 📦 **Software Dependencies**

```bash
# Core dependencies
pip install onnxruntime-gpu>=1.17.1  # GPU acceleration
pip install numpy>=1.20.0
pip install torch>=1.9.0            # For model generation

# Verify CUDA setup (if using GPU)
python -c "import onnxruntime as ort; print('CUDA available:', 'CUDAExecutionProvider' in ort.get_available_providers())"
```

### 📁 **Required Files**

```
models/
├── blender_0.onnx     # Neural blending models
├── blender_1.onnx     # (43 models total)
├── ...
└── blender_42.onnx

plans/
└── tournament_archive.json  # Tournament #2 parameters

controllers/
└── neural_blended.py        # Controller implementation
```

## Usage Examples

### 🚗 **Example 1: Basic Control Loop**

```python
import numpy as np
from controllers.neural_blended import Controller
from collections import namedtuple

# Initialize
controller = Controller()
State = namedtuple('State', ['v_ego', 'roll_lataccel', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel'])

# Driving scenarios
scenarios = [
    # Low-speed turn
    {
        'state': State(v_ego=25, roll_lataccel=0.0, a_ego=0.0),
        'target': 2.0, 'current': 1.5,
        'future_plan': FuturePlan(lataccel=[1.5, 1.8, 2.0])
    },
    # High-speed turn  
    {
        'state': State(v_ego=65, roll_lataccel=0.1, a_ego=-1.0),
        'target': 1.0, 'current': 0.8,
        'future_plan': FuturePlan(lataccel=[0.8, 0.9, 1.0])
    }
]

for i, scenario in enumerate(scenarios):
    output = controller.update(
        scenario['target'], 
        scenario['current'], 
        scenario['state'], 
        scenario['future_plan']
    )
    print(f"Scenario {i+1}: Output = {output:.3f}")
```

### 📊 **Example 2: Performance Monitoring**

```python
import time
from controllers.neural_blended import Controller

# Performance timing test
controller = Controller()
times = []

for _ in range(100):
    start_time = time.time()
    output = controller.update(1.5, 1.0, state, future_plan)
    elapsed = time.time() - start_time
    times.append(elapsed)

print(f"Average inference time: {np.mean(times)*1000:.2f}ms")
print(f"GPU acceleration: {'✅' if controller.blender_session else '❌'}")
```

### 🔄 **Example 3: Fallback Testing**

```python
# Test fallback behavior
controller = Controller()

# Check neural model status
if controller.blender_session:
    print("✅ Neural blending active")
else:
    print("⚠️ Using velocity-based fallback")
    print("📊 Fallback parameters loaded from Tournament #2 archive")

# Both modes produce valid outputs
output = controller.update(1.5, 1.0, state, future_plan)
print(f"Control output: {output:.3f}")
```

## Configuration Options

### 🎛️ **Controller Initialization Options**

```python
# Default initialization (recommended)
controller = Controller()

# Custom blending model path
controller = Controller(blender_model_path="custom_models/my_blender.onnx")

# Access internal configuration
print(f"Low-speed PID: {controller.low_speed_pid}")
print(f"High-speed PID: {controller.high_speed_pid}")
print(f"Neural session active: {controller.blender_session is not None}")
```

### ⚙️ **ONNX Runtime Configuration**

```python
# Configure ONNX session (advanced users)
import onnxruntime as ort

# Check available providers
providers = ort.get_available_providers()
print(f"Available providers: {providers}")

# Session options (handled automatically by controller)
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

### 🗃️ **Tournament Archive Integration**

```python
# Verify Tournament #2 parameter loading
from pathlib import Path
import json

archive_path = Path('plans/tournament_archive.json')
if archive_path.exists():
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    # Find best parameters
    best_entry = min(archive['archive'], 
                    key=lambda x: x.get('stats', {}).get('avg_total_cost', float('inf')))
    
    print(f"Loaded Tournament #2 parameters:")
    print(f"  Low-speed: {best_entry.get('pid1_gains', 'N/A')}")
    print(f"  High-speed: {best_entry.get('pid2_gains', 'N/A')}")
    print(f"  Best cost: {best_entry.get('stats', {}).get('avg_total_cost', 'N/A')}")
```

## Performance Optimization

### 🚀 **GPU Acceleration**

**Enable GPU Acceleration:**
```bash
# Install GPU-enabled ONNX Runtime
pip install onnxruntime-gpu

# Verify CUDA setup
nvidia-smi
python -c "import onnxruntime as ort; print('CUDA providers:', [p for p in ort.get_available_providers() if 'CUDA' in p])"
```

**Performance Benefits:**
- **Neural Inference**: 3-5x speedup with GPU
- **Physics Simulation**: GPU-accelerated TinyPhysics model
- **Memory Efficiency**: Optimized tensor operations

### 📈 **Performance Monitoring**

```python
# Monitor performance characteristics
import time
import numpy as np

def benchmark_controller(controller, iterations=1000):
    times = []
    for _ in range(iterations):
        start = time.time()
        output = controller.update(1.0, 0.5, state, future_plan)
        times.append(time.time() - start)
    
    return {
        'mean_time': np.mean(times) * 1000,  # ms
        'std_time': np.std(times) * 1000,    # ms
        'min_time': np.min(times) * 1000,    # ms
        'max_time': np.max(times) * 1000     # ms
    }

# Run benchmark
stats = benchmark_controller(controller)
print(f"Performance: {stats['mean_time']:.2f}±{stats['std_time']:.2f}ms")
```

### 🎯 **Optimization Tips**

1. **Model Loading**: Neural models are lazy-loaded on first use
2. **GPU Memory**: Models stay in GPU memory after first inference
3. **Batch Processing**: For multiple evaluations, reuse controller instance
4. **Fallback Performance**: Velocity-based fallback is ~10x faster than neural

## Troubleshooting

### ❌ **Common Issues**

#### **Issue 1: Neural Models Not Loading**

**Symptoms:**
```
⚠️ Neural blending failed: [ONNXRuntime] Unable to load model
🔄 Falling back to velocity-based blending
```

**Solutions:**
```bash
# Check model files exist
ls -la models/blender_*.onnx

# Verify model integrity
python -c "
import onnxruntime as ort
try:
    session = ort.InferenceSession('models/blender_0.onnx')
    print('✅ Model loads correctly')
except Exception as e:
    print(f'❌ Model error: {e}')
"

# Regenerate models if corrupted
python generate_neural_blending_models.py
```

#### **Issue 2: GPU Not Available**

**Symptoms:**
```
🔧 GPU acceleration: CPU ONLY
Warning: CUDA not available, using CPU fallback
```

**Solutions:**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install GPU-enabled ONNX Runtime
pip uninstall onnxruntime
pip install onnxruntime-gpu

# Verify GPU providers
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('GPU available:', 'CUDAExecutionProvider' in providers)
"
```

#### **Issue 3: Tournament Archive Missing**

**Symptoms:**
```
⚠️ Tournament archive not found - will use fallback parameters
FileNotFoundError: plans/tournament_archive.json
```

**Solutions:**
```bash
# Verify archive exists
ls -la plans/tournament_archive.json

# Create directory if missing
mkdir -p plans

# Run Tournament #2 to generate archive (if needed)
python optimization/tournament_optimizer.py
```

### 🔧 **Diagnostic Commands**

```python
# Complete system diagnostic
python -c "
import sys
sys.path.append('.')

print('🔍 Neural Blending Diagnostics')
print('=' * 40)

# 1. Import test
try:
    from controllers.neural_blended import Controller
    print('✅ Controller import: SUCCESS')
except Exception as e:
    print(f'❌ Controller import: {e}')
    exit(1)

# 2. Dependencies
try:
    import onnxruntime as ort
    print(f'✅ ONNX Runtime: {ort.__version__}')
    providers = ort.get_available_providers()
    gpu_available = 'CUDAExecutionProvider' in providers
    print(f'🔧 GPU acceleration: {\"AVAILABLE\" if gpu_available else \"CPU ONLY\"}')
except Exception as e:
    print(f'❌ ONNX Runtime: {e}')

# 3. Models
from pathlib import Path
models_dir = Path('models')
if models_dir.exists():
    blender_models = list(models_dir.glob('blender_*.onnx'))
    print(f'📄 Neural models: {len(blender_models)} found')
    if blender_models:
        sizes = [m.stat().st_size for m in blender_models[:3]]
        print(f'📏 Model sizes: {sizes} bytes (showing first 3)')
else:
    print('❌ Models directory not found')

# 4. Tournament archive
archive_path = Path('plans/tournament_archive.json')
if archive_path.exists():
    print('✅ Tournament archive: FOUND')
else:
    print('⚠️ Tournament archive: MISSING (will use fallback)')

# 5. Controller initialization
try:
    controller = Controller()
    print(f'✅ Controller init: SUCCESS')
    print(f'🧠 Neural blending: {\"ACTIVE\" if controller.blender_session else \"FALLBACK\"}')
except Exception as e:
    print(f'❌ Controller init: {e}')
"
```

## Advanced Usage

### 🧪 **Custom Neural Models**

```python
# Using custom trained models
controller = Controller(blender_model_path="custom_models/optimized_blender.onnx")

# Multiple model evaluation
model_paths = [
    "models/blender_0.onnx",
    "custom_models/experimental_blender.onnx"
]

results = {}
for model_path in model_paths:
    controller = Controller(blender_model_path=model_path)
    output = controller.update(target, current, state, future_plan)
    results[model_path] = output

print("Model comparison:", results)
```

### 🔬 **Feature Analysis**

```python
# Analyze neural network input features
def analyze_features(controller, state, future_plan, target, current):
    if hasattr(controller, '_extract_features'):
        features = controller._extract_features(state, future_plan, target, current)
        
        feature_names = [
            'velocity', 'current_lataccel', 'roll_lataccel', 'a_ego',
            'error', 'low_pid_integral', 'high_pid_integral', 'future_plan_stat'
        ]
        
        for name, value in zip(feature_names, features.flatten()):
            print(f"{name}: {value:.3f}")
    
    return controller.update(target, current, state, future_plan)

# Feature analysis example
output = analyze_features(controller, state, future_plan, 1.5, 1.0)
```

### 📊 **Performance Profiling**

```python
import cProfile
import pstats

# Profile neural blending performance
def profile_neural_blending():
    controller = Controller()
    
    for _ in range(100):
        output = controller.update(1.5, 1.0, state, future_plan)
    
    return controller

# Run profiler
profiler = cProfile.Profile()
profiler.enable()
profile_neural_blending()
profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

## Integration Guide

### 🔌 **Integration with Evaluation Pipeline**

```python
# Custom evaluation script
import sys
sys.path.append('.')
from controllers.neural_blended import Controller
from tinyphysics import run_rollout

def evaluate_neural_blending(data_files, model_path):
    """Evaluate neural blending on multiple files"""
    controller = Controller()
    results = []
    
    for data_file in data_files:
        try:
            result = run_rollout(data_file, controller, model_path)
            results.append({
                'file': data_file,
                'cost': result.get('total_cost', float('inf')),
                'neural_active': controller.blender_session is not None
            })
        except Exception as e:
            print(f"Error processing {data_file}: {e}")
    
    return results

# Usage
data_files = ['data/00000.csv', 'data/00001.csv']
results = evaluate_neural_blending(data_files, 'models/tinyphysics.onnx')
```

### 🏆 **Tournament System Integration**

```python
# Integration with tournament optimization
from optimization.tournament_optimizer import run_tournament

# Custom tournament with neural blending evaluation
def neural_tournament_evaluation(params, data_files, model):
    """Evaluate parameters using neural blending"""
    # Create temporary controller with custom parameters
    controller = Controller()
    controller.low_speed_pid.set_gains(*params['low_gains'])
    controller.high_speed_pid.set_gains(*params['high_gains'])
    
    costs = []
    for data_file in data_files:
        result = run_rollout(data_file, controller, model)
        costs.append(result['total_cost'])
    
    return sum(costs) / len(costs)  # Average cost
```

## Best Practices

### ✅ **Recommended Practices**

1. **Always Check Neural Model Status**
   ```python
   controller = Controller()
   if controller.blender_session:
       print("Neural blending active")
   else:
       print("Using fallback mode")
   ```

2. **Reuse Controller Instances**
   ```python
   # Good: Reuse for batch processing
   controller = Controller()
   for data_file in data_files:
       result = evaluate_file(data_file, controller)
   
   # Avoid: Creating new instance each time
   for data_file in data_files:
       controller = Controller()  # Unnecessary model loading
       result = evaluate_file(data_file, controller)
   ```

3. **Monitor Performance Regression**
   ```python
   # Compare against Tournament #2 baseline
   TOURNAMENT_2_BASELINE = 324.83
   
   neural_cost = evaluate_neural_blending(data_files)
   regression = neural_cost - TOURNAMENT_2_BASELINE
   
   if regression > 0:
       print(f"⚠️ Performance regression: {regression:.2f}")
   else:
       print(f"✅ Performance improvement: {-regression:.2f}")
   ```

4. **Handle Fallback Gracefully**
   ```python
   controller = Controller()
   
   # Fallback is not a failure - it's a feature
   if not controller.blender_session:
       print("Using proven Tournament #2 parameters as fallback")
   ```

### ⚠️ **Common Pitfalls**

1. **Don't Assume Neural Models Work Better**
   - Current neural models show performance regression
   - Fallback mode may perform better in many cases
   - Always benchmark against Tournament #2 baseline

2. **Don't Ignore GPU Setup**
   - GPU acceleration provides significant speedup
   - Verify CUDA installation and ONNX Runtime GPU support
   - CPU fallback works but is slower

3. **Don't Skip Error Handling**
   ```python
   # Good: Handle potential failures
   try:
       controller = Controller()
       output = controller.update(target, current, state, future_plan)
   except Exception as e:
       print(f"Neural blending failed: {e}")
       # Implement fallback logic
   
   # Bad: Assume everything works
   controller = Controller()  # May fail silently
   output = controller.update(target, current, state, future_plan)
   ```

### 🎯 **Performance Expectations**

**Current Reality** (be aware of limitations):
- **Average Performance**: 566.33 cost (worse than Tournament #2)
- **Median Performance**: 289.89 cost (better than Tournament #2)  
- **Success Rate**: 56% of files perform better than baseline
- **High Variance**: Standard deviation of 905.62 indicates inconsistency

**When to Use Neural Blending**:
- ✅ Research and development
- ✅ Experimental evaluation
- ✅ Learning about neural-enhanced control
- ⚠️ Production deployment (consider Tournament #2 instead)

**When to Use Tournament #2**:
- ✅ Production deployment
- ✅ Consistent performance requirements
- ✅ Proven 324.83 cost baseline
- ✅ Lower variance and more predictable results

---

## Support and Resources

- **Architecture Documentation**: [`docs/architecture.md`](architecture.md)
- **Status Report**: [`TOURNAMENT_3_NEURAL_BLENDING_STATUS_REPORT.md`](../TOURNAMENT_3_NEURAL_BLENDING_STATUS_REPORT.md)
- **Implementation**: [`controllers/neural_blended.py`](../controllers/neural_blended.py)
- **Model Generation**: [`generate_neural_blending_models.py`](../generate_neural_blending_models.py)

For issues and questions, refer to the troubleshooting section or examine the diagnostic output for system-specific problems.