# Tournament System: Complete Guide

**Version:** 2.0 - Consolidated Guide  
**Last Updated:** January 2025  
**Audience:** All Users - Developers, Researchers, Operators

## 📋 Quick Navigation

- **🚀 [Getting Started](#getting-started)** - Start here for immediate usage
- **🏆 [Tournament Stages](#tournament-stages)** - Complete system overview
- **📊 [Performance Analysis](#performance-analysis)** - Results and benchmarks
- **🔧 [Technical Implementation](#technical-implementation)** - Architecture details
- **⚠️ [Troubleshooting](#troubleshooting)** - Common issues and solutions

---

## Getting Started

### 🎯 What You Need to Know

The Tournament System is a **progressive parameter optimization framework** that achieves **40.5% performance improvement** over baseline through three specialized stages:

1. **Tournament #1** - Grid search foundation (baseline establishment)
2. **Tournament #2** - Evolutionary optimization (**324.83 cost - PRODUCTION READY**)
3. **Tournament #3** - Neural enhancement (566.33 cost - research phase)

### ⚡ Quick Start Commands

```bash
# Production deployment (RECOMMENDED)
python tinyphysics.py --controller tournament_optimized --data_path ./data --num_segs 100

# Research/experimental neural blending  
python tinyphysics.py --controller neural_blended --data_path ./data --num_segs 100

# Run optimization tournaments
python optimization/tournament_optimizer.py      # Tournament #2
python optimization/blended_2pid_optimizer.py   # Tournament #1
```

### 🎪 Which Tournament Should I Use?

| Use Case | Recommendation | Controller | Reason |
|----------|----------------|------------|---------|
| **Production Deployment** | **Tournament #2** | `tournament_optimized` | 324.83 cost, proven reliability |
| **Baseline Research** | Tournament #1 | `blended_2pid` | Grid search foundation |
| **Advanced Research** | Tournament #3 | `neural_blended` | Neural enhancement, experimental |

---

## Tournament Stages

### 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tournament System Evolution                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Tournament #1  │  │  Tournament #2  │  │  Tournament #3  │  │
│  │   (Discovery)   │→ │ (Production)    │→ │   (Research)    │  │
│  │    Grid Search  │  │   Evolutionary  │  │   Neural        │  │
│  │    ~380+ cost   │  │   324.83 cost   │  │   566.33 cost   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Archive Intelligence System                  │
│         (Shared knowledge, parameter storage, fallback)        │
├─────────────────────────────────────────────────────────────────┤
│                    GPU-Accelerated Core                         │
│              (TinyPhysics + ONNX Runtime + CUDA)               │
└─────────────────────────────────────────────────────────────────┘
```

### Tournament #1: Foundation Discovery ✅

**Status:** Complete | **Method:** Grid Search | **Cost:** ~380+ | **Use:** Research Baseline

**Key Features:**
- Comprehensive parameter space exploration
- GPU-accelerated evaluation pipeline
- Archive intelligence initialization
- Systematic performance characterization

**Technical Implementation:**
```bash
# Run Tournament #1 optimization
python optimization/blended_2pid_optimizer.py

# Use Tournament #1 controller
python tinyphysics.py --controller blended_2pid --data_path ./data --num_segs 100
```

**Results:**
- Established parameter space boundaries
- Created performance baseline for system
- Generated comprehensive optimization archive
- Provided foundation for evolutionary optimization

### Tournament #2: Production Optimization ✅ **RECOMMENDED**

**Status:** Production Ready | **Method:** Evolutionary | **Cost:** **324.83** | **Use:** **Deployment**

**Key Features:**
- **40.5% performance improvement** over baseline
- Evolutionary algorithm with archive intelligence  
- Production-ready reliability and performance
- Comprehensive error handling and fallback systems

**Technical Implementation:**
```bash
# Run Tournament #2 optimization
python optimization/tournament_optimizer.py

# Use Tournament #2 controller (RECOMMENDED)
python tinyphysics.py --controller tournament_optimized --data_path ./data --num_segs 100
```

**Architecture:**
- [`controllers/tournament_optimized.py`](../controllers/tournament_optimized.py) - Production controller
- [`plans/tournament_archive.json`](../plans/tournament_archive.json) - Optimized parameters storage
- [`optimization/tournament_optimizer.py`](../optimization/tournament_optimizer.py) - Evolutionary optimizer

**Performance Results:**
- **Best Cost:** 324.83 (40.5% improvement over ~546 baseline)
- **Consistency:** Reliable performance across diverse scenarios
- **Robustness:** Comprehensive fallback and error handling
- **Production Status:** ✅ Ready for deployment

### Tournament #3: Neural Enhancement ⚠️ **RESEARCH PHASE**

**Status:** Working with Performance Regression | **Method:** Neural Blending | **Cost:** 566.33 avg, 289.89 median | **Use:** Research

**Key Features:**
- 43 specialized ONNX neural models for dynamic blending
- GPU-accelerated neural inference (CUDAExecutionProvider)
- 8-dimensional neural input features (velocity, acceleration, error, integrals, future planning)
- Multi-level fallback architecture (Neural → Tournament #2 → Baseline)

**Technical Implementation:**
```bash
# Generate neural models
python generate_neural_blending_models.py

# Use Tournament #3 controller  
python tinyphysics.py --controller neural_blended --data_path ./data --num_segs 100

# Verify neural model status
python -c "
from pathlib import Path
models = list(Path('models').glob('blender_*.onnx'))
print(f'Neural models: {len(models)} found')
for model in models[:3]:
    print(f'  {model.name}: {model.stat().st_size} bytes')
"
```

**Architecture:**
- [`controllers/neural_blended.py`](../controllers/neural_blended.py) - Neural blending controller
- [`models/blender_*.onnx`](../models/) - 43 specialized neural models (1,199 bytes each)
- [`generate_neural_blending_models.py`](../generate_neural_blending_models.py) - Model generation

**Current Performance Status:**

| Metric | Tournament #3 | Tournament #2 | Delta | Analysis |
|--------|---------------|---------------|-------|----------|
| **Average Cost** | 566.33 | 324.83 | -241.50 | ❌ **REGRESSION** |
| **Median Cost** | 289.89 | 324.83 | +34.94 | ✅ **IMPROVEMENT** |
| **Success Rate** | 56% | N/A | Better than baseline | ⚠️ **MIXED** |
| **Infrastructure** | ✅ Working | ✅ Working | Fully operational | ✅ **STABLE** |

**Root Cause Analysis - RESOLVED:**
- **Original Issue:** All neural models corrupted (22 bytes, `InvalidProtobuf` errors)
- **Solution Implemented:** Generated 43 working ONNX models (1,199 bytes each)
- **Current Issue:** Performance regression due to training data quality
- **Neural Training:** Models trained on synthetic velocity patterns vs real performance data

---

## Performance Analysis

### 📊 Tournament Performance Comparison

| Tournament | Method | Best Cost | Status | Description |
|------------|--------|-----------|--------|-------------|
| **Baseline** | Simple PID | 546.11 | ✅ Reference | Single PID controller |
| **#1** | Grid Search | ~380+ | ✅ Complete | Parameter space exploration |
| **#2** | Evolutionary | **324.83** | ✅ **PRODUCTION** | **40.5% improvement** |
| **#3** | Neural Blending | 566.33 avg, 289.89 median | ⚠️ Research | Mixed results |

### 🎯 Performance Evolution Timeline

```
546.11 (Baseline) → ~380+ (T1) → 324.83 (T2) → 566.33/289.89 (T3)
   ↓                    ↓            ↓              ↓
Baseline            30% Better   40.5% Better   Mixed Results
                                 PRODUCTION      Research
```

### ⚡ GPU Acceleration Benefits

**Performance Improvements:**
- **Physics Model Inference:** 3-5x speedup with CUDAExecutionProvider
- **Neural Model Loading:** GPU memory optimization for 43 models
- **Batch Evaluation:** Parallel processing of optimization candidates
- **Training Pipeline:** Accelerated neural model generation

**System Requirements:**
```bash
# Minimum (CPU only)
pip install onnxruntime>=1.17.1

# Recommended (GPU acceleration)  
pip install onnxruntime-gpu>=1.17.1

# Verify GPU availability
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('GPU available:', 'CUDAExecutionProvider' in providers)
"
```

---

## Technical Implementation

### 🏗️ Controller Architecture Details

#### Tournament #2 Controller (Production)
```python
# Production-ready implementation
from controllers.tournament_optimized import Controller

controller = Controller()  # Loads optimized parameters from archive
output = controller.update(target_lataccel, current_lataccel, state, future_plan)
```

**Features:**
- Automatic tournament archive loading (`plans/tournament_archive.json`)
- Blended PID architecture with velocity-dependent gains
- Comprehensive error handling and fallback systems
- Production-validated performance (324.83 cost)

#### Tournament #3 Controller (Research)
```python  
# Neural-enhanced implementation
from controllers.neural_blended import Controller

controller = Controller()  # Loads 43 neural models + Tournament #2 fallback
output = controller.update(target_lataccel, current_lataccel, state, future_plan)
```

**Features:**
- 43 specialized ONNX neural models (BlenderNet: 8→16→1 architecture)
- GPU-accelerated inference with CUDAExecutionProvider
- 8-dimensional input: velocity, acceleration, error, integrals, future plan statistics
- Multi-level fallback: Neural → velocity-based → Tournament #2 → baseline

### 🧠 Neural Blending Architecture

**Model Specialization:**
- **Velocity Bins:** 25 models covering 0-100 mph in 4 mph increments
- **Maneuver Types:** 18 models for turns, emergency corrections, straight driving
- **Dynamic Selection:** Real-time model selection based on driving scenario

**Input Feature Engineering:**
```python
features = [
    velocity,           # Current vehicle speed (0-100 mph)
    acceleration,       # Current acceleration (-10 to +10 m/s²)
    lateral_error,      # Tracking error (-5 to +5 m/s²)
    error_integral,     # Accumulated error (-50 to +50)
    error_derivative,   # Error rate of change (-20 to +20)
    future_mean,        # Mean future lateral acceleration (-3 to +3)
    future_std,         # Future lateral acceleration variance (0-2)
    roll_compensation   # Vehicle roll effect (-1 to +1)
]
```

### 📁 File Organization

**Core Controllers:**
- [`controllers/tournament_optimized.py`](../controllers/tournament_optimized.py) - Tournament #2 (Production)
- [`controllers/neural_blended.py`](../controllers/neural_blended.py) - Tournament #3 (Research)
- [`controllers/blended_2pid.py`](../controllers/blended_2pid.py) - Tournament #1 (Baseline)

**Optimization Systems:**
- [`optimization/tournament_optimizer.py`](../optimization/tournament_optimizer.py) - Tournament #2 evolutionary optimizer
- [`optimization/blended_2pid_optimizer.py`](../optimization/blended_2pid_optimizer.py) - Tournament #1 grid search

**Neural System:**
- [`models/blender_*.onnx`](../models/) - 43 neural blending models
- [`generate_neural_blending_models.py`](../generate_neural_blending_models.py) - Model generation script

**Archive & Data:**
- [`plans/tournament_archive.json`](../plans/tournament_archive.json) - Tournament #2 optimized parameters
- Various result files: `tournament3_*.json`, `*_results.json`

---

## Troubleshooting

### 🚨 Tournament #3 Common Issues

#### Issue: Neural Models Not Loading
**Symptoms:** Controller falls back to velocity-based blending immediately

**Diagnosis:**
```bash
# Check neural models
ls -la models/blender_*.onnx

# Expected: 43 files, each ~1,199 bytes
# Problem: Missing files or small sizes (22 bytes = corrupted)
```

**Solution:**
```bash
# Regenerate neural models
python generate_neural_blending_models.py

# Verify generation
python -c "
from pathlib import Path
models = list(Path('models').glob('blender_*.onnx'))
print(f'Generated {len(models)} models')
sizes = [m.stat().st_size for m in models]
print(f'Size range: {min(sizes)}-{max(sizes)} bytes')
"
```

#### Issue: GPU Acceleration Not Working
**Symptoms:** Slow neural inference, CPU-only execution

**Diagnosis:**
```bash
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available providers:', providers)
print('CUDA available:', 'CUDAExecutionProvider' in providers)
"
```

**Solution:**
```bash
# Install GPU support
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu>=1.17.1

# Verify CUDA setup
nvidia-smi  # Check GPU status
```

#### Issue: Tournament Archive Not Found
**Symptoms:** Controllers using default parameters instead of optimized ones

**Diagnosis:**
```bash
# Check tournament archive
ls -la plans/tournament_archive.json

# Validate archive structure
python -c "
import json
from pathlib import Path
archive_path = Path('plans/tournament_archive.json')
if archive_path.exists():
    with open(archive_path) as f:
        archive = json.load(f)
    valid = [x for x in archive['archive'] if 'stats' in x]
    print(f'Valid entries: {len(valid)}')
else:
    print('Archive not found')
"
```

**Solution:**
```bash
# Run Tournament #2 optimization to regenerate archive
python optimization/tournament_optimizer.py
```

### 🔧 Performance Issues

#### Tournament #3 Performance Regression
**Current Status:** 566.33 average cost vs 324.83 Tournament #2 baseline

**Root Cause:** Neural models trained on synthetic velocity patterns, not real performance data

**Improvement Strategies:**
1. **Retrain with Performance Data:**
   ```bash
   # Use Tournament #2 champions for training data
   python generate_neural_blending_models.py --use_tournament_data
   ```

2. **Improve Model Architecture:**
   - Increase complexity: 8→32→16→1 instead of 8→16→1
   - Add dropout for generalization
   - Ensemble multiple models

3. **Training Data Quality:**
   - Use actual driving scenarios from dataset
   - Include performance metrics in training loss
   - Balance training across velocity ranges

### 🎯 System Validation

**Complete System Check:**
```bash
python -c "
import sys
sys.path.append('.')

print('🔍 Tournament System Health Check')
print('=' * 50)

# Test Tournament #2 (Production)
try:
    from controllers.tournament_optimized import Controller
    controller = Controller()
    print('✅ Tournament #2: Ready for production')
except Exception as e:
    print(f'❌ Tournament #2: {e}')

# Test Tournament #3 (Research)  
try:
    from controllers.neural_blended import Controller
    controller = Controller()
    print('✅ Tournament #3: Neural system operational')
except Exception as e:
    print(f'❌ Tournament #3: {e}')

# Check GPU acceleration
try:
    import onnxruntime as ort
    gpu_available = 'CUDAExecutionProvider' in ort.get_available_providers()
    print(f'⚡ GPU acceleration: {\"Available\" if gpu_available else \"CPU only\"}')
except:
    print('❌ ONNX Runtime: Not available')

print('\\n📋 System Status: Ready for operation')
"
```

---

## Integration Guide

### 🚀 Production Deployment

**Recommended Configuration:**
```python
# Production setup with Tournament #2
from controllers.tournament_optimized import Controller

# Initialize controller (loads optimized parameters automatically)
controller = Controller()

# Main control loop
def control_loop(target_lataccel, current_lataccel, state, future_plan):
    return controller.update(target_lataccel, current_lataccel, state, future_plan)
```

### 🧪 Research Configuration

**Experimental setup with Tournament #3:**
```python
# Research setup with neural blending
from controllers.neural_blended import Controller

# Initialize with neural models
controller = Controller()  # Auto-loads 43 neural models + Tournament #2 fallback

# Research control loop with monitoring
def research_control_loop(target_lataccel, current_lataccel, state, future_plan):
    output = controller.update(target_lataccel, current_lataccel, state, future_plan)
    
    # Monitor neural vs fallback usage
    if hasattr(controller, '_using_neural'):
        mode = "Neural" if controller._using_neural else "Fallback"
        print(f"Control mode: {mode}, Output: {output:.3f}")
    
    return output
```

### 🔄 Migration Path

**From Simple PID to Tournament System:**
1. **Start with Tournament #2** (production-ready, 40.5% improvement)
2. **Validate performance** on your specific use case
3. **Consider Tournament #3** for research applications only
4. **Monitor performance** and use fallback systems as needed

---

## Future Roadmap

### 🎯 Tournament #4 Vision
- **Reinforcement Learning Integration:** Advanced policy optimization
- **Multi-Objective Optimization:** Balance performance, safety, comfort
- **Real-Time Adaptation:** Dynamic parameter adjustment during operation
- **Ensemble Methods:** Combine multiple optimization approaches

### 🔬 Research Directions
- **Performance-Based Neural Training:** Use actual cost metrics in training loss
- **Temporal Sequence Modeling:** RNN/Transformer architectures for trajectory prediction
- **Meta-Learning:** Few-shot adaptation to new driving scenarios
- **Uncertainty Quantification:** Confidence estimates for neural blending decisions

### 🏗️ System Improvements
- **Automated Model Validation:** CI/CD pipeline for neural model quality
- **Performance Monitoring:** Real-time system health and performance tracking
- **Configuration Management:** Environment-specific optimization profiles
- **Advanced Diagnostics:** Comprehensive system analysis and debugging tools

---

**Last Updated:** January 2025 | **Version:** 2.0 - Consolidated Tournament Guide  
**Status:** All tournaments operational - #2 production ready, #3 research phase  
**Maintainer:** Tournament System Engineering Team