# Neural Blending Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting steps for the Tournament #3 neural blending system. It covers common issues, diagnostic procedures, and solutions for neural model problems.

## 🚨 Critical Issues

### 1. Corrupted Neural Models

**Symptoms:**
- Models are only 22 bytes in size (should be ~1,199 bytes)
- `InvalidProtobuf` errors during model loading
- Controller falls back to velocity-based blending immediately
- Error messages: "Invalid ONNX model" or "Corrupted protobuf"

**Diagnosis:**
```bash
# Check model file sizes
ls -la models/blender_*.onnx

# Expected output: Each file should be ~1,199 bytes
# Problem: Files showing 22 bytes or other small sizes
```

**Solution:**
```bash
# Regenerate all neural models
python generate_neural_blending_models.py

# Verify models are properly generated
python -c "
import onnxruntime as ort
from pathlib import Path

models_dir = Path('models')
blender_models = list(models_dir.glob('blender_*.onnx'))
print(f'Found {len(blender_models)} models')

for model_path in blender_models[:3]:  # Test first 3
    try:
        session = ort.InferenceSession(str(model_path))
        print(f'✅ {model_path.name}: Valid ({model_path.stat().st_size} bytes)')
    except Exception as e:
        print(f'❌ {model_path.name}: Invalid - {e}')
"
```

### 2. Missing Neural Models

**Symptoms:**
- `models/` directory doesn't exist
- No `blender_*.onnx` files found
- Controller immediately uses fallback parameters
- Warning: "No neural models found, using fallback"

**Diagnosis:**
```bash
# Check if models directory exists
ls -la models/

# Check for specific blender models
find . -name "blender_*.onnx" -type f
```

**Solution:**
```bash
# Create models directory if missing
mkdir -p models

# Generate neural models
python generate_neural_blending_models.py

# Verify 43 models were created
ls models/blender_*.onnx | wc -l  # Should output: 43
```

### 3. ONNX Runtime Issues

**Symptoms:**
- `ModuleNotFoundError: No module named 'onnxruntime'`
- GPU acceleration not available when expected
- Slow neural inference performance

**Diagnosis:**
```bash
# Check ONNX Runtime installation
python -c "
try:
    import onnxruntime as ort
    print(f'ONNXRuntime version: {ort.__version__}')
    providers = ort.get_available_providers()
    print(f'Available providers: {providers}')
    print(f'GPU available: {\"CUDAExecutionProvider\" in providers}')
except ImportError as e:
    print(f'ONNX Runtime not installed: {e}')
"
```

**Solution:**
```bash
# Install CPU version (minimum requirement)
pip install onnxruntime>=1.17.1

# Install GPU version (recommended for performance)
pip install onnxruntime-gpu>=1.17.1

# Verify CUDA compatibility (if using GPU)
python -c "
import onnxruntime as ort
session = ort.InferenceSession('models/blender_0.onnx', 
                              providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(f'Active provider: {session.get_providers()[0]}')
"
```

## ⚠️ Performance Issues

### 4. Performance Regression

**Symptoms:**
- Tournament #3 cost (566.33) worse than Tournament #2 (324.83)
- High variance in results
- Some segments perform well (median 289.89) but others poorly

**Diagnosis:**
```bash
# Run comparative evaluation
python tinyphysics.py --controller tournament_optimized --data_path ./data --num_segs 20
python tinyphysics.py --controller neural_blended --data_path ./data --num_segs 20

# Check neural model training quality
python -c "
from controllers.neural_blended import Controller
controller = Controller()
print(f'Neural models loaded: {len(controller.neural_models) if hasattr(controller, \"neural_models\") else 0}')
print(f'Using fallback: {controller._using_fallback if hasattr(controller, \"_using_fallback\") else \"Unknown\"}')
"
```

**Root Cause Analysis:**
- Neural models trained on synthetic velocity-based patterns
- Lack of performance-optimized real driving data
- Models not learning optimal control strategies

**Solutions:**
1. **Retrain with Performance Data:**
   ```bash
   # Use Tournament #2 champions for training data
   python generate_neural_blending_models.py --use_tournament_data
   ```

2. **Adjust Model Architecture:**
   - Increase model complexity (8→32→16→1 instead of 8→16→1)
   - Add dropout for better generalization
   - Use different activation functions

3. **Improve Training Data Quality:**
   - Use actual driving scenarios from dataset
   - Include performance metrics in training loss
   - Balance training data across velocity ranges

### 5. GPU Memory Issues

**Symptoms:**
- `CUDA out of memory` errors
- System crashes during neural inference
- Slow performance despite GPU availability

**Diagnosis:**
```bash
# Check GPU memory usage
nvidia-smi

# Test memory-limited model loading
python -c "
import onnxruntime as ort
import psutil
import os

print(f'System RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')
print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')

# Try loading models with memory monitoring
for i in range(5):  # Test first 5 models
    try:
        session = ort.InferenceSession(f'models/blender_{i}.onnx',
                                     providers=['CUDAExecutionProvider'])
        print(f'Model {i}: OK')
    except Exception as e:
        print(f'Model {i}: Failed - {e}')
"
```

**Solutions:**
```bash
# Reduce batch size in neural inference
# Edit controllers/neural_blended.py to use smaller batches

# Use CPU execution if GPU memory insufficient
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode

# Optimize ONNX Runtime memory usage
python -c "
import onnxruntime as ort
session_options = ort.SessionOptions()
session_options.enable_mem_pattern = False
session_options.enable_cpu_mem_arena = False
# Use these options when creating inference sessions
"
```

## 🔧 System Integration Issues

### 6. Tournament Archive Problems

**Symptoms:**
- Neural blending falls back to default parameters
- No Tournament #2 optimization benefits
- Warning: "Tournament archive not found"

**Diagnosis:**
```bash
# Check tournament archive existence and format
ls -la plans/tournament_archive.json

# Validate archive structure
python -c "
import json
from pathlib import Path

archive_path = Path('plans/tournament_archive.json')
if archive_path.exists():
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    print(f'Archive entries: {len(archive.get(\"archive\", []))}')
    
    valid_entries = [x for x in archive[\"archive\"] 
                    if \"stats\" in x and \"avg_total_cost\" in x[\"stats\"]]
    print(f'Valid entries: {len(valid_entries)}')
    
    if valid_entries:
        best = min(valid_entries, key=lambda x: x[\"stats\"][\"avg_total_cost\"])
        print(f'Best cost: {best[\"stats\"][\"avg_total_cost\"]}')
else:
    print('Archive not found')
"
```

**Solution:**
```bash
# Regenerate tournament archive if corrupted
python optimization/tournament_optimizer.py --quick_archive_rebuild

# Verify archive loading in neural controller
python -c "
from controllers.neural_blended import Controller
controller = Controller()
print('Controller initialized successfully with tournament fallback')
"
```

### 7. Import and Dependency Conflicts

**Symptoms:**
- `ImportError` when importing neural_blended controller
- Conflicts between tinyphysics and tinyphysics_custom
- Missing dependencies for neural model training

**Diagnosis:**
```bash
# Test all required imports
python -c "
import sys
sys.path.append('.')

try:
    from controllers.neural_blended import Controller
    print('✅ Neural controller import: OK')
except Exception as e:
    print(f'❌ Neural controller import: {e}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError:
    print('❌ PyTorch: Missing')

try:
    import onnxruntime as ort
    print(f'✅ ONNX Runtime: {ort.__version__}')
except ImportError:
    print('❌ ONNX Runtime: Missing')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError:
    print('❌ NumPy: Missing')
"
```

**Solution:**
```bash
# Install all required dependencies
pip install torch>=1.9.0
pip install onnxruntime-gpu>=1.17.1  # or onnxruntime for CPU-only
pip install numpy>=1.20.0

# Fix import path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify installation
python -c "from controllers.neural_blended import Controller; print('All imports successful')"
```

## 🛠️ Diagnostic Tools

### Quick System Check Script

```bash
python -c "
import sys
sys.path.append('.')
from pathlib import Path
import json

print('🔍 Neural Blending System Diagnostic')
print('=' * 50)

# 1. Check models directory
models_dir = Path('models')
if models_dir.exists():
    blender_models = list(models_dir.glob('blender_*.onnx'))
    print(f'✅ Neural models: {len(blender_models)} found')
    
    if blender_models:
        sizes = [m.stat().st_size for m in blender_models]
        avg_size = sum(sizes) / len(sizes)
        print(f'   Average size: {avg_size:.0f} bytes (expected: ~1,199)')
        
        if avg_size < 100:
            print('   ⚠️  Models appear corrupted (too small)')
        elif avg_size > 1000:
            print('   ✅ Models appear healthy')
    else:
        print('   ❌ No blender_*.onnx models found')
else:
    print('❌ models/ directory not found')

# 2. Check dependencies
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    gpu_available = 'CUDAExecutionProvider' in providers
    print(f'✅ ONNX Runtime: {ort.__version__} (GPU: {gpu_available})')
except ImportError:
    print('❌ ONNX Runtime: Not installed')

# 3. Check tournament archive
archive_path = Path('plans/tournament_archive.json')
if archive_path.exists():
    try:
        with open(archive_path, 'r') as f:
            archive = json.load(f)
        valid_entries = [x for x in archive['archive'] 
                        if 'stats' in x and 'avg_total_cost' in x['stats']]
        print(f'✅ Tournament archive: {len(valid_entries)} valid entries')
    except Exception as e:
        print(f'❌ Tournament archive: Corrupted - {e}')
else:
    print('⚠️  Tournament archive: Not found (will use defaults)')

# 4. Test controller initialization
try:
    from controllers.neural_blended import Controller
    controller = Controller()
    print('✅ Neural controller: Initialization successful')
except Exception as e:
    print(f'❌ Neural controller: Initialization failed - {e}')

print('\\n📋 Diagnostic complete')
"
```

### Performance Benchmark Script

```bash
python -c "
import sys
sys.path.append('.')
import time
import numpy as np
from collections import namedtuple

print('⚡ Neural Blending Performance Test')
print('=' * 40)

State = namedtuple('State', ['v_ego', 'roll_lataccel', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel'])

try:
    from controllers.neural_blended import Controller
    controller = Controller()
    
    # Warm-up
    state = State(v_ego=40, roll_lataccel=0.1, a_ego=0.0)
    future_plan = FuturePlan(lataccel=[0.0, 0.5, 1.0])
    controller.update(1.0, 0.5, state, future_plan)
    
    # Benchmark
    times = []
    for i in range(100):
        start = time.time()
        output = controller.update(1.0, 0.5, state, future_plan)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    print(f'Average inference time: {avg_time:.2f} ms')
    print(f'Throughput: {1000/avg_time:.0f} Hz')
    
    if avg_time < 1.0:
        print('✅ Performance: Excellent')
    elif avg_time < 5.0:
        print('✅ Performance: Good')
    else:
        print('⚠️  Performance: Slow (check GPU acceleration)')
        
except Exception as e:
    print(f'❌ Performance test failed: {e}')
"
```

## 🔄 Recovery Procedures

### Complete System Reset

If all else fails, use this procedure to reset the neural blending system:

```bash
# 1. Backup current configuration
cp -r models/ models_backup/ 2>/dev/null || echo "No models to backup"
cp plans/tournament_archive.json plans/tournament_archive.json.backup 2>/dev/null || echo "No archive to backup"

# 2. Clean installation
rm -rf models/
mkdir -p models/

# 3. Reinstall dependencies
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu>=1.17.1

# 4. Regenerate neural models
python generate_neural_blending_models.py

# 5. Validate system
python -c "
from controllers.neural_blended import Controller
controller = Controller()
print('✅ System reset complete and functional')
"
```

### Emergency Fallback Activation

To force the system to use Tournament #2 parameters when neural models fail:

```python
# Edit controllers/neural_blended.py
# Change line in __init__ method:
self.force_fallback = True  # Add this line to skip neural model loading
```

## 📞 Getting Help

### Log Collection for Support

```bash
# Generate comprehensive system report
python -c "
import sys
import os
import json
import traceback
from pathlib import Path

print('=== NEURAL BLENDING SYSTEM REPORT ===')
print(f'Python version: {sys.version}')
print(f'Working directory: {os.getcwd()}')
print(f'Python path: {sys.path[:3]}...')

try:
    import onnxruntime as ort
    print(f'ONNX Runtime: {ort.__version__}')
    print(f'Providers: {ort.get_available_providers()}')
except Exception as e:
    print(f'ONNX Runtime: ERROR - {e}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except Exception as e:
    print(f'PyTorch: ERROR - {e}')

models_dir = Path('models')
if models_dir.exists():
    models = list(models_dir.glob('blender_*.onnx'))
    print(f'Neural models: {len(models)} found')
    if models:
        sizes = [m.stat().st_size for m in models]
        print(f'Size range: {min(sizes)}-{max(sizes)} bytes')
else:
    print('Neural models: Directory not found')

try:
    from controllers.neural_blended import Controller
    controller = Controller()
    print('Controller: Initialization successful')
except Exception as e:
    print(f'Controller: Initialization failed')
    traceback.print_exc()

print('=== END REPORT ===')
" > neural_blending_report.txt

echo "Report saved to neural_blending_report.txt"
```

### Common Error Messages and Solutions

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `InvalidProtobuf` | Corrupted ONNX models | Regenerate models with `python generate_neural_blending_models.py` |
| `No module named onnxruntime` | Missing dependency | `pip install onnxruntime-gpu` |
| `CUDA out of memory` | GPU memory exhaustion | Use CPU mode or reduce batch sizes |
| `No neural models found` | Missing model files | Create models directory and generate models |
| `Tournament archive not found` | Missing fallback data | Run tournament optimization to create archive |

## 📈 Performance Optimization Tips

1. **Use GPU acceleration** when available for 3-5x performance improvement
2. **Monitor model file sizes** - healthy models are ~1,199 bytes each
3. **Validate tournament archive** regularly for optimal fallback parameters
4. **Test neural model loading** during system startup to catch issues early
5. **Keep fallback systems active** for robust operation when neural models fail

## 🔮 Future Improvements

- **Automated model validation** during startup
- **Performance-based model retraining** pipeline  
- **Real-time model health monitoring**
- **Automatic fallback quality assessment**
- **Integration with continuous optimization system**

---

**Last Updated:** Tournament #3 Neural Blending System Documentation  
**Version:** 1.0 - Comprehensive troubleshooting coverage  
**Status:** Production troubleshooting guide with complete diagnostic procedures