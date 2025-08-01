# System Troubleshooting Guide

**Document Version:** 1.0  
**Date:** January 2025  
**Status:** Consolidated System-Wide Troubleshooting Documentation

## Quick Navigation

- **[System Health Check](#system-health-check)** - Rapid diagnostic workflow
- **[Neural Blending Issues](#neural-blending-issues)** - Tournament #3 neural model problems
- **[Pipeline Issues](#pipeline-issues)** - Multi-stage optimization problems
- **[Performance Problems](#performance-problems)** - Poor results and regression analysis
- **[GPU Acceleration Issues](#gpu-acceleration-issues)** - CUDA and hardware problems
- **[Time Step Problems](#time-step-problems)** - dt compatibility issues
- **[Controller Issues](#controller-issues)** - PID and blending problems
- **[Installation Issues](#installation-issues)** - Dependencies and setup problems
- **[Advanced Diagnostics](#advanced-diagnostics)** - In-depth system analysis

## System Health Check

### 🚀 Quick Diagnostic Script

Run this comprehensive system check first to identify the most likely issues:

```bash
# Complete system diagnostic (run from project root)
python -c "
import sys
sys.path.append('.')
import json
from pathlib import Path

print('🔍 COMMA.AI CONTROLS CHALLENGE - SYSTEM HEALTH CHECK')
print('=' * 60)

# 1. Check Python Environment
print('\\n📦 Python Environment Check:')
try:
    import numpy as np
    import onnxruntime as ort
    print(f'✅ Python: {sys.version.split()[0]}')
    print(f'✅ NumPy: {np.__version__}')
    print(f'✅ ONNX Runtime: {ort.__version__}')
    
    # Check GPU availability
    providers = ort.get_available_providers()
    gpu_available = 'CUDAExecutionProvider' in providers
    print(f'🔧 GPU Acceleration: {\"AVAILABLE\" if gpu_available else \"CPU ONLY\"}')
    print(f'📊 Providers: {providers}')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')

# 2. Check Core Files
print('\\n📁 Core File Structure:')
critical_files = [
    'tinyphysics.py',
    'controllers/neural_blended.py',
    'controllers/tournament_optimized.py',
    'controllers/shared_pid.py',
    'optimization/tournament_optimizer.py',
    'optimization/__init__.py'
]

for file_path in critical_files:
    if Path(file_path).exists():
        print(f'✅ {file_path}')
    else:
        print(f'❌ MISSING: {file_path}')

# 3. Check Tournament Archive
print('\\n🏆 Tournament Archive Check:')
archive_path = Path('plans/tournament_archive.json')
if archive_path.exists():
    try:
        with open(archive_path, 'r') as f:
            archive = json.load(f)
        
        valid_entries = [x for x in archive['archive'] if 'stats' in x and 'avg_total_cost' in x['stats']]
        if valid_entries:
            best = min(valid_entries, key=lambda x: x['stats']['avg_total_cost'])
            print(f'✅ Tournament archive loaded: {len(valid_entries)} valid entries')
            print(f'🏆 Best cost: {best[\"stats\"][\"avg_total_cost\"]:.2f}')
        else:
            print('⚠️ Archive loaded but no valid entries found')
    except Exception as e:
        print(f'❌ Archive parsing failed: {e}')
else:
    print('⚠️ Tournament archive not found')

# 4. Check Neural Models
print('\\n🧠 Neural Models Check:')
models_dir = Path('models')
if models_dir.exists():
    blender_models = list(models_dir.glob('blender_*.onnx'))
    print(f'📊 Found {len(blender_models)} neural models')
    
    if blender_models:
        # Check model sizes
        healthy_models = 0
        for model_path in blender_models:
            size = model_path.stat().st_size
            if 1000 <= size <= 2000:  # Healthy range: ~1,199 bytes
                healthy_models += 1
            elif size < 100:
                print(f'❌ Corrupted model: {model_path.name} ({size} bytes)')
        
        print(f'✅ Healthy models: {healthy_models}/{len(blender_models)}')
        
        # Test model loading
        if healthy_models > 0:
            try:
                test_model = blender_models[0]
                session = ort.InferenceSession(str(test_model))
                print(f'✅ Model loading test: SUCCESS')
            except Exception as e:
                print(f'❌ Model loading test: {e}')
    else:
        print('⚠️ No neural models found')
else:
    print('❌ models/ directory not found')

# 5. Check Time Step Consistency
print('\\n⏰ Time Step Consistency Check:')
try:
    from controllers.shared_pid import SpecializedPID
    pid = SpecializedPID(1.0, 1.0, 1.0, 'test')
    pid.update(1.0)  # First update
    pid.update(1.0)  # Second update
    
    # Should be 0.2 (2 updates * 1.0 error * 0.1 dt)
    if abs(pid.error_integral - 0.2) < 0.001:
        print('✅ Time step scaling: dt = 0.1 verified')
    else:
        print(f'❌ Time step issue: integral = {pid.error_integral}, expected = 0.2')
except Exception as e:
    print(f'❌ Time step check failed: {e}')

print('\\n' + '=' * 60)
print('🏁 HEALTH CHECK COMPLETE')
print('Refer to specific sections below for detailed troubleshooting')
"
```

### Health Check Interpretation

| Status | Meaning | Next Steps |
|--------|---------|------------|
| **All ✅** | System healthy | Proceed with normal operations |
| **⚠️ Warnings** | Minor issues | Check specific sections below |
| **❌ Errors** | Critical problems | Follow troubleshooting steps immediately |

## Neural Blending Issues

### Issue 1: Corrupted Neural Models

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

**Root Cause:** Neural model generation script failed during training, creating placeholder files instead of valid ONNX models.

**Solution:**
```bash
# Step 1: Remove corrupted models
rm -f models/blender_*.onnx

# Step 2: Regenerate all neural models
python generate_neural_blending_models.py

# Step 3: Verify models are properly generated
python -c "
import onnxruntime as ort
from pathlib import Path

models_dir = Path('models')
blender_models = list(models_dir.glob('blender_*.onnx'))
print(f'Generated {len(blender_models)} models')

for model_path in blender_models[:3]:  # Test first 3
    try:
        session = ort.InferenceSession(str(model_path))
        print(f'✅ {model_path.name}: Valid ({model_path.stat().st_size} bytes)')
    except Exception as e:
        print(f'❌ {model_path.name}: Invalid - {e}')
"
```

### Issue 2: Missing Neural Models

**Symptoms:**
- `models/` directory doesn't exist
- No `blender_*.onnx` files found
- Controller immediately uses fallback parameters
- Warning: "No neural models found, using fallback"

**Solution:**
```bash
# Create models directory if missing
mkdir -p models

# Generate neural models
python generate_neural_blending_models.py

# Verify 43 models were created
ls models/blender_*.onnx | wc -l  # Should output: 43
```

### Issue 3: Neural Model Performance Regression

**Symptoms:**
- Neural models load correctly but perform worse than fallback
- High variance with extreme outliers (std dev > 500)
- Good median performance but poor average
- 56% success rate but dragged down by extreme failures

**Root Cause Analysis:**
```python
# Analyze performance distribution
import numpy as np
import json

# Load Tournament #3 results
with open('tournament3_results.json', 'r') as f:
    results = json.load(f)

costs = [r['cost'] for r in results['individual_results']]
costs_array = np.array(costs)

print(f"Average: {costs_array.mean():.2f}")
print(f"Median: {np.median(costs_array):.2f}")
print(f"Std Dev: {costs_array.std():.2f}")

# Identify outliers
q1, q3 = np.percentile(costs_array, [25, 75])
iqr = q3 - q1
outliers = costs_array[(costs_array < q1 - 1.5*iqr) | (costs_array > q3 + 1.5*iqr)]

print(f"Outliers: {len(outliers)}/{len(costs_array)} ({len(outliers)/len(costs_array)*100:.1f}%)")
print(f"Outlier range: [{outliers.min():.2f}, {outliers.max():.2f}]")
```

**Solutions:**
1. **Training Data Enhancement**: Replace synthetic data with performance-optimized real scenarios
2. **Cost-Minimization Training**: Implement performance-focused loss functions
3. **Outlier Management**: Enhanced edge case handling in training pipeline
4. **Statistical Validation**: Larger dataset validation for robust performance

## Pipeline Issues

### Issue 1: Time Step Incompatibility

**Symptoms:**
- Poor performance results despite good optimization
- Parameters don't work with `eval.py`
- Performance degradation between pipeline stages
- Error messages about dt scaling

**Diagnosis:**
```bash
# Run validation script
python validate_timestep_fix.py
```

**Expected Output:**
```
🔧 Testing controller template fix...
✅ Controller template fix VERIFIED - dt = 0.1 scaling found

🔧 Testing tournament controller fix...
✅ Tournament controller fix VERIFIED - loads without errors
```

**Root Cause:** PID implementations using different time steps across pipeline stages.

**Solution:**
```python
# Correct PID implementation (should be everywhere)
def update(self, error):
    dt = 0.1  # Match tinyphysics DEL_T = 0.1 (10 Hz)
    self.error_integral += error * dt         # WITH dt scaling
    error_diff = (error - self.prev_error) / dt  # WITH dt scaling
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff
```

### Issue 2: Multi-Format Loading Problems

**Symptoms:**
- Tournament stages can't load previous results
- Format mismatch errors between Stage 1 and Tournament formats
- Missing archive keys or malformed JSON

**Diagnosis:**
```python
# Test format detection
from optimization.tournament_optimizer import load_champions_from_file

try:
    # Test Stage 1 format
    champions = load_champions_from_file('plans/blended_2pid_comprehensive_results.json', 20)
    print(f'✅ Stage 1 format: {len(champions)} champions loaded')
except Exception as e:
    print(f'❌ Stage 1 format error: {e}')

try:
    # Test Tournament format
    champions = load_champions_from_file('plans/tournament_archive.json', 20)
    print(f'✅ Tournament format: {len(champions)} champions loaded')
except Exception as e:
    print(f'❌ Tournament format error: {e}')
```

**Solution:** The multi-format support should handle this automatically. If issues persist:
```bash
# Verify file integrity
python -c "
import json
from pathlib import Path

for file_path in ['plans/blended_2pid_comprehensive_results.json', 'plans/tournament_archive.json']:
    if Path(file_path).exists():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f'✅ {file_path}: Valid JSON')
        except Exception as e:
            print(f'❌ {file_path}: {e}')
    else:
        print(f'⚠️ {file_path}: File not found')
"
```

### Issue 3: Pipeline Stage Failures

**Symptoms:**
- Stage completes but produces no output files
- Memory errors during large-scale optimization
- GPU context issues during long runs

**Solutions:**
1. **Memory Management**: Implement batch processing and cleanup
2. **GPU Resource Management**: Single session pattern, cleanup between stages
3. **Checkpoint System**: Save intermediate results to prevent data loss

## Performance Problems

### Issue 1: Performance Regression After Optimization

**Symptoms:**
- Optimization shows good results but validation performs poorly
- High variance in performance across different files
- Parameters work well in optimization but fail in eval.py

**Root Cause Analysis:**
```python
# Compare optimization vs validation results
def analyze_performance_gap(optimization_file, validation_file):
    import json
    
    # Load results
    with open(optimization_file, 'r') as f:
        opt_data = json.load(f)
    with open(validation_file, 'r') as f:
        val_data = json.load(f)
    
    opt_cost = opt_data.get('best_cost', opt_data.get('avg_cost'))
    val_cost = val_data.get('avg_cost', val_data.get('cost'))
    
    gap = val_cost - opt_cost
    gap_percent = (gap / opt_cost) * 100
    
    print(f"Optimization cost: {opt_cost:.2f}")
    print(f"Validation cost: {val_cost:.2f}")
    print(f"Performance gap: {gap:.2f} ({gap_percent:.1f}%)")
    
    if gap_percent > 20:
        print("❌ SIGNIFICANT PERFORMANCE GAP - Check time step consistency")
    elif gap_percent > 10:
        print("⚠️ Moderate performance gap - Normal for different datasets")
    else:
        print("✅ Acceptable performance gap")

# Example usage
analyze_performance_gap('tournament2_results.json', 'tournament2_validation.json')
```

**Solutions:**
1. **Time Step Consistency**: Ensure dt = 0.1 throughout entire pipeline
2. **Dataset Consistency**: Use same evaluation methodology as optimization
3. **Parameter Validation**: Test optimized parameters on larger dataset

### Issue 2: Inconsistent Performance Across Files

**Symptoms:**
- Some files perform excellently, others poorly
- High standard deviation (>500)
- Median much better than average

**Analysis Framework:**
```python
def analyze_performance_variance(results_file):
    import json
    import numpy as np
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    costs = np.array([r['cost'] for r in data['individual_results']])
    
    # Statistical analysis
    stats = {
        'mean': costs.mean(),
        'median': np.median(costs),
        'std': costs.std(),
        'min': costs.min(),
        'max': costs.max(),
        'q25': np.percentile(costs, 25),
        'q75': np.percentile(costs, 75)
    }
    
    # Outlier detection
    iqr = stats['q75'] - stats['q25']
    outlier_threshold_low = stats['q25'] - 1.5 * iqr
    outlier_threshold_high = stats['q75'] + 1.5 * iqr
    
    outliers = costs[(costs < outlier_threshold_low) | (costs > outlier_threshold_high)]
    outlier_rate = len(outliers) / len(costs)
    
    print(f"Performance Statistics:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std Dev: {stats['std']:.2f}")
    print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"  Outlier Rate: {outlier_rate*100:.1f}%")
    
    if outlier_rate > 0.2:
        print("❌ HIGH VARIANCE - Check training data quality")
    elif outlier_rate > 0.1:
        print("⚠️ Moderate variance - Consider outlier handling")
    else:
        print("✅ Acceptable variance")
    
    return stats

# Example usage
analyze_performance_variance('tournament3_results.json')
```

**Solutions:**
1. **Outlier Investigation**: Identify common characteristics of poor-performing files
2. **Training Data Balancing**: Ensure training covers diverse scenario types
3. **Model Ensemble**: Use multiple models for robust predictions
4. **Fallback Logic**: Implement intelligent fallback for detected edge cases

## GPU Acceleration Issues

### Issue 1: CUDA Not Available

**Symptoms:**
- Slow optimization performance (3-4 hours vs 44 minutes)
- CPU-only execution despite CUDA availability
- "CUDAExecutionProvider not available" warnings

**Diagnosis:**
```python
import onnxruntime as ort

# Check CUDA availability
providers = ort.get_available_providers()
print(f"Available providers: {providers}")

if 'CUDAExecutionProvider' not in providers:
    print("❌ CUDA not available")
    
    # Check system CUDA
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("✅ NVIDIA drivers installed")
        print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
    except FileNotFoundError:
        print("❌ NVIDIA drivers not found")
else:
    print("✅ CUDA available")
```

**Solutions:**
1. **Install GPU ONNX Runtime**: 
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu>=1.17.1
   ```

2. **Verify CUDA Version Compatibility**:
   ```bash
   # Check CUDA version
   nvcc --version
   
   # Required: CUDA 11.8 and cuDNN 8.9.7.29
   ```

3. **Provider Priority Fix**:
   ```python
   # Ensure CUDAExecutionProvider comes first
   session_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
   session = ort.InferenceSession(model_path, providers=session_providers)
   ```

### Issue 2: GPU Memory Issues

**Symptoms:**
- `CUDA out of memory` errors
- System crashes during neural inference
- Slow performance despite GPU availability

**Solutions:**
```python
# GPU memory management pattern
class GPUOptimizedOptimizer:
    def __init__(self, model_path):
        # Single session with memory management
        session_options = ort.SessionOptions()
        session_options.enable_mem_pattern = False  # Reduce memory fragmentation
        session_options.enable_cpu_mem_arena = False
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, session_options, providers)
    
    def __del__(self):
        # Explicit cleanup
        if hasattr(self, 'session'):
            del self.session
```

## Time Step Problems

### Issue 1: dt Scaling Inconsistency

**Symptoms:**
- Parameters work in optimization but fail in eval.py
- Performance degradation between pipeline stages
- Integral values don't accumulate correctly

**Quick Test:**
```python
# Test dt scaling behavior
from controllers.shared_pid import SpecializedPID

pid = SpecializedPID(1.0, 1.0, 1.0, 'test')
pid.update(1.0)  # First update
pid.update(1.0)  # Second update

expected_integral = 0.2  # 2 updates * 1.0 error * 0.1 dt
if abs(pid.error_integral - expected_integral) < 0.001:
    print('✅ Time step scaling: CORRECT')
else:
    print(f'❌ Time step scaling: integral = {pid.error_integral}, expected = {expected_integral}')
```

**Solution:**
Ensure all PID implementations use this pattern:
```python
def update(self, error):
    dt = 0.1  # Match tinyphysics DEL_T = 0.1 (10 Hz)
    self.error_integral += error * dt
    error_diff = (error - self.prev_error) / dt
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff
```

## Controller Issues

### Issue 1: Import Errors

**Symptoms:**
- `ModuleNotFoundError` for controllers
- Import path issues with relative imports
- Controller not found during optimization

**Solutions:**
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to script
import sys
sys.path.append('.')
```

### Issue 2: Controller Initialization Failures

**Symptoms:**
- Controller object creation fails
- Attribute errors during initialization
- Missing tournament archive data

**Diagnosis:**
```python
# Test controller initialization
try:
    from controllers.neural_blended import Controller
    controller = Controller()
    print(f'✅ Controller initialized: {controller}')
    
    # Test basic functionality
    from collections import namedtuple
    State = namedtuple('State', ['v_ego', 'roll_lataccel', 'a_ego'])
    FuturePlan = namedtuple('FuturePlan', ['lataccel'])
    
    state = State(v_ego=30, roll_lataccel=0.1, a_ego=0.0)
    future_plan = FuturePlan(lataccel=[0.0, 0.5, 1.0])
    
    output = controller.update(1.0, 0.5, state, future_plan)
    print(f'✅ Controller update: {output:.3f}')
    
except Exception as e:
    print(f'❌ Controller error: {e}')
    import traceback
    traceback.print_exc()
```

## Installation Issues

### Issue 1: Missing Dependencies

**Symptoms:**
- Import errors for numpy, onnxruntime, etc.
- Version compatibility issues
- Package conflicts

**Solution:**
```bash
# Install core dependencies
pip install numpy>=1.21.0
pip install onnxruntime-gpu>=1.17.1

# For development
pip install scipy matplotlib

# Verify installation
python -c "
import numpy as np
import onnxruntime as ort
print(f'NumPy: {np.__version__}')
print(f'ONNX Runtime: {ort.__version__}')
print(f'Providers: {ort.get_available_providers()}')
"
```

### Issue 2: File Structure Problems

**Symptoms:**
- Missing critical files
- Incorrect directory structure
- Permission issues

**Verification:**
```bash
# Check essential files
for file in tinyphysics.py controllers/neural_blended.py optimization/__init__.py; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ MISSING: $file"
    fi
done

# Check permissions
ls -la controllers/
ls -la optimization/
ls -la models/
```

## Advanced Diagnostics

### Performance Profiling

```python
# Comprehensive performance analysis
import time
import json
from pathlib import Path

def profile_system_performance():
    """Profile all system components"""
    results = {}
    
    # 1. Model Loading Performance
    start_time = time.time()
    try:
        import onnxruntime as ort
        model_path = Path('models').glob('blender_*.onnx')
        session = ort.InferenceSession(str(next(model_path)))
        results['model_loading'] = time.time() - start_time
        print(f"✅ Model loading: {results['model_loading']:.3f}s")
    except Exception as e:
        results['model_loading'] = None
        print(f"❌ Model loading failed: {e}")
    
    # 2. Controller Initialization
    start_time = time.time()
    try:
        from controllers.neural_blended import Controller
        controller = Controller()
        results['controller_init'] = time.time() - start_time
        print(f"✅ Controller init: {results['controller_init']:.3f}s")
    except Exception as e:
        results['controller_init'] = None
        print(f"❌ Controller init failed: {e}")
    
    # 3. Tournament Archive Loading
    start_time = time.time()
    try:
        from optimization.tournament_optimizer import load_champions_from_file
        champions = load_champions_from_file('plans/tournament_archive.json', 10)
        results['archive_loading'] = time.time() - start_time
        print(f"✅ Archive loading: {results['archive_loading']:.3f}s")
    except Exception as e:
        results['archive_loading'] = None
        print(f"❌ Archive loading failed: {e}")
    
    return results

# Run profiling
profile_results = profile_system_performance()
```

### Memory Usage Analysis

```python
# Monitor memory usage during operations
import psutil
import os

def monitor_memory_usage():
    """Monitor system memory during operations"""
    process = psutil.Process(os.getpid())
    
    print(f"Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    # Test memory usage during model loading
    import onnxruntime as ort
    from pathlib import Path
    
    models = list(Path('models').glob('blender_*.onnx'))
    sessions = []
    
    for i, model_path in enumerate(models[:5]):  # Test first 5 models
        session = ort.InferenceSession(str(model_path))
        sessions.append(session)
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"After model {i+1}: {memory_mb:.1f} MB")
    
    # Cleanup
    del sessions
    import gc
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"After cleanup: {final_memory:.1f} MB")

# Run memory monitoring
monitor_memory_usage()
```

### Log Analysis Framework

```python
# Analyze system logs for patterns
import re
from pathlib import Path

def analyze_system_logs():
    """Analyze log files for common error patterns"""
    
    error_patterns = {
        'cuda_errors': r'CUDA.*error|out of memory',
        'model_errors': r'InvalidProtobuf|ONNX.*error',
        'import_errors': r'ModuleNotFoundError|ImportError',
        'time_step_errors': r'dt.*scaling|time.*step',
        'performance_issues': r'regression|outlier|variance'
    }
    
    # Check recent Python output (if available)
    log_sources = [
        'optimization_log.txt',
        'tournament_results.log',
        'error.log'
    ]
    
    for log_file in log_sources:
        if Path(log_file).exists():
            print(f"\\n📊 Analyzing {log_file}:")
            with open(log_file, 'r') as f:
                content = f.read()
            
            for pattern_name, pattern in error_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    print(f"  ⚠️ {pattern_name}: {len(matches)} occurrences")
                    print(f"     Sample: {matches[0][:100]}...")
        else:
            print(f"⚠️ {log_file} not found")

# Run log analysis
analyze_system_logs()
```

## Emergency Recovery Procedures

### Complete System Reset

If multiple issues persist, use this nuclear option:

```bash
#!/bin/bash
echo "🚨 EMERGENCY SYSTEM RESET"
echo "This will reset all generated files and rebuild the system"
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 1. Clean generated files
    echo "🧹 Cleaning generated files..."
    rm -rf models/blender_*.onnx
    rm -f tournament*_results.json
    rm -f plans/tournament_archive.json
    
    # 2. Reinstall dependencies
    echo "📦 Reinstalling dependencies..."
    pip uninstall -y onnxruntime onnxruntime-gpu
    pip install onnxruntime-gpu>=1.17.1
    
    # 3. Regenerate neural models
    echo "🧠 Regenerating neural models..."
    python generate_neural_blending_models.py
    
    # 4. Run system health check
    echo "🔍 Running health check..."
    python -c "exec(open('docs/TROUBLESHOOTING_GUIDE.md').read().split('```bash')[1].split('```')[0])"
    
    echo "✅ Emergency reset complete"
else
    echo "❌ Reset cancelled"
fi
```

## Getting Help

### Information to Gather Before Reporting Issues

1. **System Health Check Output**: Run the diagnostic script at the top
2. **Error Messages**: Full error traceback, not just the final message
3. **System Information**: OS, Python version, GPU details
4. **Recent Changes**: What was modified before the issue appeared
5. **Reproduction Steps**: Minimal steps to reproduce the problem

### Diagnostic Data Collection

```bash
# Generate comprehensive diagnostic report
python -c "
import sys, platform, json
from pathlib import Path
import onnxruntime as ort

report = {
    'system': {
        'platform': platform.platform(),
        'python': sys.version,
        'onnxruntime': ort.__version__,
        'providers': ort.get_available_providers()
    },
    'files': {
        'controllers': list(str(p) for p in Path('controllers').glob('*.py')),
        'models': len(list(Path('models').glob('*.onnx'))) if Path('models').exists() else 0,
        'plans': list(str(p) for p in Path('plans').glob('*.json'))
    }
}

with open('diagnostic_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('✅ Diagnostic report saved to diagnostic_report.json')
"
```

### Quick Reference

| Issue Type | First Try | If That Fails |
|-----------|-----------|---------------|
| **Neural Models** | Regenerate models | Check ONNX Runtime installation |
| **Performance** | Check time steps | Analyze dataset consistency |
| **GPU Issues** | Install onnxruntime-gpu | Check CUDA compatibility |
| **Pipeline Breaks** | Run validation script | Check file permissions |
| **Import Errors** | Fix PYTHONPATH | Verify file structure |

## Conclusion

This troubleshooting guide consolidates diagnostic procedures from across the entire system. Start with the System Health Check, then navigate to specific sections based on your symptoms.

**Key Principles:**
1. **Start with the health check** - Most issues are caught by comprehensive diagnostics
2. **Check time step consistency** - This is the most common source of problems
3. **Verify file integrity** - Corrupted models and missing files cause many issues
4. **Use fallback systems** - The system is designed to gracefully degrade
5. **Monitor resource usage** - GPU memory and system resources can bottleneck performance

The system is designed to be robust and self-recovering. Most issues can be resolved by following the systematic diagnostic procedures outlined above.