<div align="center">
<h1>comma Controls Challenge v2</h1>


<h3>
  <a href="https://comma.ai/leaderboard">Leaderboard</a>
  <span> · </span>
  <a href="https://comma.ai/jobs">comma.ai/jobs</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
  <span> · </span>
  <a href="https://x.com/comma_ai">X</a>
</h3>

</div>

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.


## Getting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual car and road states from [openpilot](https://github.com/commaai/openpilot) users.

```
# install required packages
# recommended python==3.11
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid 
```

There are some other scripts to help you get aggregate metrics: 
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## Advanced Features

### 🏆 Tournament Optimization System

This project includes an advanced multi-stage parameter optimization system that achieves **40.5% performance improvement** over baseline through progressive optimization techniques.

#### Tournament Stages

**Tournament #1 - Foundation Discovery**
```bash
# Grid search optimization with GPU acceleration
python optimization/blended_2pid_optimizer.py
```

**Tournament #2 - Production Optimization** ✅ **Recommended for Production**
```bash
# Evolutionary optimization with archive intelligence
python optimization/tournament_optimizer.py

# Use optimized controller (324.83 cost)
python tinyphysics.py --controller tournament_optimized --data_path ./data --num_segs 100
```

**Tournament #3 - Neural Enhancement** ⚠️ **Research Phase**
```bash
# Neural blending with learned weights
python tinyphysics.py --controller neural_blended --data_path ./data --num_segs 100

# Generate neural models
python generate_neural_blending_models.py
```

#### Performance Results

| Tournament | Method | Best Cost | Status | Use Case |
|------------|--------|-----------|--------|----------|
| **#1** | Grid Search | ~380+ | ✅ Complete | Research Baseline |
| **#2** | Evolutionary | **324.83** | ✅ **Production Ready** | **Deployment** |
| **#3** | Neural Blending | 566.33 avg, 289.89 median | ⚠️ Research | Experimental |

### 🧠 Neural Blending Controllers

Advanced controllers that combine traditional PID control with learned neural network weights for intelligent, velocity-specific optimization.

```python
# Neural blending with GPU acceleration
from controllers.neural_blended import Controller

controller = Controller()  # Loads 43 specialized neural models
output = controller.update(target_lataccel, current_lataccel, state, future_plan)
```

**Features:**
- ⚡ GPU-accelerated neural inference (CUDAExecutionProvider)
- 🧠 8-dimensional neural input features (velocity, acceleration, error, integrals)
- 🛡️ Robust fallback to Tournament #2 parameters
- 📊 43 specialized ONNX models for different driving scenarios

### ⚡ GPU Acceleration

The system provides **3-5x performance improvement** with GPU acceleration:

```bash
# Install GPU support
pip install onnxruntime-gpu

# Verify GPU availability
python -c "import onnxruntime as ort; print('CUDA available:', 'CUDAExecutionProvider' in ort.get_available_providers())"
```

**GPU Benefits:**
- Physics model inference acceleration
- Neural model GPU inference
- Batch evaluation optimization
- Reduced optimization time

### 📊 System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- Multi-core CPU

**Recommended for GPU Acceleration:**
- NVIDIA GPU with CUDA 11.8+
- 16GB RAM
- 8GB+ GPU memory

**Dependencies:**
```bash
pip install onnxruntime-gpu>=1.17.1  # GPU acceleration
pip install torch>=1.9.0            # Neural model training
pip install numpy>=1.20.0
```

## 📚 Documentation

### Complete System Documentation

- **[Architecture Overview](docs/architecture.md)** - Complete system architecture including Tournament #3 neural blending
- **[Tournament System Guide](docs/TOURNAMENT_SYSTEM_OVERVIEW.md)** - Comprehensive guide to all three tournament stages
- **[Neural Blending User Guide](docs/NEURAL_BLENDING_USER_GUIDE.md)** - Complete user guide with examples and troubleshooting
- **[GPU Acceleration Results](docs/GPU_ACCELERATION_RESULTS.md)** - Performance analysis and optimization guide
- **[Tournament #3 Status Report](TOURNAMENT_3_NEURAL_BLENDING_STATUS_REPORT.md)** - Current neural blending performance analysis

### Quick Start Guides

```bash
# Production deployment (recommended)
python tinyphysics.py --controller tournament_optimized --data_path ./data --num_segs 100

# Research/experimental neural blending
python tinyphysics.py --controller neural_blended --data_path ./data --num_segs 100

# Generate new neural models
python generate_neural_blending_models.py
```

### Performance Benchmarks

| Controller | Cost | Status | Description |
|------------|------|--------|-------------|
| [`simple_pid`](controllers/simple_pid.py) | 546.11 | ✅ Baseline | Single PID controller |
| [`tournament_optimized`](controllers/tournament_optimized.py) | **324.83** | ✅ **Production** | **40.5% improvement** |
| [`neural_blended`](controllers/neural_blended.py) | 566.33 avg, 289.89 median | ⚠️ Research | Neural enhancement |

### System Architecture

```
🏆 Tournament System Evolution
├── Tournament #1: Grid Search Foundation
│   ├── Parameter space exploration
│   ├── GPU-accelerated evaluation
│   └── Archive intelligence initialization
├── Tournament #2: Evolutionary Optimization ✅
│   ├── Production-ready optimization
│   ├── 40.5% performance improvement
│   └── Robust fallback architecture
└── Tournament #3: Neural Enhancement ⚠️
    ├── 43 specialized ONNX models
    ├── GPU-accelerated neural inference
    └── Dynamic blending weights
```

### Contributing

When contributing to the tournament system:

1. **Test with Tournament #2** first for production validation
2. **Use GPU acceleration** when available for faster optimization
3. **Update documentation** when adding new controllers or features
4. **Follow performance benchmarks** in the guides

For neural blending development:
1. **Generate fresh neural models** using the provided script
2. **Test fallback behavior** when neural models fail
3. **Monitor GPU memory usage** during training and inference
4. **Document training methodology** for reproducibility

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. Its inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`), and a steer input (`steer_action`), then it predicts the resultant lateral acceleration of the car.


## Controllers
Your controller should implement a new [controller](https://github.com/commaai/controls_challenge/tree/master/controllers). This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.


## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(actual\\_lat\\_accel - target\\_lat\\_accel)^2}{steps} * 100$

- `jerk_cost`: $\dfrac{\Sigma((actual\\_lat\\_accel\_t - actual\\_lat\\_accel\_{t-1}) / \Delta t)^2}{steps - 1} * 100$

It is important to minimize both costs. `total_cost`: $(lataccel\\_cost * 50) + jerk\\_cost$

## Submission
Run the following command, then submit `report.html` and your code to [this form](https://forms.gle/US88Hg7UR6bBuW3BA).

```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller pid
```

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma

Like this sort of stuff? You might want to work at comma!
[comma.ai/jobs](https://comma.ai/jobs)
