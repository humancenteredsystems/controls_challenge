# Tournament #3 Simplification Plan

## Problem Analysis

**Current Tournament #3 is overcomplicated:**
- 43 neural ONNX models (1,199 bytes each)
- Complex 8-dimensional feature extraction
- 1000+ lines of optimization infrastructure
- Multiple controller approaches with conflicting costs
- Trying to re-optimize PIDs that Tournament #2 already perfected

## Clean Solution Architecture

### Tournament #2 Winner (Static Base)
From [`plans/tournament_archive.json`](plans/tournament_archive.json:179-199):
- **Low-speed PID**: P=0.374, I=0.01, D=-0.05  
- **High-speed PID**: P=0.4, I=0.05, D=-0.053
- **Performance**: 324.83 avg cost (40.5% improvement)
- **Status**: These parameters are PROVEN and should remain STATIC

### Simple Tournament #3 Controller

```python
class SimpleTournament3Controller(BaseController):
    def __init__(self):
        # STATIC Tournament #2 winner PIDs (DO NOT OPTIMIZE)
        self.tournament2_low = SpecializedPID(0.374, 0.01, -0.05, "Tournament2_Low")
        self.tournament2_high = SpecializedPID(0.4, 0.05, -0.053, "Tournament2_High")
        
        # ONLY optimization target: single neural blending weight
        self.blend_network = load_simple_blend_network()  # 8→16→1 architecture
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        
        # Get Tournament #2 outputs (proven controllers)
        low_output = self.tournament2_low.update(error)
        high_output = self.tournament2_high.update(error)
        
        # ONLY neural component: predict blend weight [0,1]
        features = extract_simple_features(state, error, future_plan)  # 8 features
        blend_weight = self.blend_network.predict(features)  # Single output
        
        # Simple weighted blend
        return blend_weight * low_output + (1 - blend_weight) * high_output
```

## Implementation Steps

### 1. Create Clean Tournament #3 Controller
- **File**: [`controllers/tournament3_simple.py`](controllers/tournament3_simple.py)
- **Lines**: ~80 lines (vs current 197)
- **Static PIDs**: Use Tournament #2 winner parameters exactly
- **Neural component**: Single 8→16→1 network for blend weight only

### 2. Create Quick Evaluation Script
- **File**: [`quick_eval.py`](quick_eval.py)
- **Purpose**: Fast testing before full [`eval.py`](eval.py) runs
- **Features**: 
  - Test 5-10 data files quickly
  - Compare Tournament #3 vs Tournament #2 baseline
  - Validate basic functionality

### 3. Training Data Generation
- **Focus**: Learn optimal blend weights, not PID parameters
- **Training objective**: For each driving scenario, what blend weight minimizes cost?
- **Simple approach**: 
  ```python
  for scenario in training_data:
      # Try different blend weights [0.0, 0.1, 0.2, ..., 1.0]
      # Find weight that minimizes actual cost
      # Train network: features → optimal_weight
  ```

### 4. Archive Complex Infrastructure
- **Archive**: Move complex files to `archive/tournament3_complex/`
- **Files to archive**:
  - [`controllers/neural_blended.py`](controllers/neural_blended.py) (197 lines)
  - [`optimization/blender_tournament_optimizer.py`](optimization/blender_tournament_optimizer.py) (457 lines)
  - [`optimization/simple_blender_optimizer.py`](optimization/simple_blender_optimizer.py) (460 lines)
  - 43 `blender_*.onnx` models

## Expected Results

### Performance Targets
- **Minimum performance**: Never worse than Tournament #2 (324.83 cost)
- **Realistic improvement**: 5-15% better through smart blending
- **Target range**: 275-310 cost

### Implementation Benefits
- **Simplicity**: ~100 lines vs 1000+ lines of infrastructure
- **Clarity**: Single optimization target (blend weight)
- **Reliability**: Built on proven Tournament #2 foundation
- **Speed**: Hours to train vs days of complex optimization

## Files to Create

1. **[`controllers/tournament3_simple.py`](controllers/tournament3_simple.py)** - Clean implementation
2. **[`quick_eval.py`](quick_eval.py)** - Fast testing script
3. **[`train_tournament3.py`](train_tournament3.py)** - Simple training script
4. **[`TOURNAMENT3_RESULTS.md`](TOURNAMENT3_RESULTS.md)** - Performance documentation

## Files to Archive

1. **Complex Controllers**:
   - [`controllers/neural_blended.py`](controllers/neural_blended.py)
   - [`controllers/optimized_blender.py`](controllers/optimized_blender.py)
   - [`controllers/optimized_ensemble.py`](controllers/optimized_ensemble.py)

2. **Complex Optimization**:
   - [`optimization/blender_tournament_optimizer.py`](optimization/blender_tournament_optimizer.py)
   - [`optimization/simple_blender_optimizer.py`](optimization/simple_blender_optimizer.py)

3. **Neural Models**:
   - All 43 `models/blender_*.onnx` files

## Next Steps

1. Switch to Code mode for implementation
2. Create simple Tournament #3 controller
3. Create quick_eval.py for testing
4. Test and validate approach
5. Document results and improvements