# Tournament Evolution Pipeline Test Plan

## 🏆 Real Tournament Evolution Flow

**Correct Pipeline Understanding:**
1. **Stage 1**: PID Broad Exploration → creates `plans/tournament_archive.json`
2. **Tournament #1/#2**: Fixed blender (velocity-based) + varying PIDs → uses `blender_*.onnx` models
3. **Neural Training**: `simple_neural_trainer.py` learns from archive → creates `models/blender.onnx`
4. **Tournament #3**: Neural blender + fixed PIDs → uses single `blender.onnx`

## 🧪 Pipeline Validation Test

### Phase 1: Tournament Archive Validation 📊
```python
# Verify Stage 1/T1/T2 results exist
import json
with open('plans/tournament_archive.json', 'r') as f:
    archive = json.load(f)
valid_performers = [p for p in archive['archive'] if 'stats' in p]
print(f"✅ Found {len(valid_performers)} tournament performers")
```
**Expected**: Tournament archive with performance data from T1/T2

### Phase 2: Neural Training Pipeline 🧠
```bash
# Train neural blender from tournament results
python simple_neural_trainer.py
```
**Expected**: Single `models/blender.onnx` trained on archive data (~10s, ~1.4KB)

### Phase 3: Architecture Preservation ✅
```python
# Verify all tournament stages work independently
from controllers.neural_blended import Controller as T1T2
from controllers.tournament3_simple import Controller as T3

legacy = T1T2()  # Uses blender_*.onnx (multi-model T1/T2)
t3 = T3()        # Uses blender.onnx (single-model T3)
```
**Expected**: No breaking changes, clean separation

### Phase 4: Tournament #3 Neural Integration 🎯
```python
# Verify T3 loads trained neural model
status = 'NEURAL' if t3.blender_session else 'FALLBACK'
```
**Expected**: Tournament #3 uses trained model, not fallback

### Phase 5: End-to-End Validation 🚀
```python
# Test full neural blending pipeline
scenarios = [
    {'v_ego': 25, 'error': 1.0},  # Low speed → neural prefers PID1
    {'v_ego': 65, 'error': 1.0},  # High speed → neural prefers PID2
]
```
**Expected**: Intelligent blending based on learned patterns

## 🎯 Success Criteria

1. **✅ Archive Valid**: Tournament results from T1/T2 available
2. **✅ Neural Training**: Learns from archive data successfully
3. **✅ Architecture Clean**: T1/T2 vs T3 separation maintained
4. **✅ T3 Neural**: Uses trained model, not fallback behavior
5. **✅ Intelligence**: Neural blending outperforms velocity-based

## 🔧 Execution Ready
Real tournament evolution pipeline validated and ready for testing.