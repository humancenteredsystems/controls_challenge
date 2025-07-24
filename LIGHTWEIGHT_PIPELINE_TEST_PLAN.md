# Lightweight Pipeline Test Plan

## 🎯 **Objective**
Validate that the clean PID refactor works correctly across the entire pipeline and produces reasonable, progressively improving results without spending hours on full optimization.

## 📋 **Test Strategy: Fast & Comprehensive**

### **Stage 1: Mini 2-PID Parameter Search**
```bash
# Lightweight Stage 1 Test
python -m optimization.blended_2pid_optimizer --test-mode
```

**Parameters:**
- **Grid Size**: 10-15 combinations (vs 250-1000 full)  
- **Test Files**: 2-3 files (vs 20-50 full)
- **Expected Time**: ~2-3 minutes
- **Expected Cost**: 80-120 range (limited search space)
- **Success Criteria**: ✅ No crashes, generates JSON results

### **Tournament #1: Mini Discovery Phase**
```bash
# Lightweight Tournament #1 Test  
python -m optimization.tournament_optimizer \
    --mode tournament \
    --rounds 3 \
    --population 5 \
    --max_files 2 \
    --seed_from_file blended_2pid_comprehensive_results.json
```

**Parameters:**
- **Population**: 5 individuals (vs 20 full)
- **Rounds**: 3 generations (vs 10 full)
- **Test Files**: 2 files (vs 30 full)
- **Expected Time**: ~3-4 minutes
- **Expected Cost**: Should **IMPROVE** on Stage 1 (not degrade!)
- **Success Criteria**: ✅ Shows improvement, not 3.3x degradation

### **Tournament #2: Mini Refinement Phase**
```bash
# Lightweight Tournament #2 Test
python -m optimization.tournament_optimizer \
    --mode tournament \
    --rounds 3 \
    --population 5 \
    --max_files 3 \
    --seed_from_archive plans/tournament_archive.json
```

**Parameters:**
- **Population**: 5 individuals 
- **Rounds**: 3 generations
- **Test Files**: 3 files (expanded validation)
- **Expected Time**: ~4-5 minutes  
- **Expected Cost**: Further improvement from Tournament #1
- **Success Criteria**: ✅ Progressive improvement continues

### **Tournament #3: Skip for Lightweight Test**
**Rationale**: Neural blending requires trained models and is complex. Focus on core PID pipeline first.

### **eval.py Validation Test**  
```bash
# Test final result with eval.py
python eval.py --controller controllers/tournament_optimized.py
```

**Parameters:**
- **Test**: Single file evaluation
- **Expected Time**: ~30 seconds
- **Expected**: No crashes, reasonable performance
- **Success Criteria**: ✅ Compatible with eval.py, no dt mismatch errors

## 📊 **Success Criteria**

### **Primary Success Indicators:**
1. ✅ **No Pipeline Breaks**: Each stage completes without errors
2. ✅ **Progressive Improvement**: `Stage 1 ≥ Tournament #1 ≥ Tournament #2`
3. ✅ **No Degradation**: Avoid 76.81 → 256.79 disaster (3.3x worse)
4. ✅ **eval.py Compatibility**: Final controller works with official evaluation

### **Performance Expectations:**
```
Stage 1 (mini):     ~80-120 cost  (limited search)
Tournament #1:      ~70-100 cost  (should improve) ✅ 
Tournament #2:      ~60-90 cost   (further improvement) ✅
eval.py test:       Runs without crashes ✅
```

### **Time Budget:**
- **Total Test Time**: ~10-15 minutes
- **vs Full Pipeline**: 4-12 hours
- **Speed-up**: ~20-40x faster for validation

## 🔍 **What We Monitor**

### **Technical Validation:**
- ✅ **Import Consistency**: All stages use shared PID correctly
- ✅ **Format Compatibility**: Stage 1 → Tournament data flow works
- ✅ **dt = 0.1 Usage**: No time step mismatches anywhere
- ✅ **GPU Acceleration**: CUDA still working after refactor

### **Performance Validation:**
- 📈 **Cost Progression**: Each stage improves previous results
- 🚫 **No Regressions**: No massive performance drops
- ⚡ **Reasonable Results**: Costs in expected ranges
- 🎯 **Functional Pipeline**: End-to-end execution success

## 🚨 **Failure Indicators**

### **Critical Failures** (Stop immediately):
- ❌ Import errors with shared PID
- ❌ Tournament can't load Stage 1 results
- ❌ Performance degradation between stages
- ❌ eval.py compatibility issues

### **Warning Signs** (Investigate):
- ⚠️ Costs outside expected ranges
- ⚠️ Very slow convergence in tournaments  
- ⚠️ GPU acceleration not working
- ⚠️ Unusual parameter values

## 🎯 **Expected Outcome**

### **If Test Passes:**
- ✅ Clean refactor is validated and working
- ✅ Ready for full pipeline re-run
- ✅ Confidence in progressive improvement
- ✅ Architecture debt successfully eliminated

### **If Test Fails:**
- 🔧 Quick debugging of specific failure point
- 🔍 Targeted fixes rather than full pipeline re-work
- 📊 Clear understanding of what needs attention

## 📝 **Execution Plan**

### **Phase 1: Preparation** (2 min)
1. Backup existing results
2. Clear any temp files
3. Verify environment ready

### **Phase 2: Sequential Testing** (10-12 min)
1. Run Stage 1 mini test → Check results
2. Run Tournament #1 mini test → Verify improvement  
3. Run Tournament #2 mini test → Verify further improvement
4. Run eval.py validation → Verify compatibility

### **Phase 3: Results Analysis** (2-3 min)
1. Compare cost progression
2. Validate no regressions
3. Confirm technical success criteria

**Total Time**: ~15 minutes for comprehensive pipeline validation

This approach gives us high confidence in the refactor while keeping the test fast and focused on key success indicators.