# Cleaner Pipeline Solution: Multi-Format Support

## The Problem
Stage 1 outputs `blended_2pid_comprehensive_results.json` but Tournament #1 expects `tournament_archive.json` format.

## Current Band-Aid Solution ❌
Create external converter script to transform formats.

## Cleaner Solution ✅
Modify [`tournament_optimizer.py`](optimization/tournament_optimizer.py) to **natively support both formats**.

## Implementation

### Current Code (Lines 37-49)
```python
# Extract archive list from the JSON structure
archive_list = archive_data.get('archive', [])

# Sort by avg_cost, take top half of population as champions
champions = sorted(
    [ps for ps in archive_list if ps.get('stats', {}).get('avg_total_cost') != float('inf')],
    key=lambda x: x.get('stats', {}).get('avg_total_cost', float('inf'))
)[:n//2]
```

### Enhanced Code (Multi-Format Support)
```python
def load_champions_from_file(file_path: str, n: int) -> List[Dict]:
    """Load champions from either Stage 1 or Tournament archive format"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Detect format and extract champions
    if 'all_results' in data:
        # Stage 1 format: blended_2pid_comprehensive_results.json
        print(f"📊 Detected Stage 1 format: {file_path}")
        candidates = data['all_results']
        # Sort by avg_total_cost (direct field)
        champions = sorted(
            [c for c in candidates if c.get('avg_total_cost', float('inf')) != float('inf')],
            key=lambda x: x.get('avg_total_cost', float('inf'))
        )[:n//2]
        
    elif 'archive' in data:
        # Tournament format: tournament_archive.json  
        print(f"🏆 Detected Tournament format: {file_path}")
        candidates = data['archive']
        # Sort by stats.avg_total_cost (nested field)
        champions = sorted(
            [c for c in candidates if c.get('stats', {}).get('avg_total_cost', float('inf')) != float('inf')],
            key=lambda x: x.get('stats', {}).get('avg_total_cost', float('inf'))
        )[:n//2]
        
    else:
        raise ValueError(f"Unknown format in {file_path}. Expected 'all_results' or 'archive' key.")
    
    return champions

def extract_gains_from_champion(champion: Dict) -> Tuple[List[float], List[float]]:
    """Extract low_gains and high_gains from either format"""
    return champion['low_gains'], champion['high_gains']
```

### Modified initialize_population()
```python
def initialize_population(n: int, seed_from_archive: Optional[str] = None) -> List[ParameterSet]:
    """Generate initial population, optionally seeding best performers from archive."""
    population: List[ParameterSet] = []
    
    # Seed with champions if provided (supports both Stage 1 and Tournament formats)
    if seed_from_archive and Path(seed_from_archive).exists():
        try:
            champions = load_champions_from_file(seed_from_archive, n)
            print(f"✅ Seeding {len(champions)} champions from {seed_from_archive}")
            
            for i, champ in enumerate(champions):
                low_gains, high_gains = extract_gains_from_champion(champ)
                ps = ParameterSet(low_gains, high_gains)
                ps.id = f"champion_{i}"
                ps.rounds_survived = 0  # Reset for new tournament
                population.append(ps)
                
        except Exception as e:
            print(f"❌ Failed to load archive {seed_from_archive}: {e}")
            print("🔄 Falling back to random initialization")
    
    # Fill remaining slots with random generation
    # ... rest stays the same
```

## Benefits

### ✅ Clean Architecture
- **Single responsibility**: Tournament optimizer handles format detection
- **No external dependencies**: No converter scripts to maintain
- **Backward compatible**: Still works with existing tournament archives

### ✅ Direct Pipeline Flow
```bash
# Stage 1: Broad Parameter Search → blended_2pid_comprehensive_results.json
python optimization/blended_2pid_optimizer.py --output_file stage1_results.json

# Tournament #1: Direct consumption of Stage 1 results  
python optimization/tournament_optimizer.py \
  --archive_file stage1_results.json \
  --output_file tournament1_results.json

# No converter script needed! 🎉
```

### ✅ Format Detection
- **Auto-detects** Stage 1 format (`all_results` key)
- **Auto-detects** Tournament format (`archive` key) 
- **Graceful error** for unknown formats

### ✅ Unified Interface
```bash
# Works with Stage 1 output
--archive_file blended_2pid_comprehensive_results.json

# Works with Tournament output  
--archive_file tournament_archive.json

# Works with any previous tournament results
--archive_file tournament1_results.json
```

## Implementation Plan

1. **Modify tournament_optimizer.py** - Add multi-format support
2. **Test with Stage 1 results** - Verify seeding works
3. **Run corrected pipeline** - Stage 1 → Tournament #1 direct flow
4. **Validate improvement** - Tournament #1 should achieve < 76.81 cost

## Command Sequence
```bash
# Run corrected Tournament #1 directly with Stage 1 results
python optimization/tournament_optimizer.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data \
  --archive_file blended_2pid_comprehensive_results.json \
  --population_size 100 \
  --rounds 10 \
  --output_file tournament1_corrected_results.json
```

This is **architecturally cleaner** because:
- ✅ No external scripts
- ✅ Single point of format handling  
- ✅ Maintains existing functionality
- ✅ Direct pipeline flow
- ✅ Self-documenting format detection