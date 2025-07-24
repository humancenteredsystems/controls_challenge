# Pipeline Format Bridge Solution

## Problem Analysis

The corrected time-step pipeline is broken because **Stage 1 output format** does NOT match **Tournament #1 input format**.

### Stage 1 Output Format ([`blended_2pid_comprehensive_results.json`](blended_2pid_comprehensive_results.json))
```json
{
  "best_cost": 76.81,
  "all_results": [
    {
      "combination_id": 278,
      "low_gains": [0.286, 0.094, -0.088],
      "high_gains": [0.155, 0.005, -0.03],
      "avg_total_cost": 76.81,
      "std_cost": 31.54,
      "min_cost": 8.87,
      "max_cost": 127.55,
      "num_files": 25,
      "test_time": 7.95
    }
  ]
}
```

### Tournament #1 Expected Input Format ([`plans/tournament_archive.json`](plans/tournament_archive.json))
```json
{
  "archive": [
    {
      "id": "uuid-string",
      "low_gains": [0.381, 0.115, -0.104],
      "high_gains": [0.3, 0.017, -0.131],
      "stats": {
        "avg_total_cost": 2689.53,
        "std_cost": 482.80,
        "min_cost": 1516.52,
        "max_cost": 3453.88,
        "num_files": 30
      },
      "rounds_survived": 10,
      "status": "active"
    }
  ]
}
```

## Format Transformation Required

| Stage 1 Field | Tournament Field | Transformation |
|---------------|------------------|----------------|
| `all_results` | `archive` | Rename array |
| `combination_id` | `id` | Convert int → UUID string |
| `avg_total_cost` | `stats.avg_total_cost` | Move to nested object |
| `std_cost` | `stats.std_cost` | Move to nested object |
| `min_cost` | `stats.min_cost` | Move to nested object |
| `max_cost` | `stats.max_cost` | Move to nested object |
| `num_files` | `stats.num_files` | Move to nested object |
| N/A | `rounds_survived` | Add default: 0 |
| N/A | `status` | Add default: "champion" |

## Solution: Converter Script

Create [`scripts/convert_stage1_to_tournament.py`](scripts/convert_stage1_to_tournament.py):

```python
#!/usr/bin/env python3
"""
Convert Stage 1 blended_2pid_comprehensive_results.json to Tournament #1 archive format
Enables proper pipeline flow from corrected Stage 1 → Tournament #1
"""

import json
import uuid
import argparse
from pathlib import Path

def convert_stage1_to_tournament(stage1_file: str, output_file: str, top_n: int = 50):
    """
    Convert Stage 1 comprehensive results to Tournament archive format
    
    Args:
        stage1_file: Path to blended_2pid_comprehensive_results.json
        output_file: Path to write tournament archive format
        top_n: Number of top performers to convert (default: 50)
    """
    
    # Load Stage 1 results
    with open(stage1_file, 'r') as f:
        stage1_data = json.load(f)
    
    # Extract and sort by cost (best first)
    all_results = stage1_data['all_results']
    sorted_results = sorted(all_results, key=lambda x: x['avg_total_cost'])[:top_n]
    
    # Convert to tournament format
    tournament_archive = []
    
    for result in sorted_results:
        tournament_entry = {
            "id": str(uuid.uuid4()),
            "low_gains": result["low_gains"],
            "high_gains": result["high_gains"],
            "stats": {
                "avg_total_cost": result["avg_total_cost"],
                "std_cost": result["std_cost"], 
                "min_cost": result["min_cost"],
                "max_cost": result["max_cost"],
                "num_files": result["num_files"]
            },
            "rounds_survived": 0,  # New champions start at 0
            "status": "champion"   # Mark as champions from Stage 1
        }
        tournament_archive.append(tournament_entry)
    
    # Create final tournament format
    tournament_data = {
        "archive": tournament_archive
    }
    
    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(tournament_data, f, indent=2)
    
    print(f"✅ Converted {len(tournament_archive)} Stage 1 champions to tournament format")
    print(f"📁 Output: {output_file}")
    print(f"🏆 Best cost: {sorted_results[0]['avg_total_cost']:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Convert Stage 1 results to Tournament format")
    parser.add_argument("--stage1_file", 
                       default="blended_2pid_comprehensive_results.json",
                       help="Input Stage 1 results file")
    parser.add_argument("--output_file",
                       default="plans/tournament_archive_stage1_champions.json", 
                       help="Output tournament archive file")
    parser.add_argument("--top_n", type=int, default=50,
                       help="Number of top performers to convert")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.stage1_file).exists():
        print(f"❌ Error: Input file {args.stage1_file} not found")
        return 1
    
    # Create output directory if needed
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    convert_stage1_to_tournament(args.stage1_file, args.output_file, args.top_n)
    
    return 0

if __name__ == "__main__":
    exit(main())
```

## Implementation Steps

### Step 1: Create Converter Script
```bash
# Create the converter script
cat > scripts/convert_stage1_to_tournament.py << 'EOF'
[paste script content above]
EOF

# Make executable
chmod +x scripts/convert_stage1_to_tournament.py
```

### Step 2: Convert Stage 1 Results
```bash
# Convert corrected Stage 1 results to tournament format
python scripts/convert_stage1_to_tournament.py \
  --stage1_file blended_2pid_comprehensive_results.json \
  --output_file plans/tournament_archive_stage1_champions.json \
  --top_n 50
```

### Step 3: Run Tournament #1 with Proper Seeding
```bash
# Run Tournament #1 with Stage 1 champions as seeds
python optimization/tournament_optimizer.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data \
  --archive_file plans/tournament_archive_stage1_champions.json \
  --population_size 100 \
  --rounds 10 \
  --output_file tournament_stage2_results.json
```

## Expected Results

### Before Fix (Broken Pipeline):
- **Stage 1**: 76.81 cost ✅
- **Tournament #1**: 335.66 cost ❌ (random initialization, ignored Stage 1)

### After Fix (Proper Pipeline Flow):
- **Stage 1**: 76.81 cost ✅  
- **Tournament #1**: **< 76.81 cost** ✅ (seeded with Stage 1 champions)

## Verification Commands

```bash
# Verify Stage 1 results format
head -20 blended_2pid_comprehensive_results.json

# Run converter
python scripts/convert_stage1_to_tournament.py

# Verify tournament format
head -50 plans/tournament_archive_stage1_champions.json

# Check tournament seeding works
python optimization/tournament_optimizer.py --help
```

## Pipeline Flow Diagram

```
Stage 1: Broad Parameter Search
├── Input: Random parameter exploration
├── Output: blended_2pid_comprehensive_results.json
└── Best: 76.81 cost
         ↓
[FORMAT CONVERTER] ← **THIS WAS MISSING!**
├── Input: blended_2pid_comprehensive_results.json  
├── Output: tournament_archive_stage1_champions.json
└── Transform: all_results → archive with UUIDs
         ↓
Tournament #1: Competition Refinement
├── Input: tournament_archive_stage1_champions.json (seeded)
├── Output: tournament_stage2_results.json
└── Expected: < 76.81 cost (improvement)
         ↓
Tournament #2 → Blender Tournament → Final Controller
```

This converter script bridges the format gap and enables the **corrected time-step pipeline** to flow properly from Stage 1 → Tournament #1.