#!/usr/bin/env python3
import json
from pathlib import Path

# Load and examine archive structure
archive_path = "plans/tournament_archive.json"

if Path(archive_path).exists():
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    print("Archive structure:")
    print(f"Top-level keys: {archive.keys()}")
    
    if 'archive' in archive:
        performers = archive['archive']
        print(f"Number of performers: {len(performers)}")
        
        if performers:
            print("\nFirst performer structure:")
            first = performers[0]
            for key, value in first.items():
                print(f"  {key}: {type(value)} = {value}")
            
            if 'stats' in first:
                print(f"\nStats keys: {first['stats'].keys()}")
                for stat_key, stat_value in first['stats'].items():
                    print(f"  {stat_key}: {stat_value}")
        
        # Try to access avg_total_cost from all performers
        print(f"\nChecking avg_total_cost access:")
        for i, p in enumerate(performers[:3]):  # Check first 3
            try:
                cost = p['stats']['avg_total_cost']
                print(f"  Performer {i}: {cost}")
                break
            except KeyError as e:
                print(f"  Performer {i}: KeyError {e}")
                print(f"    Available keys: {p.keys()}")
                if 'stats' in p:
                    print(f"    Stats keys: {p['stats'].keys()}")
    else:
        print("No 'archive' key found")
else:
    print(f"Archive file {archive_path} not found")