#!/usr/bin/env python3
"""
Generate training data for BlenderNet from PID tournament archive results.

This script analyzes the tournament archive to create supervised learning data
for training neural networks to predict optimal PID blending weights.
"""

import argparse
import json
import numpy as np
import random
from pathlib import Path
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.logging import print_banner, print_params, print_summary, tqdm, EMOJI_LAB, EMOJI_OK

def load_tournament_archive(archive_path="plans/tournament_archive.json"):
    """Load PID tournament archive"""
    if not Path(archive_path).exists():
        raise FileNotFoundError(f"Tournament archive not found: {archive_path}")
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    return archive

def analyze_archive_performance(archive):
    """Analyze archive to understand performance patterns"""
    print("📊 Archive Analysis:")
    print("-" * 40)
    if isinstance(archive, dict) and 'archive' in archive and isinstance(archive['archive'], list):
        performers = archive['archive']
    elif isinstance(archive, dict):
        performers = list(archive.values())
    else:
        performers = archive

    costs = []
    min_costs = []
    for p in performers:
        stats = p.get('stats', {})
        cost = stats.get('mean_cost', stats.get('avg_total_cost', float('nan')))
        low_min = stats.get('min_cost', float('nan'))
        costs.append(cost)
        min_costs.append(low_min)

    print(f"Total parameter combinations: {len(performers)}")
    print(f"Average cost range: {min(costs):.2f} - {max(costs):.2f}")
    print(f"Best minimum cost: {min(min_costs):.2f}")
    print(f"Performance gap: {max(costs)/min(costs):.1f}x")

    sorted_performers = sorted(
        performers,
        key=lambda x: x.get('stats', {}).get('mean_cost', x.get('stats', {}).get('avg_total_cost', float('nan')))
    )
    top_performers = sorted_performers[:10]

    print(f"\nTop 10 Performers:")
    for i, p in enumerate(top_performers):
        params = p.get('parameters', {})
        low = p.get('low_gains', params.get('low_gains', []))
        high = p.get('high_gains', params.get('high_gains', []))
        cost = p.get('stats', {}).get('mean_cost', p.get('stats', {}).get('avg_total_cost', float('nan')))
        print(f"  {i+1:2d}. Cost: {cost:.2f} Low: {low} High: {high}")

    return top_performers

def compute_optimal_blend_weight(v_ego, error_magnitude, future_complexity, scenario_type="normal"):
    """Compute optimal blend weight based on vehicle state and scenario."""
    speed_blend = np.clip((v_ego - 20) / 30, 0, 1)
    error_adjustment = 0.2 if error_magnitude > 0.5 else -0.2
    complexity_adjustment = 0.1 if future_complexity > 0.7 else -0.1
    if scenario_type == "highway":
        scenario_adjustment = 0.1
    elif scenario_type == "city":
        scenario_adjustment = -0.1
    else:
        scenario_adjustment = 0.0
    optimal_blend = speed_blend + error_adjustment + complexity_adjustment + scenario_adjustment
    return float(np.clip(optimal_blend, 0.0, 1.0))

def generate_training_samples(top_performers, num_samples):
    """Generate training samples for BlenderNet with progress bars and summaries."""
    print(f"\n🔬 Generating {num_samples} training samples...")
    training_samples = []
    total_performers = len(top_performers)
    samples_per_combo = num_samples // total_performers

    # Define speed bins to ensure balanced coverage across driving regimes.
    # Low speeds (<25 mph) typify city driving, medium speeds (25–45 mph)
    # represent transitional/suburban travel, and high speeds (45–70 mph)
    # capture highway conditions. Sampling an equal number from each bin
    # prevents the training set from being biased toward any one regime.
    speed_bins = [
        ("low", 5, 25),
        ("medium", 25, 45),
        ("high", 45, 70),
    ]
    num_bins = len(speed_bins)

    for idx, combo in enumerate(tqdm(top_performers, desc="Combos", unit="combo"), start=1):
        combo_samples = []

        # Determine how many samples to draw for each bin so each bin
        # contributes equally. Any remainder is distributed across the first
        # few bins.
        base_count = samples_per_combo // num_bins
        remainder = samples_per_combo % num_bins
        bin_counts = [base_count + (1 if i < remainder else 0) for i in range(num_bins)]

        for (bin_name, v_min, v_max), bin_count in zip(speed_bins, bin_counts):
            for _ in tqdm(range(bin_count), desc=f" Samples ({bin_name})", leave=False, unit="sample"):
                v_ego = np.random.uniform(v_min, v_max)
                roll_lataccel = np.clip(np.random.normal(0, 1.5), -4, 4)
                a_ego = np.clip(np.random.normal(0, 1.0), -3, 3)
                error = np.clip(np.random.laplace(0, 0.3), -2, 2)
                error_integral = np.clip(np.random.normal(0, 0.2), -1, 1)
                error_derivative = np.clip(np.random.normal(0, 0.1), -0.5, 0.5)
                future_lataccel_std = np.clip(np.random.exponential(0.5), 0, 2)
                features = [
                    v_ego, roll_lataccel, a_ego,
                    error, error_integral, error_derivative,
                    np.clip(np.random.normal(0, 1.0), -3, 3),
                    future_lataccel_std
                ]
                blend = compute_optimal_blend_weight(
                    v_ego,
                    abs(error),
                    future_lataccel_std,
                    "highway" if v_ego > 50 else "city" if v_ego < 25 else "normal",
                )
                combo_samples.append((features, blend))

        training_samples.extend(combo_samples)
        print(f"  ✅ Generated {len(combo_samples)} samples for combo {idx}/{total_performers}")

    # Fill any remainder
    remainder_idx = 0
    while len(training_samples) < num_samples:
        combo = random.choice(top_performers)
        bin_name, v_min, v_max = speed_bins[remainder_idx % num_bins]
        remainder_idx += 1
        v_ego = np.random.uniform(v_min, v_max)
        error = np.clip(np.random.laplace(0, 0.3), -2, 2)
        future_complexity = np.clip(np.random.exponential(0.5), 0, 2)
        blend = compute_optimal_blend_weight(
            v_ego,
            abs(error),
            future_complexity,
            "highway" if v_ego > 50 else "city" if v_ego < 25 else "normal",
        )
        features = [v_ego, 0, 0, error, 0, 0, 0, future_complexity]
        training_samples.append((features, blend))

    print_summary("Training Samples Summary", {
        "total_samples": len(training_samples),
        "samples_per_combo": samples_per_combo,
        "combos": total_performers
    })
    return training_samples

def save_training_data(training_samples, output_path):
    """Save training data to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Compute feature statistics for normalization
    feature_array = np.array([s[0] for s in training_samples], dtype=np.float32)
    feature_means = feature_array.mean(axis=0).tolist()
    feature_stds = feature_array.std(axis=0).tolist()

    data = {
        'num_samples': len(training_samples),
        'feature_names': [
            'v_ego', 'roll_lataccel', 'a_ego', 'error',
            'error_integral', 'error_derivative',
            'future_lataccel_mean', 'future_lataccel_std'
        ],
        'feature_stats': {
            'mean': feature_means,
            'std': feature_stds,
        },
        'samples': [
            {'features': s[0], 'blend_weight': s[1]}
            for s in training_samples
        ]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n{EMOJI_OK} Training data saved to: {output_path}")
    print_summary("Data Generation Complete", {"Total Samples": len(training_samples)})

def main():
    parser = argparse.ArgumentParser(description="BlenderNet training data generation")
    parser.add_argument("--samples", type=int, default=15000,
                        help="Number of training samples to generate")
    parser.add_argument("--output-path", dest="output_path", type=str,
                        default="plans/blender_training_data.json",
                        help="Path to write training data JSON")
    parser.add_argument("--data-seed", dest="data_seed", type=int, default=None,
                        help="Seed for reproducible sample generation")
    args = parser.parse_args()

    if args.data_seed is not None:
        random.seed(args.data_seed)
        np.random.seed(args.data_seed)

    print_banner(4, "Data Generation & Pre-Training")
    print_params({
        "archive": "plans/tournament_archive.json",
        "sample count": args.samples,
        "output path": args.output_path
    })

    try:
        archive = load_tournament_archive()
        top = analyze_archive_performance(archive)
        samples = generate_training_samples(top, args.samples)
        save_training_data(samples, args.output_path)
        print("\n✅ Training data generation complete!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
