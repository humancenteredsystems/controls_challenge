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
from utils.blending import get_smooth_blend_weight

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

def compute_optimal_blend_weight(v_ego: float) -> float:
    """
    Compute optimal blend weight based on vehicle state.
    This now uses the centralized smooth blending function.
    """
    return get_smooth_blend_weight(v_ego)

def generate_training_samples(top_performers, num_samples):
    """Generate training samples for BlenderNet with balanced blend weights.

    The generator keeps track of the proportion of low (<0.3), mid and high
    (>0.7) blend weights. Sampling is resubmitted with a bias toward the most
    underrepresented category until the final distribution is within ±5% of the
    target ratios (uniform by default).
    """

    print(f"\n🔬 Generating {num_samples} training samples...")
    training_samples = []
    total_performers = len(top_performers)
    samples_per_combo = num_samples // total_performers

    categories = {"low": 0, "mid": 0, "high": 0}
    target_ratio = 1.0 / 3.0
    target_counts = {
        "low": int(target_ratio * num_samples),
        "mid": int(target_ratio * num_samples),
        "high": int(target_ratio * num_samples),
    }
    # Adjust for rounding errors
    while sum(target_counts.values()) < num_samples:
        target_counts["mid"] += 1

    def categorize(blend):
        if blend < 0.3:
            return "low"
        if blend > 0.7:
            return "high"
        return "mid"

    def deficit(category):
        return target_counts[category] - categories[category]

    for idx, combo in enumerate(tqdm(top_performers, desc="Combos", unit="combo"), start=1):
        combo_samples = []
        for _ in tqdm(range(samples_per_combo), desc=" Samples", leave=False, unit="sample"):
            while True:
                v_ego = np.clip(np.random.gamma(2, 15), 5, 70)

                roll_lataccel = np.clip(np.random.normal(0, 1.5), -4, 4)
                a_ego = np.clip(np.random.normal(0, 1.0), -3, 3)
                error = np.clip(np.random.laplace(0, 0.3), -2, 2)
                error_integral = np.clip(np.random.normal(0, 0.2), -1, 1)
                error_derivative = np.clip(np.random.normal(0, 0.1), -0.5, 0.5)
                future_lataccel_std = np.clip(np.random.exponential(0.5), 0, 2)

                bias = 0.0
                # steer sampling toward underrepresented categories
                most_needed = max(categories, key=lambda c: deficit(c))
                if most_needed == "low":
                    bias = -0.2
                elif most_needed == "high":
                    bias = 0.2

                blend = compute_optimal_blend_weight(v_ego)
                cat = categorize(blend)
                if categories[cat] < target_counts[cat]:
                    features = [
                        v_ego,
                        roll_lataccel,
                        a_ego,
                        error,
                        error_integral,
                        error_derivative,
                        np.clip(np.random.normal(0, 1.0), -3, 3),
                        future_lataccel_std,
                    ]
                    combo_samples.append((features, blend))
                    categories[cat] += 1
                    break


        training_samples.extend(combo_samples)
        print(f"  ✅ Generated {len(combo_samples)} samples for combo {idx}/{total_performers}")

    # Fill any remainder
    remainder_idx = 0
    while len(training_samples) < num_samples:
        combo = random.choice(top_performers)

        while True:
            v_ego = np.clip(np.random.gamma(2, 15), 5, 70)
            error = np.clip(np.random.laplace(0, 0.3), -2, 2)
            future_complexity = np.clip(np.random.exponential(0.5), 0, 2)

            most_needed = max(categories, key=lambda c: deficit(c))
            bias = -0.2 if most_needed == "low" else 0.2 if most_needed == "high" else 0.0

            blend = compute_optimal_blend_weight(v_ego)
            cat = categorize(blend)
            if categories[cat] < target_counts[cat]:
                features = [v_ego, 0, 0, error, 0, 0, 0, future_complexity]
                training_samples.append((features, blend))
                categories[cat] += 1
                break

    proportions = {k: v / num_samples for k, v in categories.items()}
    print_summary(
        "Training Samples Summary",
        {
            "total_samples": len(training_samples),
            "samples_per_combo": samples_per_combo,
            "combos": total_performers,
            "blend_weight_proportions": proportions,
        },
    )

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
