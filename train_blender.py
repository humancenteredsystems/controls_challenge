#!/usr/bin/env python3
"""
Train the BlenderNet model on the generated training data.
"""

import argparse
from pathlib import Path
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from neural_blender_net import train_blender_net_from_json
from utils.logging import print_banner, print_params, print_summary, EMOJI_OK

def main():
    parser = argparse.ArgumentParser(description="BlenderNet training script")
    parser.add_argument("--data-path", dest="data_path", type=str,
                        default="plans/blender_training_data.json",
                        help="Path to training data JSON file")
    parser.add_argument("--model-output", dest="model_output", type=str,
                        default="models/blender_trained.pth",
                        help="Path to write trained model")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for optimizer")
    args = parser.parse_args()

    print_banner(4, "BlenderNet Training")
    print_params({
        "data path": args.data_path,
        "model output": args.model_output,
        "epochs": args.epochs,
        "batch size": args.batch_size,
        "learning rate": args.lr
    })

    try:
        results = train_blender_net_from_json(
            args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_output=args.model_output,
            lr=args.lr
        )
        print_summary(
            "Training Complete",
            {
                "Train Loss": f"{results['train_loss']:.4f}",
                "Val Loss": f"{results['val_loss']:.4f}",
                "Best Val Loss": f"{results['best_val_loss']:.4f}"
            }
        )
        print(f"\n{EMOJI_OK} Trained model saved to: {args.model_output}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
