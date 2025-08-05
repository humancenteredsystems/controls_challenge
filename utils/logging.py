"""Shared logging utilities for consistent terminal reporting across pipeline stages."""

from tqdm import tqdm as _tqdm

EMOJI_TROPHY = "🏆"
EMOJI_LAB = "🔬"
EMOJI_OK = "✅"
EMOJI_WARN = "⚠️"
EMOJI_PARTY = "🎉"

def tqdm(iterable, **kwargs):
    """Wrapper around tqdm."""
    return _tqdm(iterable, **kwargs)

def print_banner(stage_num, stage_name):
    """Print a bold banner for a pipeline stage."""
    banner = f"STAGE {stage_num}: {stage_name}" if stage_num is not None else stage_name
    sep = "=" * len(banner)
    print(f"\n{EMOJI_LAB} {banner}\n{sep}", flush=True)

def print_params(params):
    """Print key parameters at the start of a stage."""
    print("Parameters:", flush=True)
    for k, v in params.items():
        print(f"  - {k}: {v}", flush=True)

def print_summary(summary_title, summary_dict):
    """Print a concise summary after a major phase."""
    print(f"\n{EMOJI_TROPHY} {summary_title}", flush=True)
    for k, v in summary_dict.items():
        print(f"  - {k}: {v}", flush=True)

def print_goal_progress(current_cost, goal=45.0):
    """Track progress toward the <45 total_cost goal."""
    if current_cost < goal:
        print(f"🎯 GOAL ACHIEVED: {current_cost:.2f} < {goal}", flush=True)
    else:
        remaining = current_cost - goal
        print(f"📊 Goal progress: {current_cost:.2f} (need to reduce by {remaining:.2f})", flush=True)
