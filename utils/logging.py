"""Shared logging utilities for consistent terminal reporting across pipeline stages."""

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, **kwargs):
        # Fallback: no progress bar
        return iterable

EMOJI_TROPHY = "🏆"
EMOJI_LAB = "🔬"
EMOJI_OK = "✅"
EMOJI_WARN = "⚠️"
EMOJI_PARTY = "🎉"

def tqdm(iterable, **kwargs):
    """Wrapper around tqdm with fallback."""
    return _tqdm(iterable, **kwargs)

def print_banner(stage_num, stage_name):
    """Print a bold banner for a pipeline stage."""
    banner = f"STAGE {stage_num}: {stage_name}" if stage_num is not None else stage_name
    sep = "=" * len(banner)
    print(f"\n{EMOJI_LAB} {banner}\n{sep}")

def print_params(params):
    """Print key parameters at the start of a stage."""
    print("Parameters:")
    for k, v in params.items():
        print(f"  - {k}: {v}")

def print_summary(summary_title, summary_dict):
    """Print a concise summary after a major phase."""
    print(f"\n{EMOJI_TROPHY} {summary_title}")
    for k, v in summary_dict.items():
        print(f"  - {k}: {v}")
