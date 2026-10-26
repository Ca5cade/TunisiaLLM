import os
import sys
import random
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure UTF-8 output on all operating systems (Windows cp1252 fix)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def is_kaggle_environment() -> bool:
    """Detect if running inside a Kaggle notebook environment."""
    return os.path.exists("/kaggle/working")


def load_config(config_path: Optional[str] = None, smoke_test_override: Optional[bool] = None) -> Dict[str, Any]:
    """
    Load configuration from yaml file, automatically adapting paths for Kaggle or local dev.
    """
    if config_path is None:
        # Check standard locations
        candidates = [
            "config.yaml",
            "/kaggle/working/config.yaml",
            os.path.join(os.path.dirname(__file__), "config.yaml")
        ]
        for c in candidates:
            if os.path.isfile(c):
                config_path = c
                break
        if config_path is None:
            raise FileNotFoundError("Could not find config.yaml in workspace or current directory.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow CLI flag or environment override
    if smoke_test_override is not None:
        cfg["smoke_test"] = smoke_test_override

    # Adapt paths if running in Kaggle
    in_kaggle = is_kaggle_environment()
    root_dir = "/kaggle/working" if in_kaggle else "."

    for k in ["data_dir", "output_dir", "checkpoint_dir", "tokenizer_path", "sft_data_dir", "log_dir"]:
        if k in cfg["paths"]:
            p = cfg["paths"][k]
            if not os.path.isabs(p):
                cfg["paths"][k] = os.path.abspath(os.path.join(root_dir, p))

    # Ensure directories exist
    for k in ["data_dir", "output_dir", "checkpoint_dir", "sft_data_dir", "log_dir"]:
        os.makedirs(cfg["paths"][k], exist_ok=True)

    return cfg


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_num_params(num_params: int) -> str:
    """Format parameter count into human-readable string (e.g. 78.4M)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def format_bytes(num_bytes: int) -> str:
    """Format bytes into KB, MB, GB."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"
