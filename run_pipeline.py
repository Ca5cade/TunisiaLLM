#!/usr/bin/env python3
"""
run_pipeline.py - Master Automated 12-Hour Training Pipeline for TunisiaLLM
Orchestrates the entire end-to-end lifecycle on Kaggle (2x T4 GPUs):
  [Stage 1] Data Preparation & Cleaning (01_data_prep.py)
  [Stage 2] Tokenizer Training & Binary Sharding (02_tokenizer.py)
  [Stage 3] Distributed DDP Pretraining ~9.5h (04_pretrain.py)
  [Stage 4] SFT Chat & Tool Data Synthesis (05_make_sft_data.py)
  [Stage 5] SFT Fine-Tuning ~25m (06_sft.py)
  [Stage 6] Automated Inference Benchmark (07_inference.py)
  [Stage 7] Model Packaging & Zip Archive Creation
"""

import os
import sys
import time
import shutil
import zipfile
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import List, Optional

import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from utils import load_config, format_time, format_bytes


os.environ.setdefault("PYTHONUNBUFFERED", "1")

def log_banner(title: str, fill: str = "=") -> None:
    """Print an eye-catching stage header."""
    print("\n" + fill * 80)
    print(f" {title} ")
    print(fill * 80 + "\n", flush=True)


def run_command(cmd: List[str], desc: str) -> None:
    """Run a shell command with real-time streaming output and error checking."""
    log_banner(f"STARTING: {desc}")
    print(f"Executing: {' '.join(cmd)}\n", flush=True)

    start_t = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    if proc.stdout is not None:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
        proc.stdout.close()

    return_code = proc.wait()
    elapsed = time.time() - start_t

    if return_code != 0:
        print(f"\n[!] ERROR: {desc} failed with exit code {return_code} after {format_time(elapsed)}.")
        sys.exit(return_code)

    print(f"\n[OK] {desc} completed successfully in {format_time(elapsed)}.\n", flush=True)


def get_gpu_info() -> int:
    """Detect available CUDA GPUs and print VRAM status."""
    if not torch.cuda.is_available():
        print("[!] No CUDA GPU detected. Running on CPU.")
        return 0

    count = torch.cuda.device_count()
    print(f"[+] CUDA Available: {count} GPU(s) detected:")
    for i in range(count):
        prop = torch.cuda.get_device_properties(i)
        vram_gb = prop.total_memory / (1024 ** 3)
        print(f"    - GPU {i}: {prop.name} ({vram_gb:.1f} GB VRAM, Compute {prop.major}.{prop.minor})")
    return count


def package_model(checkpoint_dir: str, tokenizer_path: str, config_path: str, output_zip_path: str) -> None:
    """Zip best SFT model, tokenizer, and config for 1-click download."""
    log_banner("PACKAGING TUNISIALLM FOR DOWNLOAD")

    best_sft = os.path.join(checkpoint_dir, "sft_checkpoint_best.pt")
    if not os.path.isfile(best_sft):
        # Fallback to final or pretrained
        for candidate in ["sft_checkpoint_final.pt", "checkpoint_best.pt", "checkpoint_latest.pt"]:
            p = os.path.join(checkpoint_dir, candidate)
            if os.path.isfile(p):
                best_sft = p
                break

    print(f"Packaging files into: {output_zip_path}")
    print(f"  [+] Checkpoint: {best_sft} ({format_bytes(os.path.getsize(best_sft)) if os.path.isfile(best_sft) else 'MISSING'})")
    print(f"  [+] Tokenizer:  {tokenizer_path}")
    print(f"  [+] Config:     {config_path}")

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.isfile(best_sft):
            zf.write(best_sft, arcname="sft_checkpoint_best.pt")
        if os.path.isfile(tokenizer_path):
            zf.write(tokenizer_path, arcname="tokenizer.json")
        if os.path.isfile(config_path):
            zf.write(config_path, arcname="config.yaml")

    zip_size = os.path.getsize(output_zip_path)
    print(f"\n[OK] Package created successfully: {output_zip_path} ({format_bytes(zip_size)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="TunisiaLLM 12-Hour Master Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--total-hours", type=float, default=11.2, help="Total session budget in hours (default 11.2h)")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny smoke test verification across all steps")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pretraining (e.g. if already pretrained)")
    parser.add_argument("--force-prep", action="store_true", help="Force re-running data prep and tokenizer")
    args = parser.parse_args()

    session_start = time.time()
    total_budget_secs = args.total_hours * 3600

    cfg = load_config(args.config, smoke_test_override=args.smoke_test)
    num_gpus = get_gpu_info()

    py_exec = [sys.executable, "-u"]

    # Determine execution runner (torchrun for multi-GPU, python for single)
    if num_gpus >= 2 and not args.smoke_test:
        runner_prefix = ["torchrun", f"--nproc_per_node={num_gpus}"]
    else:
        runner_prefix = list(py_exec)

    log_banner(f"TUNISIALLM 12-HOUR AUTONOMOUS TRAINING PIPELINE")
    print(f"Session Budget:      {args.total_hours:.1f} hours ({format_time(total_budget_secs)})")
    print(f"Estimated End Time:  {datetime.now() + timedelta(seconds=total_budget_secs):%Y-%m-%d %H:%M:%S}")
    print(f"Execution Engine:    {' '.join(runner_prefix)}")
    print(f"Configuration File:  {args.config}")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # STAGE 1: Data Preparation
    # -------------------------------------------------------------------------
    data_prep_cmd = list(py_exec) + ["01_data_prep.py", "--config", args.config]
    if args.smoke_test:
        data_prep_cmd.append("--smoke-test")
    if args.force_prep:
        data_prep_cmd.append("--force")
    run_command(data_prep_cmd, "Stage 1: Data Collection & Cleaning")

    # -------------------------------------------------------------------------
    # STAGE 2: Tokenizer Training & Sharding
    # -------------------------------------------------------------------------
    tok_cmd = list(py_exec) + ["02_tokenizer.py", "--config", args.config]
    if args.smoke_test:
        tok_cmd.append("--smoke-test")
    if args.force_prep:
        tok_cmd.append("--force")
    run_command(tok_cmd, "Stage 2: Tokenizer Training & Corpus Sharding")

    # -------------------------------------------------------------------------
    # STAGE 3: Pretraining (Dynamic Time Budgeting)
    # -------------------------------------------------------------------------
    if not args.skip_pretrain:
        # Leave 1.2 hours for SFT, benchmark, and packaging
        elapsed_so_far = time.time() - session_start
        remaining_secs = total_budget_secs - elapsed_so_far
        pretrain_budget_hours = max(0.5, (remaining_secs - 4200) / 3600.0)

        pretrain_cmd = list(runner_prefix) + [
            "04_pretrain.py",
            "--config", args.config,
            "--max-session-hours", f"{pretrain_budget_hours:.2f}"
        ]
        if args.smoke_test:
            pretrain_cmd.append("--smoke-test")

        print(f"Allocating {pretrain_budget_hours:.2f} hours for Pretraining...")
        run_command(pretrain_cmd, f"Stage 3: Distributed Pretraining ({pretrain_budget_hours:.2f}h Budget)")
    else:
        print("\n[*] Skipping Stage 3 (Pretraining) as requested.\n")

    # -------------------------------------------------------------------------
    # STAGE 4: SFT Dataset Generation
    # -------------------------------------------------------------------------
    sft_data_cmd = list(py_exec) + ["05_make_sft_data.py", "--config", args.config]
    if args.smoke_test:
        sft_data_cmd.append("--smoke-test")
    run_command(sft_data_cmd, "Stage 4: SFT Chat & Tool Calling Dataset Synthesis")

    # -------------------------------------------------------------------------
    # STAGE 5: SFT Fine-Tuning
    # -------------------------------------------------------------------------
    sft_cmd = list(runner_prefix) + ["06_sft.py", "--config", args.config]
    if args.smoke_test:
        sft_cmd.append("--smoke-test")
    run_command(sft_cmd, "Stage 5: Supervised Fine-Tuning (SFT)")

    # -------------------------------------------------------------------------
    # STAGE 6: Inference Benchmark & Verification
    # -------------------------------------------------------------------------
    bench_cmd = list(py_exec) + ["07_inference.py", "--config", args.config, "--benchmark"]
    run_command(bench_cmd, "Stage 6: Automated Inference & Tool Verification Benchmark")

    # -------------------------------------------------------------------------
    # STAGE 7: Model Packaging
    # -------------------------------------------------------------------------
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    tok_path = cfg["paths"]["tokenizer_path"]
    zip_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else REPO_ROOT
    zip_path = os.path.join(zip_dir, "tunisiallm_model.zip")
    package_model(ckpt_dir, tok_path, args.config, zip_path)

    total_time = time.time() - session_start
    log_banner("TUNISIALLM 12-HOUR PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Elapsed Time: {format_time(total_time)}")
    print(f"Package Ready at:   {zip_path}")
    print("\nHow to download your model:")
    print("  1. In the right sidebar under 'Output', click Refresh 🔄 -> Three dots next to 'tunisiallm_model.zip' -> Download")
    print("  2. Or click 'Save Version' -> 'Quick Save' -> download from the Output tab.\n")


if __name__ == "__main__":
    main()
