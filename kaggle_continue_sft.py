"""
kaggle_continue_sft.py - 1-Click Continued Fine-Tuning for TunisiaLLM on Kaggle
Optimized for 2x Tesla T4 GPUs (or 1x GPU) with automatic checkpoint discovery,
multi-GPU DDP orchestration, and 1-click output zip packaging.
"""

import os
import sys
import glob
import shutil
import zipfile
import subprocess
import torch

def main() -> None:
    print("=" * 80)
    print("TUNISIALLM: CONTINUED SFT TRAINING PIPELINE (KAGGLE)")
    print("=" * 80)

    # 1. GPU Detection
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[+] CUDA Available: {torch.cuda.is_available()} | Detected GPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"    - GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / (1024**3):.1f} GB VRAM)")

    # 2. Determine Working Directory
    repo_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(repo_root)
    print(f"[+] Active Workspace: {repo_root}")

    # 3. Checkpoint Discovery
    ckpt_target = os.path.join(repo_root, "checkpoints", "sft_checkpoint_best.pt")
    os.makedirs(os.path.join(repo_root, "checkpoints"), exist_ok=True)

    if not os.path.isfile(ckpt_target):
        print("\n[*] Searching for sft_checkpoint_best.pt across /kaggle/input and /kaggle/working...")
        search_roots = ["/kaggle/input", "/kaggle/working", "tunisiallm_model"]
        candidates = []
        for root in search_roots:
            if os.path.isdir(root):
                candidates.extend(glob.glob(os.path.join(root, "**", "*sft_checkpoint_best*.pt"), recursive=True))
                candidates.extend(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True))

        found = False
        for c in candidates:
            if c.endswith(".pt") and os.path.isfile(c) and c != ckpt_target:
                print(f"  [+] Found checkpoint: {c} ({os.path.getsize(c) / (1024**2):.1f} MB)")
                shutil.copyfile(c, ckpt_target)
                found = True
                break
            elif c.endswith(".zip") and os.path.isfile(c):
                try:
                    with zipfile.ZipFile(c, "r") as zf:
                        for name in zf.namelist():
                            if name.endswith("sft_checkpoint_best.pt"):
                                print(f"  [+] Extracting checkpoint from archive: {c} -> {name}")
                                with open(ckpt_target, "wb") as out_f:
                                    out_f.write(zf.read(name))
                                found = True
                                break
                except Exception as e:
                    pass
            if found:
                break

        if not found:
            print("[!] Warning: sft_checkpoint_best.pt not found automatically.")
            print("    Looking for fallback checkpoints in checkpoints/...")
            for fallback in ["sft_checkpoint_final.pt", "checkpoint_best.pt"]:
                fb_path = os.path.join(repo_root, "checkpoints", fallback)
                if os.path.isfile(fb_path):
                    print(f"  [*] Using fallback base: {fb_path}")
                    ckpt_target = fb_path
                    found = True
                    break
    else:
        print(f"[+] Using existing checkpoint: {ckpt_target} ({os.path.getsize(ckpt_target) / (1024**2):.1f} MB)")

    # 4. Step 1: SFT Dataset Generation
    print("\n" + "=" * 80)
    print("STEP 1: PREPARING SFT DATASET (05_make_sft_data.py)")
    print("=" * 80)
    subprocess.run([sys.executable, "-u", "05_make_sft_data.py", "--config", "config.yaml"], check=True)

    # 5. Step 2: Continued SFT Training
    print("\n" + "=" * 80)
    print("STEP 2: LAUNCHING CONTINUED FINE-TUNING (06_sft.py)")
    print("=" * 80)

    if n_gpus >= 2:
        sft_cmd = [
            "torchrun", f"--nproc_per_node={n_gpus}", "06_sft.py",
            "--config", "config.yaml",
            "--checkpoint", ckpt_target,
            "--lr", "2e-5",
            "--epochs", "1"
        ]
    else:
        sft_cmd = [
            sys.executable, "-u", "06_sft.py",
            "--config", "config.yaml",
            "--checkpoint", ckpt_target,
            "--lr", "2e-5",
            "--epochs", "1"
        ]

    print(f"Running command: {' '.join(sft_cmd)}\n")
    subprocess.run(sft_cmd, check=True)

    # 6. Step 3: Packaging Model for Download
    print("\n" + "=" * 80)
    print("STEP 3: PACKAGING UPDATED MODEL FOR DOWNLOAD")
    print("=" * 80)

    zip_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else repo_root
    output_zip = os.path.join(zip_dir, "tunisiallm_model_continued.zip")

    best_ckpt = os.path.join(repo_root, "checkpoints", "sft_checkpoint_best.pt")
    tok_file = os.path.join(repo_root, "tokenizer.json")
    cfg_file = os.path.join(repo_root, "config.yaml")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.isfile(best_ckpt):
            zf.write(best_ckpt, arcname="sft_checkpoint_best.pt")
            print(f"  [+] Included Checkpoint: {best_ckpt} ({os.path.getsize(best_ckpt) / (1024**2):.1f} MB)")
        if os.path.isfile(tok_file):
            zf.write(tok_file, arcname="tokenizer.json")
            print(f"  [+] Included Tokenizer:  {tok_file}")
        if os.path.isfile(cfg_file):
            zf.write(cfg_file, arcname="config.yaml")
            print(f"  [+] Included Config:     {cfg_file}")

    print(f"\n[OK] Package created successfully: {output_zip} ({os.path.getsize(output_zip) / (1024**2):.1f} MB)")
    print(">> Download this zip from the Kaggle Output section in the right sidebar!")

    # 7. Step 4: Quick Inference Test
    print("\n" + "=" * 80)
    print("STEP 4: VERIFICATION CHAT BENCHMARK (07_inference.py)")
    print("=" * 80)
    subprocess.run([
        sys.executable, "-u", "07_inference.py",
        "--config", "config.yaml",
        "--prompt", "عسلامة، شنوة أحسن بلاصة نعمل فيها تصييفة في تونس؟"
    ])

if __name__ == "__main__":
    main()
