"""
kaggle_continue_sft.py - 1-Click Autonomous Model Scaling & Refinement for TunisiaLLM on Kaggle
Optimized for Kaggle 2x Tesla T4 GPUs (or 1x GPU) with:
  1. Automatic discovery and extraction of uploaded checkpoints
  2. Multi-dataset synthesis (7 authentic Tunisian & Arabic instruction datasets + Multi-Turn + Arabizi)
  3. Multi-GPU Distributed Data Parallel (DDP) instruction tuning (06_sft.py)
  4. Post-SFT Direct Preference Optimization (DPO) alignment on GPU (08_dpo.py)
  5. Multi-format export: FP16 (.pt), Dynamic INT8 (.pt), and GGUF + Ollama Modelfile
  6. Automated 6-category verification benchmark (QA, tools, coding, travel, Arabizi)
  7. 1-click output zip packaging for immediate download from the Kaggle sidebar
"""

import os
import sys
import glob
import shutil
import zipfile
import subprocess
import argparse
import torch

def main() -> None:
    parser = argparse.ArgumentParser(description="TunisiaLLM Autonomous Model Scaling & Refinement")
    parser.add_argument("--skip-sft", action="store_true", help="Skip dataset prep and SFT fine-tuning, resume straight from DPO alignment")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset generation")
    parser.add_argument("--skip-dpo", action="store_true", help="Skip DPO alignment and use the pure SFT model directly")
    parser.add_argument("--dpo-steps", type=int, default=50, help="Number of DPO optimization steps")
    parser.add_argument("--dpo-lr", type=float, default=3e-6, help="DPO learning rate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for benchmark")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering for benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to use")
    args = parser.parse_args()

    print("=" * 80)
    print("🇹🇳 TUNISIALLM: AUTONOMOUS MODEL SCALING & REFINEMENT PIPELINE (KAGGLE)")
    print("=" * 80)

    # 1. GPU Detection & Environment Setup
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[+] CUDA Available: {torch.cuda.is_available()} | Detected GPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"    - GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB VRAM)")

    # 2. Determine Working Directory
    repo_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(repo_root)
    print(f"[+] Active Workspace: {repo_root}")

    # 3. Checkpoint Discovery
    ckpt_target = os.path.join(repo_root, "checkpoints", "sft_checkpoint_best.pt")
    os.makedirs(os.path.join(repo_root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(repo_root, "output"), exist_ok=True)

    if not os.path.isfile(ckpt_target):
        print("\n[*] Searching for sft_checkpoint_best.pt across /kaggle/input and workspace...")
        search_roots = ["/kaggle/input", "/kaggle/working", "tunisiallm_model", "."]
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
                except Exception:
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

    # Ensure local tunisiallm_model dir also has the checkpoint
    local_tunisiallm_ckpt = os.path.join(repo_root, "tunisiallm_model", "sft_checkpoint_best.pt")
    os.makedirs(os.path.dirname(local_tunisiallm_ckpt), exist_ok=True)
    if os.path.isfile(ckpt_target) and not os.path.isfile(local_tunisiallm_ckpt):
        shutil.copyfile(ckpt_target, local_tunisiallm_ckpt)

    # 4. Step 1: Scaled SFT Dataset Generation
    if not args.skip_data and not args.skip_sft:
        print("\n" + "=" * 80)
        print("STEP 1: PREPARING SCALED SFT DATASET (05_make_sft_data.py)")
        print("Includes 7 authentic Derja/Arabic corpora, multi-turn dialogues, & Arabizi")
        print("=" * 80)
        subprocess.run([sys.executable, "-u", "05_make_sft_data.py", "--config", "config.yaml", "--force"], check=True)
    else:
        print("\n[*] Step 1 (Dataset generation) skipped as requested.")

    # 5. Step 2: Multi-GPU SFT Instruction Fine-Tuning
    if not args.skip_sft:
        print("\n" + "=" * 80)
        print("STEP 2: LAUNCHING DISTRIBUTED SFT FINE-TUNING (06_sft.py)")
        print("=" * 80)

        if n_gpus >= 2:
            sft_cmd = [
                sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n_gpus}", "06_sft.py",
                "--config", "config.yaml",
                "--checkpoint", ckpt_target,
                "--lr", "3e-5",
                "--epochs", "1"
            ]
        else:
            sft_cmd = [
                sys.executable, "-u", "06_sft.py",
                "--config", "config.yaml",
                "--checkpoint", ckpt_target,
                "--lr", "3e-5",
                "--epochs", "1"
            ]

        print(f"Running command: {' '.join(sft_cmd)}\n")
        subprocess.run(sft_cmd, check=True)
    else:
        print("\n[*] Step 2 (SFT Fine-Tuning) skipped as requested.")

    # Synchronize and discover newly trained checkpoint from Step 2
    for candidate_dir in ["/kaggle/working/checkpoints", os.path.join(repo_root, "checkpoints")]:
        for candidate_name in ["sft_checkpoint_best.pt", "sft_checkpoint_final.pt"]:
            cand_p = os.path.join(candidate_dir, candidate_name)
            if os.path.isfile(cand_p):
                print(f"[+] Found fresh SFT checkpoint: {cand_p} ({os.path.getsize(cand_p)/(1024**2):.1f} MB)")
                ckpt_target = cand_p
                # Sync into repo checkpoints and tunisiallm_model
                repo_best = os.path.join(repo_root, "checkpoints", "sft_checkpoint_best.pt")
                repo_tun = os.path.join(repo_root, "tunisiallm_model", "sft_checkpoint_best.pt")
                if cand_p != repo_best:
                    shutil.copyfile(cand_p, repo_best)
                if cand_p != repo_tun:
                    shutil.copyfile(cand_p, repo_tun)
                break
        if os.path.isfile(ckpt_target):
            break

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt_target = args.checkpoint
        print(f"[+] Explicit checkpoint specified: {ckpt_target}")

    # 6. Step 3: Direct Preference Optimization (DPO) Alignment
    if not args.skip_dpo:
        print("\n" + "=" * 80)
        print("STEP 3: LAUNCHING DIRECT PREFERENCE OPTIMIZATION (DPO) ALIGNMENT (08_dpo.py)")
        print("Penalizing mode collapse, empty greetings, and calculation errors")
        print("=" * 80)

        dpo_device = "cuda" if torch.cuda.is_available() else "cpu"
        dpo_cmd = [
            sys.executable, "-u", "08_dpo.py",
            "--checkpoint", ckpt_target,
            "--tokenizer", "tokenizer.json",
            "--output", ckpt_target,
            "--steps", str(args.dpo_steps),
            "--lr", str(args.dpo_lr),
            "--device", dpo_device
        ]
        print(f"Running command: {' '.join(dpo_cmd)}\n")
        subprocess.run(dpo_cmd, check=True)
    else:
        print("\n[*] Step 3 (DPO Alignment) skipped as requested. Using pure SFT weights.")

    # 7. Step 4: Multi-Format Low-Bit Exporting
    print("\n" + "=" * 80)
    print("STEP 4: MULTI-FORMAT EXPORTS (FP16, INT8, GGUF, OLLAMA)")
    print("=" * 80)

    # PyTorch FP16 and Dynamic INT8
    subprocess.run([
        sys.executable, "-u", "07_inference.py",
        "--checkpoint", ckpt_target,
        "--export-fp16", os.path.join("output", "model_fp16.pt"),
        "--export-int8", os.path.join("output", "model_int8.pt")
    ])

    # GGUF Export for llama.cpp / Ollama
    active_preset = "500m"
    try:
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
            active_preset = c.get("model", {}).get("preset", "500M").lower()
    except Exception:
        pass
    gguf_output = os.path.join("output", f"tunisiallm-{active_preset}-f16.gguf")
    subprocess.run([
        sys.executable, "-u", "export_gguf.py",
        "--checkpoint", ckpt_target,
        "--tokenizer", "tokenizer.json",
        "--output", gguf_output,
        "--dtype", "f16"
    ])

    # 8. Step 5: Verification & Benchmark Suite
    print("\n" + "=" * 80)
    print("STEP 5: COMPREHENSIVE VERIFICATION BENCHMARK (07_inference.py)")
    print(f"Sampling Config: Temperature={args.temperature} | Top-K={args.top_k}")
    print("=" * 80)

    test_prompts = [
        ("General Greeting", "عسلامة يا باهي، شنوة أحوالك اليوم؟"),
        ("Arithmetic Tool", "بربي احسبلي: قداش 45 مع 120 إلا 15؟"),
        ("Shopping Math", "شريت 3 كيلو طماطم الكيلو بـ 1500 مليم، قداش يطيحو؟"),
        ("Coding in Derja", "كيفاش نعمل حلقة تكرار for في بايثون؟"),
        ("Tourism & Vacation", "شنوة أحسن بلاصة نعمل فيها تصييفة في تونس؟"),
        ("Arabizi Question", "kifech na3mel fonction fi python?")
    ]

    for label, prompt in test_prompts:
        print(f"\n[{label}] User: {prompt}")
        print("Assistant: ", end="", flush=True)
        subprocess.run([
            sys.executable, "-u", "07_inference.py",
            "--config", "config.yaml",
            "--checkpoint", ckpt_target,
            "--prompt", prompt,
            "--temperature", str(args.temperature),
            "--top-k", str(args.top_k)
        ])

    # 9. Step 6: 1-Click Packaging for Download
    print("\n" + "=" * 80)
    print("STEP 6: PACKAGING REFINED MODEL FOR 1-CLICK DOWNLOAD")
    print("=" * 80)

    zip_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else repo_root
    output_zip = os.path.join(zip_dir, "tunisiallm_refined_model.zip")

    tok_file = os.path.join(repo_root, "tokenizer.json")
    cfg_file = os.path.join(repo_root, "config.yaml")
    modelfile = os.path.join(repo_root, "output", "Modelfile")
    fp16_file = os.path.join(repo_root, "output", "model_fp16.pt")
    int8_file = os.path.join(repo_root, "output", "model_int8.pt")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.isfile(ckpt_target):
            zf.write(ckpt_target, arcname="sft_checkpoint_best.pt")
            print(f"  [+] Checkpoint (PyTorch): {ckpt_target} ({os.path.getsize(ckpt_target) / (1024**2):.1f} MB)")
        if os.path.isfile(gguf_output):
            zf.write(gguf_output, arcname=os.path.basename(gguf_output))
            print(f"  [+] GGUF Model (Ollama): {gguf_output} ({os.path.getsize(gguf_output) / (1024**2):.1f} MB)")
        if os.path.isfile(modelfile):
            zf.write(modelfile, arcname="Modelfile")
            print(f"  [+] Ollama Modelfile:     {modelfile}")
        if os.path.isfile(fp16_file):
            zf.write(fp16_file, arcname="model_fp16.pt")
            print(f"  [+] FP16 Weights:         {fp16_file} ({os.path.getsize(fp16_file) / (1024**2):.1f} MB)")
        if os.path.isfile(int8_file):
            zf.write(int8_file, arcname="model_int8.pt")
            print(f"  [+] Dynamic INT8 Weights: {int8_file} ({os.path.getsize(int8_file) / (1024**2):.1f} MB)")
        if os.path.isfile(tok_file):
            zf.write(tok_file, arcname="tokenizer.json")
            print(f"  [+] Tokenizer:            {tok_file}")
        if os.path.isfile(cfg_file):
            zf.write(cfg_file, arcname="config.yaml")
            print(f"  [+] Config:               {cfg_file}")

    print(f"\n[OK] Package created successfully: {output_zip} ({os.path.getsize(output_zip) / (1024**2):.1f} MB)")
    print(">> Download 'tunisiallm_refined_model.zip' from the Kaggle Output section in the right sidebar!")
    print("=" * 80)

if __name__ == "__main__":
    main()
