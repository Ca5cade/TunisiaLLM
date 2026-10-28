"""
06_sft.py - Supervised Fine-Tuning (SFT) for Tunisian Derja Chat LLM
Continues from pretrained checkpoint with low learning rate (2e-5 - 5e-5),
trains strictly on assistant tokens (loss masking with ignore_index=-100),
supports DDP/single-GPU, fp16 GradScaler, 20-min checkpointing, and auto-resume.
"""

import os
import sys
import glob
import time
import math
import random
import argparse
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import load_config, set_seed, format_time, format_num_params, format_bytes

import importlib
model_mod = importlib.import_module("03_model")
ModelConfig = model_mod.ModelConfig
TunisiaLLM = model_mod.TunisiaLLM


# ==============================================================================
# SFT Tensor Dataset
# ==============================================================================

class SFTDataset(Dataset):
    """In-memory PyTorch dataset containing pre-tokenized inputs and masked labels."""
    def __init__(self, data_path: str):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"SFT tensor file not found at: {data_path}")
        d = torch.load(data_path, map_location="cpu")
        self.input_ids = d["input_ids"]
        self.labels = d["labels"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


# ==============================================================================
# Learning Rate Scheduler: Cosine with Warmup
# ==============================================================================

def get_cosine_lr(step: int, warmup_steps: int, total_steps: int, lr: float, min_lr: float) -> float:
    """Compute cosine learning rate with linear warmup."""
    if step < warmup_steps:
        return lr * (step + 1) / (warmup_steps + 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# ==============================================================================
# Sample Chat Generation for SFT Validation
# ==============================================================================

@torch.no_grad()
def evaluate_sft_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    max_batches: int = 30
) -> Tuple[float, float]:
    """Compute loss and perplexity on assistant tokens over validation set."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i, (inputs, labels) in enumerate(val_loader):
        if i >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        shift_inputs = inputs[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=use_fp16):
            _, loss = model(shift_inputs, targets=shift_labels)
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item()
            total_batches += 1

    model.train()
    avg_loss = total_loss / max(1, total_batches)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


@torch.no_grad()
def sample_sft_chat(
    model: nn.Module,
    tokenizer_path: str,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 50
) -> List[str]:
    """Generate assistant responses formatted with chat special tokens."""
    from tokenizers import Tokenizer
    if not os.path.isfile(tokenizer_path):
        return ["(Tokenizer not found)"]

    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    raw_model.reset_cache()

    responses = []
    for user_prompt in prompts:
        formatted_prompt = f"<bos><|system|>\nإنت مساعد ذكي بالدارجة التونسية.<|user|>\n{user_prompt}\n<|assistant|>\n"
        input_ids = tokenizer.encode(formatted_prompt).ids
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        logits, _ = raw_model(x, use_cache=True, start_pos=0)
        cur_pos = len(input_ids)
        out_ids = []

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / 0.7
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()

            if next_tok == eos_id:
                break
            out_ids.append(next_tok)

            tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            logits, _ = raw_model(tok_tensor, use_cache=True, start_pos=cur_pos)
            cur_pos += 1

        raw_model.reset_cache()
        reply = tokenizer.decode(out_ids, skip_special_tokens=False)
        responses.append(reply.strip())

    raw_model.train()
    return responses


# ==============================================================================
# SFT Training Pipeline
# ==============================================================================

def train_sft(
    cfg_path: Optional[str] = None,
    smoke_test: Optional[bool] = None,
    fresh: bool = False,
    max_steps_override: Optional[int] = None,
    init_checkpoint: Optional[str] = None,
    lr_override: Optional[float] = None,
    epochs_override: Optional[int] = None
) -> None:
    """Run Supervised Fine-Tuning."""
    cfg = load_config(cfg_path, smoke_test_override=smoke_test)
    is_smoke = cfg.get("smoke_test", False)

    # 1. Distributed Setup
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master_process = (rank == 0)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        master_process = True

    set_seed(cfg.get("seed", 42) + rank)

    sft_cfg = cfg["sft"]
    smoke_sft_cfg = sft_cfg.get("smoke_test", {}) if is_smoke else {}

    batch_size = smoke_sft_cfg.get("batch_size", sft_cfg["batch_size"])
    grad_accum_steps = smoke_sft_cfg.get("gradient_accumulation_steps", sft_cfg["gradient_accumulation_steps"])
    epochs = epochs_override if epochs_override is not None else smoke_sft_cfg.get("epochs", sft_cfg["epochs"])
    eval_interval = smoke_sft_cfg.get("eval_interval_steps", sft_cfg["eval_interval_steps"])

    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    sft_data_dir = cfg["paths"]["sft_data_dir"]
    train_pt = os.path.join(sft_data_dir, "sft_train.pt")
    val_pt = os.path.join(sft_data_dir, "sft_val.pt")

    if master_process:
        print("=" * 80)
        print("TunisiaLLM Instruction Tuning (SFT) Pipeline")
        print(f"Mode: {'SMOKE TEST' if is_smoke else 'FULL SFT'}")
        print(f"Device: {device} | DDP: {ddp} (World Size = {world_size})")
        print(f"Checkpoint Directory: {checkpoint_dir}")
        if init_checkpoint:
            print(f"Base Checkpoint: {init_checkpoint}")
        print("=" * 80)

    # Auto-generate SFT data if missing
    if not os.path.isfile(train_pt) or not os.path.isfile(val_pt):
        if master_process:
            print("[!] SFT dataset tensors missing. Generating SFT data...")
            import importlib
            sft_gen_mod = importlib.import_module("05_make_sft_data")
            sft_gen_mod.build_sft_dataset(cfg_path, smoke_test=is_smoke)
        if ddp:
            dist.barrier()

    # 2. Load Dataset & Dataloaders
    train_dataset = SFTDataset(train_pt)
    val_dataset = SFTDataset(val_pt)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    total_steps = len(train_loader) * epochs // grad_accum_steps
    max_steps_cfg = max_steps_override if max_steps_override is not None else smoke_sft_cfg.get("max_steps", sft_cfg.get("max_steps", None))
    if max_steps_cfg:
        total_steps = min(total_steps, max_steps_cfg)
    warmup_steps = int(total_steps * sft_cfg.get("warmup_ratio", 0.05))

    # 3. Model Initialization
    m_cfg = cfg["model"]
    preset = m_cfg.get("preset", "80M")
    seq_len = train_dataset.input_ids.shape[1]

    model_config = ModelConfig.from_preset(
        preset,
        vocab_size=cfg["tokenizer"]["vocab_size"],
        max_seq_len=seq_len
    )

    model = TunisiaLLM(model_config).to(device)
    raw_model = model

    # 4. Load Pretrained Weights or Resume SFT
    sft_latest_ckpt = os.path.join(checkpoint_dir, "sft_checkpoint_latest.pt")
    pretrained_best = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    pretrained_latest = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

    start_epoch = 0
    start_step = 0
    best_val_loss = float("inf")

    custom_ckpt_path = None
    if init_checkpoint:
        for p in [
            init_checkpoint,
            os.path.join(checkpoint_dir, init_checkpoint),
            os.path.join("tunisiallm_model", init_checkpoint),
            os.path.join("checkpoints", init_checkpoint)
        ]:
            if os.path.isfile(p):
                custom_ckpt_path = p
                break
        if custom_ckpt_path is None:
            raise FileNotFoundError(f"Specified checkpoint not found: {init_checkpoint}")

    if custom_ckpt_path is not None:
        if master_process:
            print(f"[*] Initializing model weights from specified checkpoint: {custom_ckpt_path}")
        ckpt = torch.load(custom_ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        raw_model.load_state_dict(state_dict, strict=False)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
    elif not fresh and os.path.isfile(sft_latest_ckpt):
        if master_process:
            print(f"[*] Resuming from ongoing SFT checkpoint: {sft_latest_ckpt}")
        ckpt = torch.load(sft_latest_ckpt, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
    elif os.path.isfile(pretrained_best):
        if master_process:
            print(f"[*] Initializing SFT from best pretrained checkpoint: {pretrained_best}")
        ckpt = torch.load(pretrained_best, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif os.path.isfile(pretrained_latest):
        if master_process:
            print(f"[*] Initializing SFT from latest pretrained checkpoint: {pretrained_latest}")
        ckpt = torch.load(pretrained_latest, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        if master_process:
            print("[!] Warning: No pretrained checkpoint found. Initializing from scratch for test...")

    if ddp:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        raw_model = model.module

    # 5. Optimizer & Scaler
    peak_lr = lr_override if lr_override is not None else sft_cfg["learning_rate"]
    min_lr = (peak_lr * 0.1) if lr_override is not None else sft_cfg["min_learning_rate"]
    use_fp16 = sft_cfg.get("fp16", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=peak_lr,
        weight_decay=sft_cfg.get("weight_decay", 0.01)
    )

    # 6. SFT Training Loop
    session_start_time = time.time()
    last_save_time = time.time()
    last_log_time = time.time()
    save_interval_secs = sft_cfg.get("save_interval_mins", 20) * 60
    max_session_secs = sft_cfg.get("max_session_hours", 11.0) * 3600

    test_chat_prompts = [
        "عسلامة، شنوة حوالك اليوم؟",
        "قداش 15 مع 30 إلا 5؟",
        "وين نلقى أحسن لبلابي في تونس؟",
        "شكون باش يربح كأس إفريقيا 2034؟"
    ]

    global_step = start_step

    if master_process:
        print(f"\n[*] Starting SFT Training ({epochs} epochs, {total_steps} total optimizer steps)...")
        print(f"    Learning Rate: {peak_lr:.2e} -> {min_lr:.2e}")
        print(f"    Assistant-only loss masking enabled (ignore_index=-100)")
        print("-" * 80)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        accum_loss = 0.0
        micro_step = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            shift_inputs = inputs[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            if ddp and (micro_step % grad_accum_steps != grad_accum_steps - 1):
                ctx = model.no_sync()
            else:
                ctx = torch.enable_grad()

            with ctx:
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=use_fp16):
                    _, loss = model(shift_inputs, targets=shift_labels)
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), sft_cfg["grad_clip"])

                lr = get_cosine_lr(global_step, warmup_steps, total_steps, peak_lr, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                now = time.time()

                # Logging
                log_every = 1 if is_smoke else 10
                if master_process and (global_step % log_every == 0):
                    vram_str = f" | VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.1f}GB" if torch.cuda.is_available() else ""
                    print(
                        f"Epoch {epoch+1:02d}/{epochs:02d} | Step {global_step:04d}/{total_steps:04d} | "
                        f"Loss: {accum_loss:.4f} | "
                        f"PPL: {math.exp(min(accum_loss, 20.0)):.2f} | "
                        f"LR: {lr:.2e} | Norm: {grad_norm:.2f}{vram_str} | "
                        f"Elapsed: {format_time(now - session_start_time)}",
                        flush=True
                    )
                accum_loss = 0.0

                # Validation & Qualitative Sampling
                if master_process and (global_step > 0 and global_step % eval_interval == 0):
                    val_loss, val_ppl = evaluate_sft_validation(raw_model, val_loader, device=device, use_fp16=use_fp16)
                    print("\n" + "=" * 70)
                    print(f"[SFT EVAL @ Step {global_step}] Assistant Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
                    print("=" * 70)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_sft_path = os.path.join(checkpoint_dir, "sft_checkpoint_best.pt")
                        torch.save({
                            "step": global_step,
                            "epoch": epoch,
                            "model_state_dict": raw_model.state_dict(),
                            "best_val_loss": best_val_loss,
                            "config": cfg
                        }, best_sft_path)
                        print(f"[+] Saved new best SFT checkpoint to {best_sft_path}")

                    print("\n--- [Sample SFT Chat Inspection] ---")
                    samples = sample_sft_chat(
                        raw_model,
                        tokenizer_path=cfg["paths"]["tokenizer_path"],
                        prompts=test_chat_prompts[:2],
                        device=device,
                        max_new_tokens=40
                    )
                    for q, ans in zip(test_chat_prompts[:2], samples):
                        print(f"User:      {q}")
                        print(f"Assistant: {ans}\n")
                    print("------------------------------------\n")

                # Periodic Checkpointing (~20 min)
                if master_process and (now - last_save_time >= save_interval_secs):
                    last_save_time = now
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if use_fp16 else None,
                        "best_val_loss": best_val_loss,
                        "config": cfg
                    }, sft_latest_ckpt)
                    print(f"[+] Periodic SFT checkpoint saved at step {global_step} to {sft_latest_ckpt}")

                # Graceful 11h Kaggle Session Exit
                if (now - session_start_time) >= max_session_secs:
                    if master_process:
                        print(f"\n[!] 11h session safeguard reached. Saving SFT checkpoint and exiting cleanly...")
                        torch.save({
                            "step": global_step,
                            "epoch": epoch,
                            "model_state_dict": raw_model.state_dict(),
                            "best_val_loss": best_val_loss,
                            "config": cfg
                        }, os.path.join(checkpoint_dir, "sft_checkpoint_session_end.pt"))
                    if ddp:
                        dist.destroy_process_group()
                    sys.exit(0)

                # Check early stop when max_steps is reached
                if max_steps_cfg is not None and global_step >= max_steps_cfg:
                    if master_process:
                        print(f"\n[+] SFT max steps ({max_steps_cfg}) reached. Finishing successfully...")
                    break
        if max_steps_cfg is not None and global_step >= max_steps_cfg:
            break

    if master_process:
        final_sft_path = os.path.join(checkpoint_dir, "sft_checkpoint_final.pt")
        final_state = {
            "step": global_step,
            "epoch": epochs,
            "model_state_dict": raw_model.state_dict(),
            "best_val_loss": best_val_loss,
            "config": cfg
        }
        torch.save(final_state, final_sft_path)
        best_sft_path = os.path.join(checkpoint_dir, "sft_checkpoint_best.pt")
        if not os.path.isfile(best_sft_path):
            torch.save(final_state, best_sft_path)
        print("\n" + "=" * 80)
        print(f"[OK] SFT Instruction Tuning Completed! Final checkpoint saved to: {final_sft_path}")
        print("=" * 80)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TunisiaLLM SFT Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt) to initialize weights from (e.g. tunisiallm_model/sft_checkpoint_best.pt)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate (e.g. 2e-5 for continued fine-tuning)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny smoke test (quick verification)")
    parser.add_argument("--fresh", action="store_true", help="Start SFT fresh from pretrained checkpoint_best.pt")
    parser.add_argument("--max-steps", type=int, default=None, help="Override SFT max steps")
    args = parser.parse_args()

    train_sft(
        args.config,
        smoke_test=args.smoke_test if args.smoke_test else None,
        fresh=args.fresh,
        max_steps_override=args.max_steps,
        init_checkpoint=args.checkpoint,
        lr_override=args.lr,
        epochs_override=args.epochs
    )
