"""
04_pretrain.py - Distributed Pretraining Pipeline for TunisiaLLM
Features:
  - DDP over 2x T4 GPUs via `torchrun` with automatic single-GPU fallback
  - fp16 autocast + GradScaler for fast Tensor Core acceleration
  - Memory-mapped zero-overhead dataloader over uint16 .bin shards
  - Cosine learning rate schedule with linear warmup
  - Gradient accumulation and gradient clipping (1.0)
  - Wall-clock periodic checkpointing (~20 mins) & auto-resume from latest checkpoint
  - Graceful session stop (~11h) to prevent Kaggle 12h timeouts
  - Validation loss, perplexity logging, and sample text generations
"""

import os
import sys

# Optimize PyTorch CUDA memory allocator against fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import time
import math
import random
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import load_config, set_seed, format_time, format_num_params, format_bytes
import importlib
model_mod = importlib.import_module("03_model")
ModelConfig = model_mod.ModelConfig
TunisiaLLM = model_mod.TunisiaLLM


# ==============================================================================
# Memory-Mapped Dataset Loader
# ==============================================================================

class ShardedDataset:
    """
    Zero-RAM memory-mapped dataloader over uint16 .bin shards.
    Tracks precise token offset and shard index for deterministic resume.
    """
    def __init__(
        self,
        shard_paths: List[str],
        seq_len: int = 1024,
        rank: int = 0,
        world_size: int = 1,
        initial_shard_idx: int = 0,
        initial_offset: int = 0
    ):
        if not shard_paths:
            raise FileNotFoundError("No .bin shard files found in data directory!")

        self.shard_paths = sorted(shard_paths)
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

        self.current_shard_idx = initial_shard_idx % len(self.shard_paths)
        self.current_offset = initial_offset
        self.epoch = 0

        self._load_current_shard()

    def _load_current_shard(self) -> None:
        """Load current shard as a memory map."""
        p = self.shard_paths[self.current_shard_idx]
        self.mmap = np.memmap(p, dtype=np.uint16, mode='r')
        self.shard_tokens = len(self.mmap)

    def state_dict(self) -> Dict:
        """Return dataset state for checkpointing."""
        return {
            "current_shard_idx": self.current_shard_idx,
            "current_offset": self.current_offset,
            "epoch": self.epoch
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore dataset state upon resume."""
        self.current_shard_idx = state.get("current_shard_idx", 0) % len(self.shard_paths)
        self.current_offset = state.get("current_offset", 0)
        self.epoch = state.get("epoch", 0)
        self._load_current_shard()

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch next batch of (inputs, targets) of shape (batch_size, seq_len).
        Strided across DDP ranks to ensure each GPU sees unique data.
        Handles wrap-around and small shards seamlessly.
        """
        tokens_per_seq = self.seq_len + 1  # input + target shifted
        rank_tokens = batch_size * tokens_per_seq
        total_batch_tokens = rank_tokens * self.world_size

        # If shard is smaller than required batch, tile shard data
        if self.shard_tokens < total_batch_tokens:
            repeats = int(math.ceil(total_batch_tokens / max(1, self.shard_tokens))) + 1
            tiled = np.tile(self.mmap[:], repeats)
            start = self.rank * rank_tokens
            raw_slice = tiled[start:start + rank_tokens].astype(np.int64)
            self.current_offset = (self.current_offset + total_batch_tokens) % max(1, self.shard_tokens)
        else:
            # Advance to next shard if remaining tokens are insufficient
            if self.current_offset + total_batch_tokens >= self.shard_tokens:
                self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_paths)
                if self.current_shard_idx == 0:
                    self.epoch += 1
                self.current_offset = 0
                self._load_current_shard()

            start = self.current_offset + self.rank * rank_tokens
            end = start + rank_tokens
            raw_slice = self.mmap[start:end].astype(np.int64)
            self.current_offset += total_batch_tokens

        # Reshape into (batch_size, seq_len + 1)
        data = torch.from_numpy(raw_slice).view(batch_size, tokens_per_seq).to(device, non_blocking=True)
        inputs = data[:, :-1].contiguous()
        targets = data[:, 1:].contiguous()
        return inputs, targets


# ==============================================================================
# Learning Rate Scheduler: Cosine with Linear Warmup
# ==============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return lr * (step + 1) / (warmup_steps + 1)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizers(model: nn.Module, weight_decay: float, lr: float, betas: Tuple[float, float], eps: float) -> torch.optim.AdamW:
    """
    Separate 2D weights (decayed) from 1D biases and layernorm weights (undecayed).
    """
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps)


# ==============================================================================
# Text Generation for Periodic Qualitative Monitoring
# ==============================================================================

@torch.no_grad()
def sample_generation(
    model: nn.Module,
    tokenizer_path: str,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 40,
    temperature: float = 0.8
) -> List[str]:
    """Generate sample completions from given prompts using temperature sampling."""
    from tokenizers import Tokenizer
    if not os.path.isfile(tokenizer_path):
        return ["(Tokenizer not found)"]

    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")

    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    raw_model.reset_cache()

    completions = []
    for p in prompts:
        input_ids = tokenizer.encode(p).ids
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Prefill prompt into KV cache
        logits, _ = raw_model(x, use_cache=True, start_pos=0)
        cur_pos = len(input_ids)
        out_ids = list(input_ids)

        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_token_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()

            out_ids.append(next_tok)
            if next_tok == eos_id:
                break

            # Autoregressive single-step forward
            tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            logits, _ = raw_model(tok_tensor, use_cache=True, start_pos=cur_pos)
            cur_pos += 1

        raw_model.reset_cache()
        text = tokenizer.decode(out_ids, skip_special_tokens=False)
        completions.append(text)

    raw_model.train()
    return completions


# ==============================================================================
# Validation Evaluation
# ==============================================================================

@torch.no_grad()
def evaluate_val_loss(
    model: nn.Module,
    val_dataset: ShardedDataset,
    eval_batches: int,
    batch_size: int,
    device: torch.device,
    use_fp16: bool
) -> Tuple[float, float]:
    """Evaluate average cross-entropy loss and perplexity on held-out validation set."""
    model.eval()
    total_loss = 0.0

    for _ in range(eval_batches):
        inputs, targets = val_dataset.get_batch(batch_size, device)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=use_fp16):
            _, loss = model(inputs, targets=targets)
        total_loss += loss.item()

    model.train()
    avg_loss = total_loss / max(1, eval_batches)
    ppl = math.exp(min(avg_loss, 20.0))  # Cap exponent to avoid overflow
    return avg_loss, ppl


# ==============================================================================
# Pretraining Orchestrator
# ==============================================================================

def train_pretrain(
    cfg_path: Optional[str] = None,
    smoke_test: Optional[bool] = None,
    max_steps_override: Optional[int] = None,
    max_session_hours_override: Optional[float] = None
) -> None:
    """Main pretraining entry point."""
    cfg = load_config(cfg_path, smoke_test_override=smoke_test)
    is_smoke = cfg.get("smoke_test", False)

    # 1. Initialize Distributed Environment (DDP or Single-GPU)
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

    if master_process:
        print("=" * 80)
        print(f"TunisiaLLM Pretraining Pipeline")
        print(f"Mode: {'SMOKE TEST' if is_smoke else 'FULL PRETRAIN'}")
        print(f"Distributed: DDP (World Size = {world_size})" if ddp else f"Single Device: {device}")
        print(f"Output Checkpoints Dir: {cfg['paths']['checkpoint_dir']}")
        print("=" * 80)

    # 2. Training Settings & Smoke Test Config
    pt_cfg = cfg["pretrain"]
    smoke_cfg = pt_cfg.get("smoke_test", {}) if is_smoke else {}

    # 3. Model Initialization
    m_cfg = cfg["model"]
    preset = m_cfg.get("preset", "80M")
    seq_len = smoke_cfg.get("max_seq_len", m_cfg.get("max_seq_len", 1024))

    model_config = ModelConfig.from_preset(
        preset,
        vocab_size=cfg["tokenizer"]["vocab_size"],
        max_seq_len=seq_len
    )
    model_config.gradient_checkpointing = pt_cfg.get("gradient_checkpointing", False)

    model = TunisiaLLM(model_config).to(device)
    raw_model = model

    # Optional torch.compile
    if cfg["pretrain"].get("compile", False) and hasattr(torch, "compile"):
        if master_process:
            print("[*] Compiling model with torch.compile...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        raw_model = model.module

    if master_process:
        p_stats = raw_model.get_num_params()
        print(f"[+] Model Architecture: {preset}")
        print(f"    Total Parameters: {format_num_params(p_stats['total_parameters'])}")
        print(f"    Unique Trainable: {format_num_params(p_stats['unique_trainable_parameters'])}")
        print(f"    Tied Savings:     {format_num_params(p_stats['tied_savings'])}")

    # 4. Optimizer & Scaler
    batch_size = smoke_cfg.get("batch_size", pt_cfg["batch_size"])
    grad_accum_steps = smoke_cfg.get("gradient_accumulation_steps", pt_cfg["gradient_accumulation_steps"])
    max_steps = max_steps_override if max_steps_override is not None else smoke_cfg.get("max_steps", pt_cfg["max_steps"])
    warmup_steps = smoke_cfg.get("warmup_steps", pt_cfg["warmup_steps"])
    eval_interval = smoke_cfg.get("eval_interval_steps", pt_cfg["eval_interval_steps"])
    sample_interval = smoke_cfg.get("sample_interval_steps", pt_cfg["sample_interval_steps"])

    peak_lr = pt_cfg["learning_rate"]
    min_lr = pt_cfg["min_learning_rate"]
    use_fp16 = pt_cfg.get("fp16", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    optimizer = configure_optimizers(
        raw_model,
        weight_decay=pt_cfg["weight_decay"],
        lr=peak_lr,
        betas=(pt_cfg["adam_beta1"], pt_cfg["adam_beta2"]),
        eps=pt_cfg["adam_eps"]
    )

    # 4. Data Shards & Memory-Mapped Loader
    data_dir = cfg["paths"]["data_dir"]
    train_shards = glob.glob(os.path.join(data_dir, "train_shard_*.bin"))
    val_shards = glob.glob(os.path.join(data_dir, "val_shard_*.bin"))

    # Auto-generate if missing
    if not train_shards or not os.path.isfile(cfg["paths"]["tokenizer_path"]):
        if master_process:
            print("[!] Training shards missing. Triggering data prep & tokenizer training...")
            import importlib
            tok_mod = importlib.import_module("02_tokenizer")
            tok_mod.run_tokenizer_pipeline(cfg_path, smoke_test=is_smoke)
        if ddp:
            dist.barrier()
        train_shards = glob.glob(os.path.join(data_dir, "train_shard_*.bin"))
        val_shards = glob.glob(os.path.join(data_dir, "val_shard_*.bin"))

    train_dataset = ShardedDataset(
        train_shards,
        seq_len=model_config.max_seq_len,
        rank=rank,
        world_size=world_size
    )

    val_dataset = ShardedDataset(
        val_shards if val_shards else train_shards,
        seq_len=model_config.max_seq_len,
        rank=0,
        world_size=1
    )

    # 5. Checkpoint Auto-Resume Logic
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

    start_step = 0
    best_val_loss = float("inf")

    if os.path.isfile(latest_ckpt_path):
        if master_process:
            print(f"\n[*] Found existing checkpoint: {latest_ckpt_path}. Auto-resuming...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and use_fp16:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "dataset_state" in ckpt:
            train_dataset.load_state_dict(ckpt["dataset_state"])
        if "rng_state" in ckpt:
            random.setstate(ckpt["rng_state"]["python"])
            np.random.set_state(ckpt["rng_state"]["numpy"])
            torch.set_rng_state(ckpt["rng_state"]["torch"])
            if torch.cuda.is_available() and "cuda" in ckpt["rng_state"]:
                torch.cuda.set_rng_state(ckpt["rng_state"]["cuda"])
        if master_process:
            print(f"[OK] Resumed successfully from step {start_step} (Best Val Loss: {best_val_loss:.4f})")

    # 6. Training Loop Setup
    tokens_per_step = batch_size * model_config.max_seq_len * grad_accum_steps * world_size
    save_interval_secs = pt_cfg.get("save_interval_mins", 20) * 60
    session_hours = max_session_hours_override if max_session_hours_override is not None else pt_cfg.get("max_session_hours", 11.0)
    max_session_secs = session_hours * 3600

    session_start_time = time.time()
    last_save_time = time.time()
    last_log_time = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    test_prompts = [
        "عسلامة يا باهي",
        "الطقس في تونس",
        "شنوة أحوالك اليوم",
        "تاريخ قرطاج وتونس"
    ]

    if master_process:
        print(f"\n[*] Starting training loop from step {start_step} to {max_steps}...")
        print(f"    Tokens per optimizer step: {tokens_per_step:,}")
        print(f"    Session timeout safeguard: {max_session_secs / 3600:.1f} hours")
        print("-" * 80)

    for step in range(start_step, max_steps):
        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, peak_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        accum_loss = 0.0

        # Gradient Accumulation Loop
        for micro_step in range(grad_accum_steps):
            inputs, targets = train_dataset.get_batch(batch_size, device)

            # In DDP, synchronize gradients only on the final micro-step
            if ddp and micro_step < grad_accum_steps - 1:
                ctx = model.no_sync()
            else:
                ctx = torch.enable_grad()

            with ctx:
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=use_fp16):
                    _, loss = model(inputs, targets=targets)
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
                accum_loss += loss.item()

        # Gradient clipping and optimizer step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), pt_cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # 7. Logging & Metrics
        now = time.time()
        log_every = 1 if is_smoke else 10
        if master_process and (step % log_every == 0 or step == max_steps - 1):
            dt = now - last_log_time
            tps = (tokens_per_step * log_every) / max(dt, 1e-5)
            last_log_time = now
            vram_str = f" | VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.1f}GB" if torch.cuda.is_available() else ""
            print(
                f"Step {step:05d}/{max_steps:05d} | "
                f"Loss: {accum_loss:.4f} | "
                f"PPL: {math.exp(min(accum_loss, 20.0)):.2f} | "
                f"LR: {lr:.2e} | "
                f"Throughput: {int(tps):,} tok/s{vram_str} | "
                f"Elapsed: {format_time(now - session_start_time)}",
                flush=True
            )

        # 8. Periodic Validation
        if master_process and (step > 0 and step % eval_interval == 0 or step == max_steps - 1):
            eval_batches = 10 if is_smoke else 30
            val_loss, val_ppl = evaluate_val_loss(
                raw_model, val_dataset, eval_batches=eval_batches, batch_size=batch_size, device=device, use_fp16=use_fp16
            )
            print("\n" + "=" * 70)
            print(f"[EVAL @ Step {step}] Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")
            print("=" * 70 + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": raw_model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": cfg
                }, best_ckpt_path)
                print(f"[+] Saved new best model checkpoint to {best_ckpt_path}")

        # 9. Periodic Sample Text Generations
        if master_process and (step > 0 and step % sample_interval == 0 or step == max_steps - 1):
            print("\n--- [Sample Generation Inspection] ---")
            samples = sample_generation(
                raw_model,
                tokenizer_path=cfg["paths"]["tokenizer_path"],
                prompts=test_prompts[:2],
                device=device,
                max_new_tokens=30
            )
            for smp in samples:
                print(f">>> {smp}\n")
            print("--------------------------------------\n")

        # 10. Wall-Clock Checkpointing (~20 minutes)
        if master_process and (now - last_save_time >= save_interval_secs or step == max_steps - 1):
            last_save_time = now
            ckpt_state = {
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_fp16 else None,
                "dataset_state": train_dataset.state_dict(),
                "best_val_loss": best_val_loss,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                },
                "config": cfg
            }
            # Save periodic step checkpoint
            step_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pt")
            torch.save(ckpt_state, step_ckpt_path)
            # Save latest checkpoint for automatic resumption
            torch.save(ckpt_state, latest_ckpt_path)
            print(f"\n[+] Checkpoint saved at step {step} to {latest_ckpt_path}")

            # Prune old checkpoints to save disk space
            all_step_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt")))
            keep_n = cfg["pretrain"].get("keep_last_n_checkpoints", 3)
            if len(all_step_ckpts) > keep_n:
                for old_ckpt in all_step_ckpts[:-keep_n]:
                    try:
                        os.remove(old_ckpt)
                    except OSError:
                        pass

        # 11. Graceful Session Stop before Kaggle 12h Limit
        if (now - session_start_time) >= max_session_secs:
            if master_process:
                print("\n" + "!" * 80)
                print(f"[!] Session time limit reached ({format_time(now - session_start_time)}).")
                print("    Saving graceful exit checkpoint and terminating for SFT...")
                exit_ckpt_state = {
                    "step": step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_fp16 else None,
                    "dataset_state": train_dataset.state_dict(),
                    "best_val_loss": best_val_loss,
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    },
                    "config": cfg
                }
                end_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_session_end.pt")
                torch.save(exit_ckpt_state, end_ckpt_path)
                torch.save(exit_ckpt_state, latest_ckpt_path)
                best_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
                if not os.path.isfile(best_ckpt_path):
                    torch.save(exit_ckpt_state, best_ckpt_path)
                print(f"    Saved: {end_ckpt_path}")
                print("!" * 80 + "\n")
            if ddp:
                dist.destroy_process_group()
            sys.exit(0)

    if master_process:
        print("\n" + "=" * 80)
        print(f"[OK] TunisiaLLM Pretraining Completed Successfully! ({max_steps} steps reached)")
        print(f"     Latest checkpoint: {latest_ckpt_path}")
        print("=" * 80)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TunisiaLLM Pretraining")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny smoke test (200 steps)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--max-session-hours", type=float, default=None, help="Override session duration in hours")
    args = parser.parse_args()

    train_pretrain(
        args.config,
        smoke_test=args.smoke_test if args.smoke_test else None,
        max_steps_override=args.max_steps,
        max_session_hours_override=args.max_session_hours
    )
