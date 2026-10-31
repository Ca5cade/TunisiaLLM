"""
08_dpo.py - Direct Preference Optimization (DPO) for TunisiaLLM
Aligns TunisiaLLM by penalizing:
  1. Hallucinated math / gibberish outputs
  2. Mode-collapsed greetings when asked technical/factual questions
  3. Spurious/irrelevant knowledge injection
And rewarding:
  1. Accurate tool calling and clean currency conversions
  2. Direct code & technical explanations in Tunisian Derja
  3. High-quality multi-turn and Arabizi answers
"""

import os
import sys
import math
import time
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import importlib

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
model_mod = importlib.import_module("03_model")
ModelConfig = model_mod.ModelConfig
TunisiaLLM = model_mod.TunisiaLLM

from utils import load_config, format_time, format_bytes

# ==============================================================================
# 1. DPO Preference Pair Synthesis & Dataset
# ==============================================================================

def generate_dpo_pairs() -> List[Dict[str, str]]:
    """Synthesize high-contrast preference pairs specifically targeting known failure modes."""
    rng = random.Random(42)
    pairs = []

    # Domain 1: Shopping Calculations (Tool Calling vs Hallucinations)
    items = [
        ("طماطم", "كيلو", 1500),
        ("بطاطا", "كيلو", 1200),
        ("بصل", "كيلو", 800),
        ("تفاح", "كيلو", 2500),
        ("حليب", "باكوات", 1400)
    ]
    for it, un, pr in items:
        for qty in [2, 3, 4, 5]:
            res = qty * pr
            d = res // 1000
            m = res % 1000
            q = f"شريت {qty} {un} {it} الكيلو بـ {pr} مليم، قداش يطيحو؟"
            chosen = f"<tool_call>calc: {qty} * {pr}</tool_call><tool_result>{res}</tool_result>يطيحو {res} مليم بالتمام (يعني {d} دينارات و {m} مليم) يا باهي."
            rejected = f"<tool_call>calc: {qty} * {pr}</tool_call><tool_result>{res}</tool_result>الماء في حال سرعته قليلة 500 دينار يطيحو المنتوج rises."
            pairs.append({"prompt": q, "chosen": chosen, "rejected": rejected, "category": "shopping_math"})

    # Domain 2: Coding Instructions (Code Answer vs Empty Greeting)
    coding_cases = [
        (
            "كيفاش نعمل حلقة تكرار for في بايثون؟",
            "باش تعمل حلقة تكرار for في بايثون، تستعمل range():\n```python\nfor i in range(5):\n    print(i)\n```\nالكود هذا يدور 5 مرات ويطبع الأرقام من 0 لـ 4 يا باهي.",
            "Ahla bik ya ghali! Labes el 7amdoullah, nchallah inti zeda cava w mrigel."
        ),
        (
            "بربي كيفاش نعمل دالة (function) في بايثون؟",
            "باش تصنع دالة في بايثون، تستعمل def هكا:\n```python\ndef ejma3(a, b):\n    return a + b\n```\nالكود هذا يرجع مجموع زوز أرقام.",
            "أهلا وسهلاً بيك! نهارك زين وسعيد إن شاء الله. تفضل شتحب نساعفك؟"
        ),
        (
            "كيفاش نزيد عنصر في list في بايثون؟",
            "تستعمل append() هكا:\n```python\nqayma = ['تفاح', 'بنان']\nqayma.append('برتقال')\n```\nوهكا تولي القائمة فيها 3 عناصر.",
            "الحمد لله ونشكروه، ديما في الخدمة بكل فرحة، شنوة سؤالك اليوم؟"
        ),
        (
            "شنوة الفرق بين list و tuple في بايثون؟",
            "الـ List تتكتب بـ `[]` وتنجم تبدلها وتزيد فيها (mutable)، بينما الـ Tuple تتكتب بـ `()` وما تتبدلش بعد الصنع (immutable).",
            "Bonjour! Ça va très bien merci, chnouwa najjem n3awnek bih lyoum?"
        )
    ]
    for q, ch, rj in coding_cases:
        for _ in range(4):
            pairs.append({"prompt": q, "chosen": ch, "rejected": rj, "category": "coding"})

    # Domain 3: Arabizi Code-Switching (Helpful Derja vs Irrelevant Greeting)
    arabizi_cases = [
        (
            "brabi kifech na3mel boucle for fi python?",
            "Bach ta3mel boucle for fi python, testa3mel range() hakka:\n```python\nfor i in range(5):\n    print(i)\n```\nEl code hedha ydour 5 marrat w yetba3 men 0 l 4 ya bahi.",
            "Bonjour cava el 7amdoullah kol chay mrigel."
        ),
        (
            "chrit 3 kilo tmatem b 1500 mlim 9adech iti7ou?",
            "<tool_call>calc: 3 * 1500</tool_call><tool_result>4500</tool_result>Yti7ou 4500 mlim (ya3ni 4 dinarat w 500 mlim) ya ghali.",
            "calc: 3 * 1500calc: 3 * 1500 el ma2 fi 7al."
        ),
        (
            "kifech nzid 7aja fi liste fi python?",
            "Testa3mel append() hakka:\n```python\nl = ['tounes', 'sfax']\nl.append('sousse')\n```\nha kka twalli fiha 3 7ajet.",
            "Labes el 7amdoullah w inti cava?"
        )
    ]
    for q, ch, rj in arabizi_cases:
        for _ in range(4):
            pairs.append({"prompt": q, "chosen": ch, "rejected": rj, "category": "arabizi"})

    # Domain 4: Travel & Tourism (Direct Recommendations vs Irrelevant Fact)
    travel_cases = [
        (
            "شنوة أحسن بلاصة نعمل فيها تصييفة في تونس؟",
            "في تونس ننصحك بالمهدية وقليبية لبحر صافي ورملة بيضاء، أو جربة للهدوء والنزل، أو الحمامات وسوسة للجو الشبابي والسهرات.",
            "تبلغ المساحة الجملية للجمهورية التونسية حوالي 163,610 كيلومتر مربع."
        ),
        (
            "وين تنصحني نحوس في تونس العاصمة؟",
            "في تونس العاصمة ننصحك بالمدينة العربي وجامع الزيتونة، سيدي بوسعيد وكعبة بمبالوني، وقرطاج ومتحف باردو.",
            "تصيبات في الصالة هذي هي أحسن بلاصة نعمل فيها تصيبة في الدار."
        )
    ]
    for q, ch, rj in travel_cases:
        for _ in range(4):
            pairs.append({"prompt": q, "chosen": ch, "rejected": rj, "category": "travel"})

    # Domain 5: Tunisian Food Differences (Explanation vs Loop)
    food_cases = [
        (
            "شنوة الفرق بين الكسكسي التونسي والمغربي؟",
            "الفرق الأساسي: الكسكسي التونسي يجي أحمر وحار بالصلصة والطماطم والهريسة وبالعصبان أو الحوت، بينما المغربي يجي أصفر بالزعفران ومعتدل وبالتفاية.",
            "الكسكسي بالعصبان يتقلى فيه الفلفل الطماطم يطيب يطيب يطيب ما نجمش نطيب."
        )
    ]
    for q, ch, rj in food_cases:
        for _ in range(5):
            pairs.append({"prompt": q, "chosen": ch, "rejected": rj, "category": "food"})

    rng.shuffle(pairs)
    print(f"[OK] Generated {len(pairs)} DPO alignment pairs.", flush=True)
    return pairs


def load_external_dpo_pairs(limit: int = 1500) -> List[Dict[str, str]]:
    """Load authentic Tunisian DPO preference dataset."""
    pairs = []
    try:
        from datasets import load_dataset
        print(f"[*] Fetching authentic Tunisian DPO preference dataset: wghezaiel/DPO-Tunisian-Derja (limit={limit})...")
        ds = load_dataset("wghezaiel/DPO-Tunisian-Derja", split=f"train[:{limit}]")
        for row in ds:
            p = row.get("prompt", "")
            c = row.get("chosen", "")
            r = row.get("rejected", "")
            if p and c and r and isinstance(p, str) and isinstance(c, str) and isinstance(r, str):
                p = p.strip()
                c = c.strip()
                r = r.strip()
                if len(p) >= 10 and len(c) >= 1 and len(r) >= 1:
                    pairs.append({
                        "prompt": p,
                        "chosen": c,
                        "rejected": r,
                        "category": "tunisian_dpo_corpus"
                    })
        print(f"  [+] Loaded {len(pairs):,} authentic DPO pairs from wghezaiel/DPO-Tunisian-Derja.", flush=True)
    except Exception as e:
        print(f"  [!] Note: External DPO dataset load skipped ({e}).", flush=True)
    return pairs


# ==============================================================================
# 2. Sequence Log-Probability Computation
# ==============================================================================

def compute_sequence_log_probs(
    model: nn.Module,
    prompt_ids: List[int],
    completion_ids: List[int],
    device: torch.device
) -> torch.Tensor:
    """
    Computes sum of log probabilities of completion tokens conditioned on prompt tokens.
    """
    full_ids = prompt_ids + completion_ids
    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    
    prompt_len = len(prompt_ids)
    comp_len = len(completion_ids)

    # Shift inputs and targets
    inputs = input_tensor[:, :-1]
    targets = input_tensor[:, 1:]

    logits, _ = model(inputs)
    log_probs = F.log_softmax(logits, dim=-1)

    # Targets for the completion span: start index is prompt_len - 1 in shifted targets
    comp_start = prompt_len - 1
    comp_end = comp_start + comp_len

    comp_targets = targets[:, comp_start:comp_end]
    comp_log_probs = log_probs[:, comp_start:comp_end, :]

    # Gather log prob of actual target tokens
    gathered = torch.gather(comp_log_probs, dim=-1, index=comp_targets.unsqueeze(-1)).squeeze(-1)
    return gathered.sum(dim=-1)


# ==============================================================================
# 3. DPO Training Engine
# ==============================================================================

def train_dpo(
    checkpoint_path: str,
    tokenizer_path: str,
    output_path: str,
    beta: float = 0.1,
    lr: float = 1.0e-5,
    steps: int = 40,
    device_str: Optional[str] = None
) -> None:
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80, flush=True)
    print("TUNISIALLM: DIRECT PREFERENCE OPTIMIZATION (DPO) ENGINE")
    print(f"Beta: {beta} | Learning Rate: {lr} | Steps: {steps}")
    print("=" * 80, flush=True)

    device = torch.device(device_str)
    tokenizer = Tokenizer.from_file(tokenizer_path)

    cfg = ModelConfig.from_preset("80M", vocab_size=16384, max_seq_len=256)

    # Load Policy Model
    print(f"[*] Loading Policy Model (pi_theta) from {checkpoint_path}...", flush=True)
    policy_model = TunisiaLLM(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    policy_model.load_state_dict(state, strict=False)
    policy_model.train()

    # Load Frozen Reference Model (pi_ref)
    print(f"[*] Loading Reference Model (pi_ref)...", flush=True)
    ref_model = TunisiaLLM(cfg).to(device)
    ref_model.load_state_dict(state, strict=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    pairs = generate_dpo_pairs()
    ext_dpo = load_external_dpo_pairs(limit=1500)
    if ext_dpo:
        pairs.extend(ext_dpo)
        random.Random(42).shuffle(pairs)
        print(f"[OK] Total combined DPO dataset size: {len(pairs):,} pairs", flush=True)
    sys_prompt = "إنت مساعد ذكي ولطيف بالدارجة التونسية، تعاون بوضوح وطلاقة وتجاوب ديما على قد السؤال."
    eos_str = "<eos>"

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=0.01)

    print("\n[*] Starting DPO Optimization loop...", flush=True)
    t0 = time.time()
    best_loss = float("inf")

    for step in range(steps):
        # Sample a preference pair
        pair = pairs[step % len(pairs)]
        prompt_text = f"<bos><|system|>\n{sys_prompt}\n<|user|>\n{pair['prompt']}\n<|assistant|>\n"
        chosen_text = f"{pair['chosen']}{eos_str}"
        rejected_text = f"{pair['rejected']}{eos_str}"

        p_ids = tokenizer.encode(prompt_text).ids
        ch_ids = tokenizer.encode(chosen_text).ids
        rj_ids = tokenizer.encode(rejected_text).ids

        # Truncate if exceeds max length
        max_l = 250
        if len(p_ids) + max(len(ch_ids), len(rj_ids)) > max_l:
            p_ids = p_ids[:max_l - max(len(ch_ids), len(rj_ids))]

        # 1. Compute reference model log-probs (no grad)
        with torch.no_grad():
            ref_logp_w = compute_sequence_log_probs(ref_model, p_ids, ch_ids, device)
            ref_logp_l = compute_sequence_log_probs(ref_model, p_ids, rj_ids, device)

        # 2. Compute policy model log-probs (with grad)
        policy_logp_w = compute_sequence_log_probs(policy_model, p_ids, ch_ids, device)
        policy_logp_l = compute_sequence_log_probs(policy_model, p_ids, rj_ids, device)

        # 3. DPO Loss & Margin
        # margin = beta * [(log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x))]
        pi_logr_w = policy_logp_w - ref_logp_w
        pi_logr_l = policy_logp_l - ref_logp_l
        margin = beta * (pi_logr_w - pi_logr_l)

        # Loss = -log sigmoid(margin) = softplus(-margin)
        loss = F.softplus(-margin).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        acc = (margin > 0).float().item()

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            speed = elapsed / (step + 1)
            print(f"  Step {step+1:02d}/{steps} | DPO Loss: {loss.item():.4f} | Margin: {margin.item():+.3f} | Accuracy: {acc*100:.0f}% | Speed: {speed:.2f}s/step", flush=True)

    elapsed_total = time.time() - t0
    print(f"\n[OK] DPO Optimization complete in {elapsed_total:.1f}s!", flush=True)

    # Save aligned checkpoint
    save_targets = [
        output_path,
        os.path.join(REPO_ROOT, "tunisiallm_model", "sft_checkpoint_best.pt"),
        os.path.join(REPO_ROOT, "checkpoints", "sft_checkpoint_best.pt")
    ]
    if os.path.isdir("/kaggle/working/checkpoints"):
        save_targets.append("/kaggle/working/checkpoints/sft_checkpoint_best.pt")
    aligned_ckpt = {
        "step": ckpt.get("step", 660) + steps,
        "epoch": 3,
        "model_state_dict": policy_model.state_dict(),
        "dpo_aligned": True,
        "config": ckpt.get("config", cfg)
    }
    for p in save_targets:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        torch.save(aligned_ckpt, p)
        print(f"[OK] Saved DPO-aligned model to: {p}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TunisiaLLM DPO Training")
    parser.add_argument("--checkpoint", type=str, default="sft_checkpoint_best.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--output", type=str, default="sft_checkpoint_best.pt")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_dpo(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        beta=args.beta,
        lr=args.lr,
        steps=args.steps,
        device_str=args.device
    )

