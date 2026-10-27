"""
patch_shopping_sft.py - Lightweight SFT Calibration for Shopping Arithmetic
Calibrates tunisiallm_model with single-item & composite shopping dialogues
so that tool calls and Tunisian Derja explanations are 100% accurate.
"""

import os
import random
import time
import torch
from tokenizers import Tokenizer
import importlib

model_mod = importlib.import_module("03_model")
ModelConfig = model_mod.ModelConfig
TunisiaLLM = model_mod.TunisiaLLM

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(REPO_ROOT, "tunisiallm_model", "sft_checkpoint_best.pt")
TOK_PATH = os.path.join(REPO_ROOT, "tunisiallm_model", "tokenizer.json")

print("[*] Loading tokenizer and model...")
tok = Tokenizer.from_file(TOK_PATH)
cfg = ModelConfig.from_preset("80M", vocab_size=16384, max_seq_len=256)
model = TunisiaLLM(cfg).to("cpu")

ckpt = torch.load(CKPT_PATH, map_location="cpu")
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=False)
model.train()

print("[*] Generating synthetic shopping calibration dialogues...")
rng = random.Random(42)
samples = []

system_prompt = "إنت مساعد ذكي بالدارجة التونسية."
pad_id = tok.token_to_id("<pad>")

items = [
    ("طماطم", "كيلو"), ("بطاطا", "كيلو"), ("بصل", "كيلو"),
    ("تفاح", "كيلو"), ("غلة", "كيلو"), ("عظم", "كعبات"),
    ("حليب", "باكوات"), ("ياغورت", "كعبات"), ("خبز", "باقات")
]

for _ in range(160):
    if rng.random() < 0.65:
        # Single-item shopping
        item, unit = rng.choice(items)
        qty = rng.randint(2, 6)
        price = rng.choice([400, 500, 800, 1200, 1400, 1500, 1800, 2200, 2500, 3000])
        
        q_templates = [
            f"شريت {qty} {unit} {item} الكيلو بـ {price} مليم، قداش يطيحو؟",
            f"شريت {qty} {unit} {item}، سوم {unit} بـ {price} مليم، قداش المجموع؟",
            f"قداش سوم {qty} {unit} {item} سوم الكيلو {price} مليم؟",
            f"بربي احسبلي: {qty} {unit} {item} بـ {price} مليم الكيلو، قداش يجيو؟"
        ]
        q = rng.choice(q_templates)
        expr = f"{qty} * {price}"
        res = qty * price
        dinar = res // 1000
        mil = res % 1000
        
        ans_templates = [
            f"المجموع يطلع {res} مليم (يعني {dinar} دينارات و {mil} مليم) يا غالي.",
            f"المنتوج النهائي يطلع {res} مليم (يعني {dinar} دينارات و {mil} مليم) يا باهي.",
            f"يطيحو {res} مليم بالتمام (يعني {dinar} دينارات و {mil} مليم)."
        ]
        ans = rng.choice(ans_templates)
    else:
        # Composite two-item shopping
        item1, unit1 = rng.choice(items)
        item2, unit2 = rng.choice(items)
        qty1 = rng.randint(2, 4)
        price1 = rng.choice([500, 800, 1500, 2000])
        qty2 = rng.randint(1, 3)
        price2 = rng.choice([600, 1200, 1800, 2500])
        
        q = f"شريت {qty1} {unit1} {item1} بـ {price1} مليم و {qty2} {unit2} {item2} بـ {price2} مليم، قداش المجموع الكل؟"
        expr = f"{qty1} * {price1} + {qty2} * {price2}"
        res = qty1 * price1 + qty2 * price2
        dinar = res // 1000
        mil = res % 1000
        ans = f"المجموع الكل يطلع {res} مليم (يعني {dinar} دينارات و {mil} مليم) يا غالي."

    assistant_content = f"<tool_call>calc: {expr}</tool_call><tool_result>{res}</tool_result>{ans}"
    full_text = f"<bos><|system|>\n{system_prompt}\n<|user|>\n{q}\n<|assistant|>\n{assistant_content}<eos>"
    
    ids = tok.encode(full_text).ids
    prefix = f"<bos><|system|>\n{system_prompt}\n<|user|>\n{q}\n<|assistant|>\n"
    prefix_len = len(tok.encode(prefix).ids)
    
    labels = list(ids)
    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100
        
    samples.append((ids, labels))

print(f"[OK] {len(samples)} calibration samples ready.")

optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=0.01)

print("[*] Starting fast 40-step calibration on CPU...")
t0 = time.time()
for step in range(40):
    batch = rng.sample(samples, 4)
    max_len = max(len(x[0]) for x in batch)
    
    inputs = [x[0] + [pad_id] * (max_len - len(x[0])) for x in batch]
    targets = [x[1] + [-100] * (max_len - len(x[1])) for x in batch]
    
    inp_t = torch.tensor(inputs, dtype=torch.long)
    tgt_t = torch.tensor(targets, dtype=torch.long)
    
    optimizer.zero_grad()
    _, loss = model(inp_t[:, :-1], targets=tgt_t[:, 1:])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if (step + 1) % 10 == 0 or step == 0:
        print(f"  Step {step+1:02d}/40 | Loss: {loss.item():.4f}")

elapsed = time.time() - t0
print(f"[OK] Calibration complete in {elapsed:.1f}s!")

# Save calibrated checkpoint
torch.save({"model_state_dict": model.state_dict(), "config": cfg}, CKPT_PATH)
print(f"[OK] Saved calibrated weights to: {CKPT_PATH}")
