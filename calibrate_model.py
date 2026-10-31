"""
calibrate_model.py - High-Impact SFT Calibration & Fine-Tuning for TunisiaLLM (Phase 2)
Focuses on:
  1. Rich Derja coding explanations (Python loops, functions, lists, conditions)
  2. Tunisian vacation, travel, tourism, and beaches recommendations
  3. Tunisian traditional food differences (Tunisian vs Moroccan couscous, recipes)
  4. Robust multi-system prompt conditioning (short & long system prompts)
  5. Flawless shopping & arithmetic tool calls (<tool_call>calc: ...</tool_call>)
"""

import os
import sys
import math
import time
import random
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

TOK_PATH = os.path.join(REPO_ROOT, "tokenizer.json")
CKPT_PATH = os.path.join(REPO_ROOT, "sft_checkpoint_best.pt")

SYSTEM_PROMPTS = [
    "إنت مساعد ذكي بالدارجة التونسية.",
    "إنت مساعد ذكي ولطيف بالدارجة التونسية، تعاون بوضوح وطلاقة وتجاوب ديما على قد السؤال.",
    "إنت مساعد ذكي ولطيف، تحكي بالدارجة التونسية بطلاقة وتجاوب بوضوح وبكل احترام."
]

def generate_targeted_dataset(tokenizer: Tokenizer) -> list:
    rng = random.Random(1337)
    samples = []

    # --------------------------------------------------------------------------
    # 1. Coding in Tunisian Derja (Python & Tech)
    # --------------------------------------------------------------------------
    coding_templates = [
        # for loop
        [
            ["كيفاش نعمل حلقة تكرار for في بايثون؟", "عطيني كود حلقة for في بايثون", "كيفاش نعمل boucle for في python؟", "بربي وريلي كيفاش نعمل for loop في بايثون"],
            "باش تعمل حلقة تكرار for في بايثون، تستعمل الدالة range() كيما هكا:\n```python\nfor i in range(5):\n    print(i)\n```\nالكود هذا يدور 5 مرات ويطبع الأرقام من 0 لـ 4 يا باهي."
        ],
        # function
        [
            ["كيفاش نعمل دالة في بايثون؟", "كيفاش نكتب function في Python؟", "كيفاش نعرف fonction في بايثون؟", "شنوة كود تعريف دالة في بايثون؟"],
            "باش تصنع دالة (function) في بايثون، تستعمل الكلمة def هكا:\n```python\ndef ejma3(a, b):\n    return a + b\n```\nومن بعد تعيطلها: `print(ejma3(3, 7))` وتخرجلك 10."
        ],
        # list append
        [
            ["كيفاش نزيد عنصر في list في بايثون؟", "كيفاش نزيد حاجة في قائمة في python؟", "كيفاش نستعمل append في بايثون؟"],
            "تستعمل الدالة append() هكا:\n```python\nqayma = ['تفاح', 'بنان']\nqayma.append('برتقال')\n```\nوهكا تولي القائمة فيها 3 عناصر."
        ],
        # if else
        [
            ["كيفاش نعمل شرط if else في بايثون؟", "كيفاش نكتب condition if في بايثون؟", "شنوة كود if else في python؟"],
            "الشرط في بايثون ساهل برشا، هاهو مثال:\n```python\n3mor = 20\nif 3mor >= 18:\n    print('راشد ومسؤول')\nelse:\n    print('ما زلت صغير')\n```"
        ],
        # list vs tuple
        [
            ["شنوة الفرق بين list و tuple في بايثون؟", "شنوة الفرق بين قائمة و tuple في python؟"],
            "الفرق الأساسي يا باهي:\n- الـ List: تتكتب بـ `[]` وتنجم تبدلها وتزيد فيها (mutable).\n- الـ Tuple: تتكتب بـ `()` وما تنجمش تبدل محتواها بعد ما تصنعها (immutable)."
        ],
        # while loop
        [
            ["كيفاش نعمل while loop في بايثون؟", "شنوة الفرق بين while و for في بايثون؟"],
            "الـ for loop تستعملها كي تبدا عارف عدد المرات، بينما الـ while loop تستعملها كي تحب تدور مادام شرط معين صحيح:\n```python\nx = 0\nwhile x < 3:\n    print(x)\n    x += 1\n```"
        ],
        # read file
        [
            ["كيفاش نقرأ فيشي في بايثون؟", "كيفاش نحل فيشي txt في python؟"],
            "تستعمل with open هكا:\n```python\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n    print(content)\n```\nوهكا يتسكر الفيشي أوتوماتيكياً."
        ]
    ]

    for q_variants, ans in coding_templates:
        for q in q_variants:
            samples.append({"q": q, "a": ans, "type": "chat"})

    # --------------------------------------------------------------------------
    # 2. Tunisian Vacation, Beaches, Tourism & Travel
    # --------------------------------------------------------------------------
    travel_templates = [
        [
            ["شنوة أحسن بلاصة نعمل فيها تصييفة في تونس؟", "وين تنصحني نصيف في تونس؟", "وين نمشي نصيف في الصيف في تونس؟", "أعطيني أحسن بلايص للتصييف في تونس"],
            "في تونس تبارك الله عنا برشا بلايص روعة للتصييف:\n1. المهدية وقليبية: إذا تحب بحر صافي وبلار ورملة بيضاء من أروع ما فمة.\n2. جربة: جزيرة الأحلام والهدوء والنزل الممتازة.\n3. الحمامات وسوسة: جو شبابي وسهرات وأوتيلات وتنشيط ممتاز.\n4. طبرقة والزوارع: بحر وجبل وغابات طبيعية هادية."
        ],
        [
            ["شنوة أحسن بحر في تونس؟", "أناهو أنظف وأصفى شط في تونس؟", "أحسن شطوط في تونس وين؟"],
            "من أصفى وأنقى الشطوط في تونس:\n- شط المنصورة في قليبية (مصنف من أصفى بحورات العالم).\n- شطوط المهدية الرملية البيضاء.\n- بحر سيدي المحرص ودوز في جربة.\n- شط الرمال في بنزرت ورأس سيراط وغار الملح."
        ],
        [
            ["وين تنصحني نحوس في تونس العاصمة؟", "شنوة البلايص الباهية في تونس العاصمة؟"],
            "في تونس العاصمة ننصحك تزور:\n- المدينة العربي وجامع الزيتونة والأسواق العتيقة.\n- سيدي بوسعيد والقهوة العالية وكعبة بمبالوني سخونة.\n- قرطاج والمسرح الروماني والموانئ البونية.\n- متحف باردو العريق للفسيفساء الرومانية."
        ],
        [
            ["شنوة المميز في الجنوب التونسي؟", "وين نمشي نحوس في الجنوب التونسي؟"],
            "الجنوب التونسي عندو سحر فريد:\n- توزر ونفطة: واحات النخيل وشط الجريد وموقع تصوير حرب النجوم.\n- مطماطة: ديار الحفر البربرية تحت الأرض.\n- تطاوين: القصور الصحراوية وقصر أولاد سلطان.\n- دوز: بوابة الصحراء التونسية والركوب على الجمال."
        ],
        [
            ["قلي على جزيرة جربة وشنوة فيها؟", "شنوة نزور في جربة؟"],
            "جربة جزيرة الأحلام، مشهورة بشطوطها الهادية، حومة السوق والصناعات التقليدية، الغريبة، قرية قلالة لصناعة الفخار، وجربة هود (Djerbahood) للفنون الحضرية."
        ]
    ]

    for q_variants, ans in travel_templates:
        for q in q_variants:
            samples.append({"q": q, "a": ans, "type": "chat"})

    # --------------------------------------------------------------------------
    # 3. Traditional Food Differences & Recipes
    # --------------------------------------------------------------------------
    food_templates = [
        [
            ["شنوة الفرق بين الكسكسي التونسي والمغربي؟", "شنوة الفرق بين الكسكسي متع تونس والمغرب؟", "قلي على الفرق بين كسكسي تونس والمغرب"],
            "الفرق الأساسي يا باهي:\n- الكسكسي التونسي: يجي أحمر وحار بالصلصة والطماطم والهريسة العربي، ومشهور بالكسكسي بالعصبان أو بالحوت (المناني/السردينة) وبالخضرة.\n- الكسكسي المغربي: يجي لونو أصفر (بالكركم والزعفران) ويكون معتدل في الحرارة، ومشهور بالخضار السبعة واللحم أو بالتفاية (بصل مكرمل بالزبيب والقرفة)."
        ],
        [
            ["قلي على اللبلابي التونسي وشنوة فيه؟", "شنوة مكونات اللبلابي التونسي؟", "كيفاش نعمل لبلابي تونسي؟"],
            "اللبلابي التونسي يجي بالخبز الشايح، الحمص السخون، الكمون، الهريسة العربي، زيت الزيتونة، عظمة مروبة، وتن وكبار حسب الرغبة. أكلة شتوية بامتياز تدفي البدن وتعدل الكيف."
        ],
        [
            ["كيفاش يطيب الكفتاجي التونسي؟", "شنوة مكونات الكفتاجي التونسي؟"],
            "الكفتاجي يتقلى فيه الفلفل الحار، الطماطم، البطاطا، القرع الأحمر وعظيمات، ومن بعد يتسطر بالسكاكين ويتفوح بالكروية والهريسة العربي وزيت الزيتونة ورشة بصل ومعدنوس."
        ],
        [
            ["شنوة هي العجة التونسية؟", "كيفاش نعمل عجة تونسية؟"],
            "العجة التونسية تطيب بصلصة طماطم وثوم وفلفل وهريسة عربي، وتكسر فوقها العظم وتخليه يروب. وأشهرها عجة بالمرقاز أو عجة بالسردينة أو القرنيط."
        ]
    ]

    for q_variants, ans in food_templates:
        for q in q_variants:
            samples.append({"q": q, "a": ans, "type": "chat"})

    # --------------------------------------------------------------------------
    # 4. Shopping & Math with Tool Calls (Reinforce the clean syntax)
    # --------------------------------------------------------------------------
    items = [
        ("طماطم", "كيلو"), ("بطاطا", "كيلو"), ("بصل", "كيلو"),
        ("تفاح", "كيلو"), ("غلة", "كيلو"), ("عظم", "كعبات"),
        ("حليب", "باكوات"), ("ياغورت", "كعبات"), ("خبز", "باقات"),
        ("لحم بقري", "كيلو"), ("اسكالوب", "كيلو")
    ]

    for _ in range(80):
        item, unit = rng.choice(items)
        qty = rng.randint(2, 6)
        price = rng.choice([400, 500, 800, 1200, 1500, 1800, 2200, 2500, 3000, 4500])
        res = qty * price
        dinar = res // 1000
        mil = res % 1000

        q = rng.choice([
            f"شريت {qty} {unit} {item} الكيلو بـ {price} مليم، قداش يطيحو؟",
            f"شريت {qty} {unit} {item} سوم {unit} {price} مليم، قداش المجموع؟",
            f"بربي احسبلي: {qty} {unit} {item} بـ {price} مليم {unit}، قداش يجيو؟"
        ])
        expr = f"{qty} * {price}"
        ans = rng.choice([
            f"المجموع يطلع {res} مليم (يعني {dinar} دينارات و {mil} مليم) يا غالي.",
            f"يطيحو {res} مليم بالتمام (يعني {dinar} دينارات و {mil} مليم) يا باهي."
        ])
        samples.append({"q": q, "tool": "calc", "expr": expr, "result": str(res), "ans": ans, "type": "tool"})

    # Pure arithmetic
    for _ in range(60):
        a = rng.randint(25, 200)
        b = rng.randint(15, 120)
        c = rng.randint(5, 50)
        q = f"بربي احسبلي: قداش {a} مع {b} إلا {c}؟"
        expr = f"{a} + {b} - {c}"
        res = a + b - c
        ans = f"النتيجة هي {res} يا باهي."
        samples.append({"q": q, "tool": "calc", "expr": expr, "result": str(res), "ans": ans, "type": "tool"})

    # --------------------------------------------------------------------------
    # 5. Greetings & Culture
    # --------------------------------------------------------------------------
    greetings = [
        ("عسلامة يا باهي، شنوة أحوالك اليوم؟", "أهلا بيك يا باهي! الحمد لله كل شي مريڤل ونحمدو ربي. وإنت شنوة أحوالك وكيفاش نجم نعاونك اليوم؟"),
        ("صباح الخير", "صباح الورد والياسمين! إن شاء الله نهارك زين ومبروك."),
        ("مساء الخير", "مساء النور والسرور! مرحباً بيك في أي وقت."),
        ("شنوة أحوالك؟", "الحمد لله كل شي مريڤل، تفضل شبيك تحتاج يا باهي؟"),
        ("قلي على تاريخ مدينة القيروان وشكون بناها؟", "مدينة القيروان أسسها القائد الإسلامي عقبة بن نافع سنة 670 ميلادي (50 هـ) كأول عاصمة إسلامية في شمال إفريقيا. وتضم جامع عقبة بن نافع التاريخي وبرك الأغالبة ومقام سيدي الصحبي.")
    ]

    for q, a in greetings:
        samples.append({"q": q, "a": a, "type": "chat"})

    rng.shuffle(samples)
    print(f"[OK] Total targeted calibration samples: {len(samples)}", flush=True)
    return samples


def tokenize_sample(sample: dict, tokenizer: Tokenizer, sys_prompt: str, max_seq_len: int = 256):
    q = sample["q"]
    pad_id = tokenizer.token_to_id("<pad>") or 0

    if sample["type"] == "tool":
        expr = sample["expr"]
        res = sample["result"]
        ans = sample["ans"]

        prefix = f"<bos><|system|>\n{sys_prompt}\n<|user|>\n{q}\n<|assistant|>\n"
        pre_tool = f"<tool_call>calc: {expr}</tool_call>"
        tool_res = f"<tool_result>{res}</tool_result>"
        post_tool = f"{ans}<eos>"

        p_ids = tokenizer.encode(prefix).ids
        pre_ids = tokenizer.encode(pre_tool).ids
        res_ids = tokenizer.encode(tool_res).ids
        post_ids = tokenizer.encode(post_tool).ids

        input_ids = p_ids + pre_ids + res_ids + post_ids
        labels = ([-100] * len(p_ids)) + pre_ids + ([-100] * len(res_ids)) + post_ids
    else:
        ans = sample["a"]
        prefix = f"<bos><|system|>\n{sys_prompt}\n<|user|>\n{q}\n<|assistant|>\n"
        reply = f"{ans}<eos>"

        p_ids = tokenizer.encode(prefix).ids
        rep_ids = tokenizer.encode(reply).ids

        input_ids = p_ids + rep_ids
        labels = ([-100] * len(p_ids)) + rep_ids

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    pad_len = max_seq_len - len(input_ids)
    input_ids = input_ids + [pad_id] * pad_len
    labels = labels + [-100] * pad_len

    return input_ids, labels


def main():
    print("=" * 80, flush=True)
    print("TUNISIALLM: PHASE 2 TARGETED INSTRUCTION CALIBRATION", flush=True)
    print("=" * 80, flush=True)

    tokenizer = Tokenizer.from_file(TOK_PATH)
    cfg = ModelConfig.from_preset("80M", vocab_size=16384, max_seq_len=256)
    model = TunisiaLLM(cfg)

    print(f"[*] Loading checkpoint: {CKPT_PATH}", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    print("[OK] Loaded model weights!", flush=True)

    raw_samples = generate_targeted_dataset(tokenizer)

    rng = random.Random(42)
    processed = []
    for s in raw_samples:
        # Train on multiple system prompt variants so model generalizes
        sp = rng.choice(SYSTEM_PROMPTS)
        processed.append(tokenize_sample(s, tokenizer, sp, max_seq_len=256))

    rng.shuffle(processed)
    val_split = 30
    val_data = processed[:val_split]
    train_data = processed[val_split:]

    val_inputs = torch.tensor([x[0] for x in val_data], dtype=torch.long)
    val_targets = torch.tensor([x[1] for x in val_data], dtype=torch.long)

    print(f"[+] Training examples: {len(train_data)} | Validation examples: {len(val_data)}", flush=True)

    # Initial validation loss
    model.eval()
    with torch.no_grad():
        _, init_loss = model(val_inputs[:, :-1], targets=val_targets[:, 1:])
    print(f"[*] Pre-calibration Validation Loss: {init_loss.item():.4f}", flush=True)

    batch_size = 4
    total_steps = 60
    lr = 2.0e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n[*] Commencing Phase 2 calibration ({total_steps} steps on CPU)...", flush=True)
    model.train()
    t0 = time.time()
    best_val_loss = init_loss.item()
    best_state = None

    for step in range(total_steps):
        batch = rng.sample(train_data, batch_size)
        inp_batch = torch.tensor([x[0] for x in batch], dtype=torch.long)
        tgt_batch = torch.tensor([x[1] for x in batch], dtype=torch.long)

        progress = step / total_steps
        current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        _, loss = model(inp_batch[:, :-1], targets=tgt_batch[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 10 == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                _, val_loss = model(val_inputs[:, :-1], targets=val_targets[:, 1:])
            model.train()
            elapsed = time.time() - t0
            t_per_step = elapsed / (step + 1)
            print(f"  Step {step+1:02d}/{total_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | LR: {current_lr:.2e} | Speed: {t_per_step:.2f}s/step", flush=True)
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.time() - t0
    print(f"\n[OK] Calibration finished in {total_time:.1f}s!", flush=True)
    print(f"[*] Initial Val Loss: {init_loss.item():.4f} -> Final Best Val Loss: {best_val_loss:.4f}", flush=True)

    if best_state is None:
        best_state = model.state_dict()

    save_targets = [
        CKPT_PATH,
        os.path.join(REPO_ROOT, "tunisiallm_model", "sft_checkpoint_best.pt"),
        os.path.join(REPO_ROOT, "checkpoints", "sft_checkpoint_best.pt")
    ]

    new_ckpt = {
        "step": ckpt.get("step", 600) + total_steps,
        "epoch": 2,
        "model_state_dict": best_state,
        "best_val_loss": best_val_loss,
        "config": ckpt.get("config", cfg)
    }

    for path in save_targets:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(new_ckpt, path)
        print(f"[OK] Saved updated model checkpoint to: {path}", flush=True)

    zip_path = os.path.join(REPO_ROOT, "tunisiallm_model.zip")
    import zipfile
    print(f"[*] Updating {zip_path} with new calibrated model...", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(CKPT_PATH, arcname="sft_checkpoint_best.pt")
        zf.write(TOK_PATH, arcname="tokenizer.json")
        zf.write(os.path.join(REPO_ROOT, "config.yaml"), arcname="config.yaml")
    print(f"[OK] {zip_path} successfully re-packaged!", flush=True)


if __name__ == "__main__":
    main()
