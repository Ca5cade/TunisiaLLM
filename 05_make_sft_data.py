"""
05_make_sft_data.py - SFT Dataset Builder with Tool-Calling & Loss Masking
Generates high-quality chat data for Tunisian Derja instruction tuning:
  1. Tunisian dialogue & QA
  2. Casual chat with rich spelling variants (Arabic script & Arabizi/French code-switching)
  3. Synthetic arithmetic with <tool_call>calc: EXPR</tool_call>
  4. Synthetic factual QA with <tool_call>search: QUERY</tool_call>
  5. Honest refusal / "I don't know" samples
Applies Assistant-only loss masking (setting prompt and tool_result tokens to -100).
"""

import os
import sys
import json
import math
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any

import torch
import numpy as np
from tokenizers import Tokenizer

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from utils import load_config, set_seed, format_num_params, format_bytes


# ==============================================================================
# Prompt Templates & Knowledge Bases
# ==============================================================================

SYSTEM_PROMPTS = [
    "إنت مساعد ذكي ولطيف، تحكي بالدارجة التونسية بطلاقة وتجاوب بوضوح وبكل احترام.",
    "إنت رفيق ذكي بالدارجة التونسية، تعاون العباد وتجاوبهم بالباهي وبطريقة ساهلة ومفهومة.",
    "مساعد ذكي تونسي، تخدم بالدارجة التونسية المفهومة، وتعاون في الحساب والمعلومات العامة بكل أمانة.",
    "Tu es un assistant IA tunisien qui parle couramment la Derja tunisienne avec du code-switching naturel en français et anglais."
]

# Greetings & Casual Chat Seeds
GREETINGS_USER = [
    "عسلامة", "عسلامة يا خويا", "أهلا وسهلا", "صباح الخير", "صباح الورد والياسمين",
    "مساء الخير", "مساء النور", "شنوة أحوالك اليوم؟", "شنوة حوالك يا باهي؟", "شبيك راقد؟",
    "ahla bik", "chnawa a7welek", "cava elyoum?", "labes 3lik?", "ya sahbi cava?",
    "bonjour khouya", "bonsoir cava?", "koll chay mrigel?", "ach khbarek?"
]

GREETINGS_RESPONSES = [
    "عسلامة يا باهي! الحمد لله كل شي مريڤل، شنوة أحوالك إنت وكيفاش نجم نعاونك اليوم؟",
    "أهلا وسهلاً بيك! نهارك زين وسعيد إن شاء الله. تفضل شتحب نساعفك؟",
    "صباح الورد والفل! إن شاء الله نهارك مبروك ومليان خير وبركة.",
    "مساء النور والسرور! مرحباً بيك، أنا في الخدمة في أي وقت.",
    "Ahla bik ya ghali! Labes el 7amdoullah, nchallah inti zeda cava w mrigel.",
    "Bonjour! Ça va très bien merci, chnouwa najjem n3awnek bih lyoum?",
    "الحمد لله ونشكروه، ديما في الخدمة بكل فرحة، شنوة سؤالك اليوم؟"
]

CITIES_TUNISIA = ["تونس", "صفاقس", "سوسة", "القيروان", "بنزرت", "قابس", "مدنين", "جربة", "نابل", "المنستير", "توزر", "الكاف", "باجة", "جندوبة"]
FOOD_TUNISIA = [
    ("كسكسي بالعصبان", "الكسكسي بالعصبان التونسي من أبن الماكلات التقليدية، يطيب بالخضرة والحمص وقرينات فلفل حار."),
    ("لبلابي", "اللبلابي التونسي يجي بالخبز الشايح، الحمص السخون، الكمون، الهريسة العربي، الزيت وعظمة مروبة."),
    ("كفتاجي", "الكفتاجي يتقلى فيه الفلفل، الطماطم، البطاطا، القرع والبيض، ومن بعد يتسطر بالسكاكين ويتفوح بالكروية والملح."),
    ("صحن تونسي", "الصحن التونسي فيه سلطة مشوية، سلطة خضراء، تن، بيض مسلوق، زيتون، كبار وهريسة عربي بنينة."),
    ("مقرونة جارية", "المقرونة الجارية بالحوت ولا باللحم في ليالي الشتاء الباردة تدفي البدن وتعدل الكيف.")
]

SEARCH_FACTS = [
    ("تاريخ تأسيس قرطاج", "تأسست مدينة قرطاج التاريخية عام 814 قبل الميلاد على يد الأميرة الفينيقية أليسا (ديدون).", "قرطاج تأسست سنة 814 قبل الميلاد على يد الفينيقيين وجات على يد الأميرة أليسا."),
    ("تاريخ تأسيس جامع الزيتونة", "تأسس جامع الزيتونة المعمور بتونس سنة 737 ميلادي في عهد الدولة الأموية.", "جامع الزيتونة تأسس عام 737 ميلادي وتعتبر جامعته من أقدم المنارات العلمية في العالم الإسلامي."),
    ("تأسيس مدينة القيروان", "أسس القائد عقبة بن نافع مدينة القيروان سنة 670 ميلادي (50 هجري).", "القيروان بناها عقبة بن نافع عام 670 ميلادي وكانت عاصمة الفتوحات الإسلامية في شمال إفريقيا."),
    ("أعلى قمة جبلية في تونس", "جبل الشعانبي بالقصرين هو أعلى قمة في تونس بارتفاع 1544 متر فوق سطح البحر.", "أعلى قمة في تونس هي جبل الشعانبي في ولاية القصرين، علوه 1544 متر."),
    ("الفائز بكأس أمم إفريقيا 2004", "فاز المنتخب الوطني التونسي بكأس إفريقيا 2004 بتغلبه على المغرب 2-1 في رادس.", "المنتخب التونسي ربح الكان 2004 بعد ما ربح المغرب في النهائي 2-1 بملعب رادس."),
    ("مساحة الجمهورية التونسية", "تبلغ مساحة تونس الإجمالية 163,610 كيلومتر مربع.", "المساحة الإجمالية لتونس هي 163,610 كم²."),
    ("تاريخ استقلال تونس", "نالت تونس استقلالها عن فرنسا يوم 20 مارس 1956.", "تونس خذات استقلالها الرسمي نهار 20 مارس 1956."),
    ("أقدم متحف في تونس", "المتحف الوطني بباردو هو من أشهر وأقدم المتاحف ويضم أكبر مجموعة فسيفساء رومانية في العالم.", "متحف باردو الوطني من أقدم وأكبر المتاحف وفيه أندر الفسيفساء الرومانية في العالم.")
]

REFUSAL_PROMPTS = [
    ("شكون باش يربح كأس إفريقيا 2034؟", "سامحني، هذي حاجة في المستقبل وما يعلم الغيب إلا ربي، ما نجمش نتكهن بنتيجة حاجة ما صارتش."),
    ("اعطيني كود ويفي ولا باسوورد فيسبوك متع فلان؟", "سامحني، ما نجمش نعاونك في أخذ معلومات شخصية أو كلمات سر، هذي خصوصية وأمانة."),
    ("شكون باش ينجح في الانتخابات الجاية؟", "ما عنديش إمكانية باش نعرف المستقبل أو نتائج انتخابات ما زالت ما تعقدتش."),
    ("كيفاش نصنع قنبلة ولا حاجة تضر الناس؟", "سامحني، قطيعاً ما نجمش نعاون في أي حاجة فيها ضرر ولا خطر على سلامة العباد."),
    ("قداش باش تولي عملة البيتكوين عام 2030؟", "أسعار العملات والبورصة في المستقبل لا يعلمها أحد، وما نجمش نعطي توقعات مالية مستقبلية غيبية.")
]


# ==============================================================================
# Synthetic Generators
# ==============================================================================

def generate_casual_chat(rng: random.Random) -> Dict:
    """Generate casual greetings and smalltalk dialogues."""
    sys_p = rng.choice(SYSTEM_PROMPTS)
    user_q = rng.choice(GREETINGS_USER)
    ans = rng.choice(GREETINGS_RESPONSES)

    # Variation: ask about weather or local town
    if rng.random() < 0.4:
        city = rng.choice(CITIES_TUNISIA)
        user_q = f"عسلامة، شنوة الطقس اليوم في {city}؟"
        ans = f"أهلا بيك يا باهي! الطقس في {city} عموماً باهي ومستقر، ديما تفقد نشرة الجو الرسمية قبل ما تخرج."

    return {
        "system": sys_p,
        "turns": [
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans}
        ]
    }


def generate_tunisian_qa(rng: random.Random) -> Dict:
    """Generate Tunisian food, traditions, and cultural Q&A."""
    sys_p = rng.choice(SYSTEM_PROMPTS)
    food_name, food_desc = rng.choice(FOOD_TUNISIA)

    templates = [
        (f"كيفاش يطيب {food_name} التونسي وشنوة فيه؟", f"{food_desc} وصحة وبالشفاء ليك ولعيلتك!"),
        (f"بربي شنوة المكونات الأساسية متع {food_name}؟", f"المكونات الأساسية: {food_desc}"),
        (f"قلي {food_name} بنين في الشتاء ولا الصيف؟", f"أكيد {food_name} بنين برشا، وخاصة {food_desc}")
    ]
    u, a = rng.choice(templates)
    return {
        "system": sys_p,
        "turns": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a}
        ]
    }


def generate_math_tool_dialogue(rng: random.Random) -> Dict:
    """Generate math problems with <tool_call>calc: EXPR</tool_call> and <tool_result>."""
    sys_p = rng.choice(SYSTEM_PROMPTS)
    op = rng.choice(["add_sub", "mul", "shopping", "discount"])

    if op == "add_sub":
        a = rng.randint(25, 999)
        b = rng.randint(15, 500)
        c = rng.randint(5, 150)
        user_q = f"بربي احسبلي: قداش {a} مع {b} إلا {c}؟"
        expr = f"{a} + {b} - {c}"
        res = a + b - c
        ans_text = f"النتيجة هي {res} يا باهي."

    elif op == "mul":
        a = rng.randint(12, 120)
        b = rng.randint(6, 45)
        user_q = f"قداش حاصل ضرب {a} في {b}؟"
        expr = f"{a} * {b}"
        res = a * b
        ans_text = f"حاصل الضرب هو {res}."

    elif op == "shopping":
        if rng.random() < 0.5:
            # Single-item shopping / weight calculation
            qty = rng.randint(2, 7)
            price = rng.choice([500, 800, 1200, 1500, 1800, 2500, 3200])
            unit = rng.choice(["كيلو", "كعبات", "باكوات"])
            item = rng.choice(["طماطم", "بطاطا", "بصل", "تفاح", "غلة"])
            user_q = rng.choice([
                f"شريت {qty} {unit} {item}، سوم {unit} بـ {price} مليم، قداش يطيحو؟",
                f"شريت {qty} {unit} بـ {price} مليم، قداش المجموع؟",
                f"قداش سوم {qty} {unit} {item} الكيلو بـ {price} مليم؟"
            ])
            expr = f"{qty} * {price}"
            res = qty * price
        else:
            # Two-item composite shopping
            qty1 = rng.randint(2, 5)
            price1 = rng.choice([250, 400, 800, 1500, 3200])
            qty2 = rng.randint(1, 4)
            price2 = rng.choice([600, 1200, 2500, 4500])
            user_q = f"شريت {qty1} كعبات بـ {price1} مليم و {qty2} كعبات بـ {price2} مليم، قداش المجموع الكل؟"
            expr = f"{qty1} * {price1} + {qty2} * {price2}"
            res = qty1 * price1 + qty2 * price2

        dinar = res // 1000
        millimes = res % 1000
        ans_text = f"المجموع يطلع {res} مليم (يعني {dinar} دينارات و {millimes} مليم) يا غالي."

    else:  # discount
        orig = rng.choice([40, 80, 120, 150, 200, 300])
        pct = rng.choice([10, 20, 25, 30, 50])
        user_q = f"حاجة سومها {orig} دينار وعليها صولد {pct}%، قداش يولي سومها؟"
        expr = f"{orig} * (1 - {pct / 100:.2f})"
        res = round(orig * (1 - pct / 100.0), 2)
        ans_text = f"بعد التخفيض، سومها يولي {res} دينار."

    assistant_content = f"<tool_call>calc: {expr}</tool_call><tool_result>{res}</tool_result>{ans_text}"

    return {
        "system": sys_p,
        "turns": [
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def generate_search_tool_dialogue(rng: random.Random) -> Dict:
    """Generate factual queries emitting <tool_call>search: QUERY</tool_call>."""
    sys_p = rng.choice(SYSTEM_PROMPTS)
    query_topic, snippet, derja_summary = rng.choice(SEARCH_FACTS)

    user_q = f"بربي قلي شنوة {query_topic}؟"
    assistant_content = f"<tool_call>search: {query_topic}</tool_call><tool_result>{snippet}</tool_result>{derja_summary}"

    return {
        "system": sys_p,
        "turns": [
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def generate_refusal_dialogue(rng: random.Random) -> Dict:
    """Generate honest refusal and out-of-scope question responses."""
    sys_p = rng.choice(SYSTEM_PROMPTS)
    user_q, ref_ans = rng.choice(REFUSAL_PROMPTS)

    return {
        "system": sys_p,
        "turns": [
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ref_ans}
        ]
    }


# ==============================================================================
# Tokenization & Loss Masking Formatter
# ==============================================================================

def format_and_mask_conversation(
    conv: Dict,
    tokenizer: Tokenizer,
    max_seq_len: int = 1024
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Format chat conversation into single token stream and compute loss mask.
    Only assistant tokens (including <tool_call>) have valid label IDs.
    Prompt tokens and <tool_result> have label = -100.
    """
    sys_prompt = conv.get("system", "")
    turns = conv["turns"]

    # Special token strings
    bos_str = "<bos>"
    eos_str = "<eos>"
    sys_tag = "<|system|>\n"
    user_tag = "<|user|>\n"
    asst_tag = "<|assistant|>\n"
    tool_res_start = "<tool_result>"
    tool_res_end = "</tool_result>"

    input_ids = []
    labels = []

    # Add BOS + System Prompt
    sys_text = f"{bos_str}{sys_tag}{sys_prompt}\n"
    sys_tokens = tokenizer.encode(sys_text).ids
    input_ids.extend(sys_tokens)
    labels.extend([-100] * len(sys_tokens))  # Mask system prompt

    for turn in turns:
        role = turn["role"]
        content = turn["content"]

        if role == "user":
            user_text = f"{user_tag}{content}\n"
            user_tokens = tokenizer.encode(user_text).ids
            input_ids.extend(user_tokens)
            labels.extend([-100] * len(user_tokens))  # Mask user input

        elif role == "assistant":
            # Assistant tag itself is part of prompt
            tag_tokens = tokenizer.encode(asst_tag).ids
            input_ids.extend(tag_tokens)
            labels.extend([-100] * len(tag_tokens))

            # Assistant content may contain <tool_result> injected from tool execution
            if tool_res_start in content and tool_res_end in content:
                # Split content into: [pre_tool, tool_result, post_tool]
                parts = content.split(tool_res_start)
                pre_tool = parts[0]  # Contains <tool_call>calc: ...</tool_call>
                rest = parts[1].split(tool_res_end)
                result_content = rest[0]
                post_tool = rest[1]

                # 1. Pre-tool (Assistant generated): Trainable!
                pre_tokens = tokenizer.encode(pre_tool).ids
                input_ids.extend(pre_tokens)
                labels.extend(pre_tokens)

                # 2. Tool result (Injected environment): Masked (-100)!
                res_text = f"{tool_res_start}{result_content}{tool_res_end}"
                res_tokens = tokenizer.encode(res_text).ids
                input_ids.extend(res_tokens)
                labels.extend([-100] * len(res_tokens))

                # 3. Post-tool answer: Trainable!
                post_text = f"{post_tool}{eos_str}"
                post_tokens = tokenizer.encode(post_text).ids
                input_ids.extend(post_tokens)
                labels.extend(post_tokens)

            else:
                # Regular assistant response: Trainable!
                asst_text = f"{content}{eos_str}"
                asst_tokens = tokenizer.encode(asst_text).ids
                input_ids.extend(asst_tokens)
                labels.extend(asst_tokens)

    # Check max length
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    return input_ids, labels


# ==============================================================================
# External Instruction Datasets Loader
# ==============================================================================

def load_external_instruction_datasets(cfg: Dict, is_smoke: bool = False) -> List[Dict]:
    """Load authentic instruction/chat datasets from HuggingFace to increase versatility."""
    ext_configs = cfg.get("sft_data", {}).get("external_instruction_datasets", [])
    if not ext_configs or not HF_DATASETS_AVAILABLE:
        return []

    conversations = []
    default_sys = "إنت مساعد ذكي ولطيف، تحكي بالدارجة التونسية بطلاقة وتجاوب بوضوح وبكل احترام."

    for item in ext_configs:
        name = item.get("name")
        ins_col = item.get("instruction_col", "instruction")
        res_col = item.get("response_col", "response")
        split = item.get("split", "train")
        limit = 50 if is_smoke else item.get("limit", 10000)

        print(f"\n[*] Fetching external instruction dataset: {name} (limit={limit})...")
        try:
            ds = load_dataset(name, split=f"{split}[:{limit}]")
            cnt = 0
            for row in ds:
                ins = ""
                res = ""
                if "conversations" in row and isinstance(row["conversations"], list):
                    # Multi-turn or ShareGPT style: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
                    convs = row["conversations"]
                    if len(convs) >= 2:
                        ins = convs[0].get("value", "")
                        res = convs[1].get("value", "")
                else:
                    ins = row.get(ins_col) or row.get("question") or row.get("prompt") or ""
                    res = row.get(res_col) or row.get("response") or row.get("output") or row.get("answer") or ""

                if ins and res and isinstance(ins, str) and isinstance(res, str):
                    ins = ins.strip()
                    res = res.strip()
                    if len(ins) >= 5 and len(res) >= 5:
                        conversations.append({
                            "system": default_sys,
                            "turns": [
                                {"role": "user", "content": ins},
                                {"role": "assistant", "content": res}
                            ],
                            "category": "instruction_tuning",
                            "source": name
                        })
                        cnt += 1
            print(f"    [+] Successfully loaded {cnt:,} instruction dialogues from {name}")
        except Exception as e:
            print(f"    [!] Could not load {name} ({e}). Skipping.")

    return conversations


# ==============================================================================
# SFT Dataset Generation Orchestrator
# ==============================================================================

def build_sft_dataset(cfg_path: Optional[str] = None, smoke_test: Optional[bool] = None, force: bool = False) -> None:
    """Generate SFT dialogues, apply loss masking, and save JSONL and tokenized datasets."""
    cfg = load_config(cfg_path, smoke_test_override=smoke_test)
    set_seed(cfg.get("seed", 42))

    sft_cfg = cfg["sft_data"]
    is_smoke = cfg.get("smoke_test", False)
    total_samples = sft_cfg.get("smoke_test_samples", 1000) if is_smoke else sft_cfg.get("num_samples", 150000)
    val_samples = 100 if is_smoke else sft_cfg.get("val_samples", 2000)

    sft_dir = cfg["paths"]["sft_data_dir"]
    os.makedirs(sft_dir, exist_ok=True)

    train_pt_path = os.path.join(sft_dir, "sft_train.pt")
    val_pt_path = os.path.join(sft_dir, "sft_val.pt")
    if not force and os.path.isfile(train_pt_path) and os.path.isfile(val_pt_path):
        print(f"[OK] SFT dataset already exists at {train_pt_path}.")
        print("     Skipping SFT data generation. (Pass --force to re-generate).")
        return

    print("=" * 80)
    print("TunisiaLLM SFT Chat Dataset Generator")
    print(f"Mode: {'SMOKE TEST' if is_smoke else 'FULL RUN'}")
    print(f"Target Total Samples: {total_samples:,} (Train: {total_samples - val_samples:,}, Val: {val_samples:,})")
    print(f"Output Directory:     {sft_dir}")
    print("=" * 80)

    tokenizer_path = cfg["paths"]["tokenizer_path"]
    if not os.path.isfile(tokenizer_path):
        print("[!] Tokenizer not found. Running tokenizer training first...")
        import importlib
        tok_mod = importlib.import_module("02_tokenizer")
        tok_mod.run_tokenizer_pipeline(cfg_path, smoke_test=is_smoke)

    tokenizer = Tokenizer.from_file(tokenizer_path)

    cats = sft_cfg.get("categories", {
        "derja_casual": 0.35,
        "derja_qa": 0.30,
        "math_tools": 0.15,
        "search_tools": 0.12,
        "refusals": 0.08
    })

    cat_names = list(cats.keys())
    cat_weights = [cats[k] for k in cat_names]

    generators = {
        "derja_casual": generate_casual_chat,
        "derja_qa": generate_tunisian_qa,
        "math_tools": generate_math_tool_dialogue,
        "search_tools": generate_search_tool_dialogue,
        "refusals": generate_refusal_dialogue
    }

    rng = random.Random(cfg.get("seed", 42))

    all_conversations = []

    # 1. Load external authentic instruction datasets (Tunisian QA, Tech, Arabic Alpaca)
    external_convs = load_external_instruction_datasets(cfg, is_smoke=is_smoke)
    if external_convs:
        all_conversations.extend(external_convs)
        print(f"[+] Loaded {len(external_convs):,} external authentic instruction samples.")

    # 2. Synthesize structured chat dialogues (casual Derja, math tools, search tools, refusals)
    synthetic_needed = max(0, total_samples - len(all_conversations)) if not is_smoke else max(100, total_samples)
    print(f"\n[*] Synthesizing {synthetic_needed:,} structured chat dialogues (casual, tools, culture)...")
    for i in range(synthetic_needed):
        cat = rng.choices(cat_names, weights=cat_weights, k=1)[0]
        gen_fn = generators[cat]
        conv = gen_fn(rng)
        conv["category"] = cat
        all_conversations.append(conv)

    rng.shuffle(all_conversations)
    val_convs = all_conversations[:val_samples]
    train_convs = all_conversations[val_samples:]

    # 1. Save Human-Readable JSONL
    train_jsonl = os.path.join(sft_dir, "sft_train.jsonl")
    val_jsonl = os.path.join(sft_dir, "sft_val.jsonl")

    print(f"Saving train JSONL to: {train_jsonl}")
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for c in train_convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Saving val JSONL to: {val_jsonl}")
    with open(val_jsonl, "w", encoding="utf-8") as f:
        for c in val_convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 2. Tokenize and Apply Assistant-Only Loss Masking
    max_seq_len = sft_cfg.get("max_seq_len", 256)
    if is_smoke:
        max_seq_len = 128

    print(f"\n[*] Pre-tokenizing with assistant-only loss masking (max_seq_len={max_seq_len})...")

    def process_split(convs: List[Dict], split_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        all_inputs = []
        all_labels = []
        total_tokens = 0
        trained_tokens = 0

        pad_id = tokenizer.token_to_id("<pad>") or 0

        for conv in convs:
            res = format_and_mask_conversation(conv, tokenizer, max_seq_len=max_seq_len)
            if res is None:
                continue
            inp, lab = res
            pad_len = max_seq_len - len(inp)
            inp_padded = inp + [pad_id] * pad_len
            lab_padded = lab + [-100] * pad_len

            all_inputs.append(inp_padded)
            all_labels.append(lab_padded)

            total_tokens += len(inp)
            trained_tokens += sum(1 for x in lab if x != -100)

        input_tensor = torch.tensor(all_inputs, dtype=torch.long)
        label_tensor = torch.tensor(all_labels, dtype=torch.long)

        pct = (trained_tokens / max(1, total_tokens)) * 100
        print(f"    [{split_name.upper()}] Processed {len(all_inputs):,} examples | "
              f"Total Tokens: {total_tokens:,} | "
              f"Trained (Assistant) Tokens: {trained_tokens:,} ({pct:.1f}%)")
        return input_tensor, label_tensor

    train_inputs, train_labels = process_split(train_convs, "train")
    val_inputs, val_labels = process_split(val_convs, "val")

    # 3. Save PyTorch Tokenized Tensors
    train_pt_path = os.path.join(sft_dir, "sft_train.pt")
    val_pt_path = os.path.join(sft_dir, "sft_val.pt")

    torch.save({"input_ids": train_inputs, "labels": train_labels}, train_pt_path)
    torch.save({"input_ids": val_inputs, "labels": val_labels}, val_pt_path)

    print(f"\n[OK] SFT Dataset Generation Complete!")
    print(f"     Train data: {train_pt_path} ({format_bytes(train_inputs.nbytes * 2)})")
    print(f"     Val data:   {val_pt_path} ({format_bytes(val_inputs.nbytes * 2)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TunisiaLLM SFT Data Generator")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Generate small smoke test dataset")
    parser.add_argument("--force", action="store_true", help="Force re-generation even if files exist")
    args = parser.parse_args()

    build_sft_dataset(args.config, smoke_test=args.smoke_test if args.smoke_test else None, force=args.force)
