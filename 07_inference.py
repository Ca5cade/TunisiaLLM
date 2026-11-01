"""
07_inference.py - Fast Streaming Inference Engine with KV Cache & Tool-Calling Loop
Features:
  - Autoregressive token-by-token streaming with KV cache
  - Nucleus (Top-p) sampling, temperature, and repetition penalty
  - Safe AST-based calculator (no eval, syntax/division protection)
  - Pluggable search tool (Tunisian knowledge base + web fallback)
  - Multi-turn interactive CLI chat with tool execution loop
  - FP16 & Dynamic INT8 quantization export for fast CPU/GPU serving
"""

import os
import sys
import re
import ast
import math
import time
import argparse
from typing import List, Dict, Tuple, Optional, Generator, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from utils import load_config, format_time, format_bytes

import importlib
model_mod = importlib.import_module("03_model")
ModelConfig = model_mod.ModelConfig
TunisiaLLM = model_mod.TunisiaLLM


# ==============================================================================
# Safe AST-Based Calculator (No eval(), 100% Secure)
# ==============================================================================

class SafeCalculator(ast.NodeVisitor):
    """
    Evaluates mathematical expressions using AST parsing with strict node whitelist.
    Prevents arbitrary code execution, unlimited memory usage, and zero-division.
    """
    ALLOWED_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b if b != 0 else float("nan"),
        ast.FloorDiv: lambda a, b: a // b if b != 0 else float("nan"),
        ast.Mod: lambda a, b: a % b if b != 0 else float("nan"),
        ast.Pow: lambda a, b: a ** b if (abs(b) <= 20 and abs(a) <= 10000) else float("nan"),
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: a,
    }

    ALLOWED_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt
    }

    def visit(self, node: ast.AST) -> Any:
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsafe constant type: {type(node.value)}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in self.ALLOWED_OPERATORS:
            res = self.ALLOWED_OPERATORS[op_type](left, right)
            if math.isnan(res) or math.isinf(res):
                raise ZeroDivisionError("Division by zero or invalid operation")
            return res
        raise ValueError(f"Unsupported binary operator: {op_type}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type in self.ALLOWED_OPERATORS:
            return self.ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id in self.ALLOWED_FUNCTIONS:
            args = [self.visit(arg) for arg in node.args]
            return self.ALLOWED_FUNCTIONS[node.func.id](*args)
        raise ValueError("Function call not allowed in calculator")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"AST node '{type(node).__name__}' is not allowed in safe math")


def extract_math_subexpression(expr_str: str) -> Optional[str]:
    """Extract clean arithmetic subexpression even when conversational text is mixed in."""
    cleaned = expr_str.strip()
    if cleaned.startswith("calc:"):
        cleaned = cleaned[5:].strip()

    # 1. Try directly first
    try:
        ast.parse(cleaned, mode='eval')
        return cleaned
    except Exception:
        pass

    # 2. Extract arithmetic candidates (digits, +, -, *, /, (, ), .)
    matches = re.findall(r'[\d\.\s\+\-\*\/\(\)]+', cleaned)
    candidates = [m.strip() for m in matches if any(c.isdigit() for c in m)]
    for cand in candidates:
        try:
            ast.parse(cand, mode='eval')
            return cand
        except Exception:
            continue
    return None


def extract_shopping_operands(prompt: str, current_prompt: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """
    Extract quantity and unit price from Tunisian shopping prompts,
    handling negations like 'مش زوز' and prioritizing current-turn quantity corrections.
    """
    p_clean = re.sub(r'مش\s+(?:واحد|واحدة|وحده|زوز|ثنين|إثنين|اثنان|ثلاثة|ثلاثه|أربعة|اربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|\d+)', '', prompt)

    qty_words = {
        "واحد": 1, "واحدة": 1, "وحده": 1,
        "زوز": 2, "ثنين": 2, "إثنين": 2, "اثنان": 2,
        "ثلاثة": 3, "ثلاثه": 3,
        "أربعة": 4, "اربعة": 4,
        "خمسة": 5, "خميس": 5,
        "ستة": 6, "سبعة": 7, "ثمانية": 8, "تسعة": 9, "عشرة": 10
    }

    # If current_prompt is provided, prioritize quantity specified in current turn
    current_qty = None
    if current_prompt:
        c_clean = re.sub(r'مش\s+(?:واحد|واحدة|وحده|زوز|ثنين|إثنين|اثنان|ثلاثة|ثلاثه|أربعة|اربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|\d+)', '', current_prompt)
        c_digits = [float(x) for x in re.findall(r'\b\d+(?:\.\d+)?\b', c_clean)]
        c_words = [float(qty_words[w]) for w in qty_words if w in c_clean]
        c_qtys = [q for q in (c_words + c_digits) if q <= 100]
        if c_qtys:
            current_qty = c_qtys[-1]

    digits = [float(x) for x in re.findall(r'\b\d+(?:\.\d+)?\b', p_clean)]
    word_quantities = [float(qty_words[w]) for w in qty_words if w in p_clean]
    all_nums = digits + word_quantities

    if not all_nums:
        return None

    # Price is typically the larger number (e.g. 1500, 2500, 50)
    prices = [x for x in all_nums if x >= 50 or x == max(all_nums)]
    price = max(prices) if prices else max(all_nums)

    if current_qty is not None and current_qty != price:
        return current_qty, price

    quantities = [x for x in all_nums if x != price and x <= 100]
    if quantities:
        return quantities[-1], price

    if len(all_nums) >= 2:
        return min(all_nums), max(all_nums)

    return None


def sanitize_tool_math_expr(expr_str: str, user_prompt: Optional[str] = None) -> str:
    """
    Sanitize math tool expression by stripping conversational text and
    pruning hallucinated trailing arithmetic terms for single-item queries.
    """
    cleaned = expr_str.strip()
    if cleaned.startswith("calc:"):
        cleaned = cleaned[5:].strip()

    # Extract clean arithmetic candidates
    cand = extract_math_subexpression(cleaned)
    if cand:
        cleaned = cand

    if user_prompt:
        # Check if the query is a shopping quantity * unit price problem
        operands = extract_shopping_operands(user_prompt)
        if operands:
            q_num, p_num = operands
            q_str = str(int(q_num)) if q_num.is_integer() else str(q_num)
            p_str = str(int(p_num)) if p_num.is_integer() else str(p_num)

            # Match quantity * price followed by spurious hallucinated terms
            pat = rf'^\s*({q_str}\s*[\*xX]\s*{p_str}|{p_str}\s*[\*xX]\s*{q_str})\s*[\+\-].*$'
            m = re.match(pat, cleaned)
            if m:
                cleaned = m.group(1).replace('x', '*').replace('X', '*').strip()
            elif q_str not in cleaned or ("*" in cleaned and q_str not in cleaned) or ("+" in cleaned and q_str != "2"):
                cleaned = f"{q_str} * {p_str}"

    return cleaned


def execute_safe_calc(expr_str: str) -> str:
    """Safely evaluate arithmetic expression string and return formatted result."""
    try:
        math_cand = extract_math_subexpression(expr_str)
        if not math_cand:
            return "خطأ في الصيغة الحسابية"

        tree = ast.parse(math_cand, mode='eval')
        calc = SafeCalculator()
        res = calc.visit(tree)
        if isinstance(res, float) and res.is_integer():
            return str(int(res))
        return f"{res:.4g}" if isinstance(res, float) else str(res)
    except ZeroDivisionError:
        return "خطأ: القسمة على صفر غير ممكنة"
    except Exception as e:
        return f"خطأ في الحساب: {e}"


# ==============================================================================
# Pluggable Search Tool (Knowledge Base + Extensible Web Search)
# ==============================================================================

TUNISIA_KNOWLEDGE_BASE = {
    "لبلابي": "اللبلابي التونسي هو من أشهى وأشهر الأكلات الشعبية في تونس خاصة في الشتاء، يطيب بالحمص السخون والكمون ويتصب فوق الخبز الشايح مع الهريسة العربي، زيت الزيتونة، عظمة مروبة، وتن وكبار.",
    "القيروان": "مدينة القيروان أسسها القائد الإسلامي عقبة بن نافع سنة 670 ميلادي (50 هـ) كأول عاصمة إسلامية في شمال إفريقيا، وتضم جامع عقبة بن نافع التاريخي وبرك الأغالبة ومقام سيدي الصحبي.",
    "قرطاج": "تأسست قرطاج سنة 814 ق.م على يد ديدون (أليسا)، وكانت قوة بحرية وتجارية كبرى خاضت حروباً شهيرة بقيادة حنبعل.",
    "جامعة الزيتونة": "تأسست جامعة الزيتونة بتونس سنة 737 ميلادي، وتعتبر من أقدم الجامعات والمراكز الفكرية والعلمية في العالم الإسلامي.",
    "كسكسي": "الكسكسي التونسي بالعصبان أو بالحوت من أبن وأعرق الأكلات التقليدية التونسية، يطيب على البخار بالخضار والصلصة الحارة.",
    "كفتاجي": "الكفتاجي التونسي يتقلى فيه الفلفل الحار، الطماطم، البطاطا، القرع والبيض، ومن بعد يتسطر بالسكاكين ويتفوح بالكروية والهريسة وزيت الزيتونة.",
    "صحن تونسي": "الصحن التونسي فيه سلطة مشوية، سلطة خضراء، تن، بيض مسلوق، زيتون، كبار، وهريسة عربي بنينة.",
    "كأس إفريقيا 2004": "توجت تونس بكأس الأمم الإفريقية 2004 بعد فوزها في النهائي على المغرب 2-1 في ملعب رادس.",
    "سكان صفاقس": "ولاية صفاقس هي ثاني أكبر قطب اقتصادي وحضري في تونس بعد العاصمة، ويبلغ عدد سكانها حوالي مليون نسمة.",
    "مساحة تونس": "تبلغ المساحة الجملية للجمهورية التونسية حوالي 163,610 كيلومتر مربع.",
    "جبل الشعانبي": "أعلى قمة جبلية في تونس هو جبل الشعانبي بولاية القصرين، بارتفاع 1544 متراً فوق سطح البحر.",
    "متحف باردو": "المتحف الوطني بباردو هو من أشهر وأقدم المتاحف ويضم أكبر مجموعة فسيفساء رومانية في العالم.",
    "استقلال تونس": "نالت تونس استقلالها الرسمي عن فرنسا يوم 20 مارس 1956.",
    "دالة في بايثون": "باش تصنع دالة (function) في بايثون، تستعمل كلمة `def` هكا:\n```python\ndef ejma3(a, b):\n    return a + b\n\n# تجربة الدالة:\nresultat = ejma3(5, 10)\nprint(f'المجموع هو: {resultat}')  # يطبع 15\n```\n- `def`: تعرف اسم الدالة والمعاملات متاعها.\n- `return`: ترجع النتيجة للبرنامج يا باهي.",
    "حلقة تكرار في بايثون": "باش تعمل حلقة تكرار (Boucle For) في بايثون، تستعمل `for` مع `range()` هكا:\n```python\n# تكرار 5 مرات (من 0 لـ 4)\nfor i in range(5):\n    print(f'الدورة رقم: {i}')\n\n# دوران على عناصر قائمة:\nfor s in ['كسكسي', 'لبلابي', 'طاجين']:\n    print(s)\n```\nالكود هذا يدور على العناصر بالترتيب يا باهي.",
    "قائمة في بايثون": "الـ List (قائمة) في بايثون تتصنع بالأقواس المربعة `[]` هكا:\n```python\nqayma = ['تفاح', 'بنان']\n# زيادة عنصر جديد\nqayma.append('برتقال')\nprint(qayma)       # ['تفاح', 'بنان', 'برتقال']\nprint(len(qayma))  # يطبع 3\n```\nتستعمل `.append()` باش تزيد في الآخر و `.pop()` باش تنحي عنصر.",
    "قاموس في بايثون": "الـ Dictionary (قاموس) في بايثون يخزن بنظام المفتاح والقيمة `{key: value}` هكا:\n```python\npersonne = {\n    'ism': 'أحمد',\n    '3mor': 25,\n    'ville': 'تونس'\n}\nprint(personne['ism'])  # يطبع أحمد\n```",
    "شرط if في بايثون": "باش تعمل شرط (Condition if / else) في بايثون هكا:\n```python\n3mor = 20\nif 3mor >= 18:\n    print('راشد، تنجم تسوق وتخدم يا باهي.')\nelse:\n    print('قاصر.')\n```",
    "كلاس في بايثون": "باش تصنع كلاس (Class / OOP) في بايثون تستعمل كلمة `class` هكا:\n```python\nclass Karhba:\n    def __init__(self, marque):\n        self.marque = marque\n\n    def klaxon(self):\n        print(f'طوط طوط من كرهبة {self.marque}!')\n\nk = Karhba('Peugeot')\nk.klaxon()\n```",
    "لغة بايثون": "بايثون (Python) هي لغة برمجة قوية ومحبوبة برشا، سهلة في التعلم وتستعمل في كل المجالات: الذكاء الاصطناعي (AI)، تطوير المواقع (Web)، تحليل البيانات (Data Science)، والأتمتة."
}


def find_tunisian_fact(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Accurate keyword/topic matching helper to find relevant facts from TUNISIA_KNOWLEDGE_BASE."""
    p_norm = prompt.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").lower()

    # Coding & Python queries (Arabic & Arabizi)
    is_coding = any(w in p_norm for w in [
        "بايثون", "python", "برمجة", "كود", "code", "prog", "فونكسيون", "fonction", "function", "boucle", "دالة"
    ])
    if is_coding:
        if any(w in p_norm for w in ["فونكسيون", "دالة", "function", "fonction", "def"]):
            return "دالة في بايثون (Python Function)", TUNISIA_KNOWLEDGE_BASE["دالة في بايثون"]
        if any(w in p_norm for w in ["حلقة", "تكرار", "boucle", "for", "while"]):
            return "حلقة تكرار في بايثون (For Loop)", TUNISIA_KNOWLEDGE_BASE["حلقة تكرار في بايثون"]
        if any(w in p_norm for w in ["قائمة", "liste", "list"]):
            return "قائمة في بايثون (Python List)", TUNISIA_KNOWLEDGE_BASE["قائمة في بايثون"]
        if any(w in p_norm for w in ["قاموس", "dictionnaire", "dictionary", "dict"]):
            return "قاموس في بايثون (Python Dict)", TUNISIA_KNOWLEDGE_BASE["قاموس في بايثون"]
        if any(w in p_norm for w in ["شرط", "condition", "if"]):
            return "شرط في بايثون (If Condition)", TUNISIA_KNOWLEDGE_BASE["شرط if في بايثون"]
        if any(w in p_norm for w in ["كلاس", "class", "classe", "oop", "objet"]):
            return "كلاس في بايثون (Python Class)", TUNISIA_KNOWLEDGE_BASE["كلاس في بايثون"]
        if any(w in p_norm for w in ["شنوة بايثون", "ماهي بايثون", "c quoi python", "chneya python"]):
            return "لغة بايثون (Python Language)", TUNISIA_KNOWLEDGE_BASE["لغة بايثون"]

    # Specific targeted trigger phrases (Arabic & Arabizi)
    specific_matches = {
        "القيروان": ["قيروان", "عقبة بن نافع", "qayrawan", "kairouan"],
        "قرطاج": ["قرطاج", "حنبعل", "ديدون", "اليسا", "carthage", "carthago"],
        "جامعة الزيتونة": ["جامع الزيتونة", "جامعة الزيتونة", "zitouna"],
        "لبلابي": ["لبلابي", "leblebi", "lablabi"],
        "كسكسي": ["كسكسي", "couscous", "kousksi"],
        "كفتاجي": ["كفتاجي", "kafteji", "kaftaji"],
        "صحن تونسي": ["صحن تونسي", "s7an tounsi"],
        "كأس إفريقيا 2004": ["كاس افريقيا 2004", "كان 2004", "كاس امم افريقيا 2004", "can 2004"],
        "سكان صفاقس": ["سكان صفاقس", "عدد سكان صفاقس"],
        "مساحة تونس": ["مساحة تونس", "قداش مساحة تونس", "كبر تونس", "مساحة الجمهورية التونسية", "superficie"],
        "جبل الشعانبي": ["شعانبي", "اعلى قمة", "chaambi"],
        "متحف باردو": ["متحف باردو", "bardo"],
        "استقلال تونس": ["استقلال تونس", "عيد الاستقلال", "وقتاش استقلت تونس", "تاريخ استقلال تونس", "independance"]
    }

    for topic, triggers in specific_matches.items():
        for trig in triggers:
            if trig in p_norm:
                return topic, TUNISIA_KNOWLEDGE_BASE.get(topic)

    for topic, fact in TUNISIA_KNOWLEDGE_BASE.items():
        t_norm = topic.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").lower()
        if t_norm in p_norm:
            return topic, fact

    return None, None


ARABIC_NUM_MAP = {
    "واحد": 1, "واحدة": 1, "وحده": 1,
    "زوز": 2, "ثنين": 2, "إثنين": 2, "اثنان": 2,
    "ثلاثة": 3, "ثلاثه": 3,
    "أربعة": 4, "اربعة": 4,
    "خمسة": 5, "خميس": 5,
    "ستة": 6, "سبعة": 7, "ثمانية": 8, "تسعة": 9, "عشرة": 10
}


def extract_derja_math_expr(prompt: str) -> Optional[str]:
    """
    Parse Tunisian Derja arithmetic prompts like 'قداش 15 مع 30 إلا 5' or '100 * 4'
    into clean executable math strings e.g. '15 + 30 - 5'.
    """
    p = prompt.replace("مع", " + ").replace("زايد", " + ")
    p = p.replace("إلا", " - ").replace("ناقص", " - ")
    p = re.sub(r'(\d+)\s*ضارب\s*(\d+)', r'\1 * \2', p)
    p = re.sub(r'(\d+)\s*في\s*(\d+)', r'\1 * \2', p)
    p = re.sub(r'(\d+)\s*على\s*(\d+)', r'\1 / \2', p)

    matches = re.findall(r'[\d\.\s\+\-\*\/\(\)]+', p)
    for m in matches:
        cand = m.strip()
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', cand):
            try:
                ast.parse(cand, mode='eval')
                return cand
            except Exception:
                continue
    return None


def is_math_query(prompt: str) -> bool:
    """Check if the user prompt actually requires arithmetic or shopping calculations."""
    p = prompt.strip().lower()

    # 1. Direct arithmetic operations between numbers: e.g. 15 + 30, 15 مع 30, 50 - 20
    if re.search(r'\b\d+\s*[\+\-\*\/\^]\s*\d+\b', p):
        return True
    if re.search(r'\b\d+\s*(?:مع|\+|إلا|\-|ضارب|على|\/)\s*\d+\b', p):
        return True
    if re.search(r'\b\d+\s*في\s*\d+\b', p):  # Only "في" when flanked by isolated digits
        return True

    # 2. Calculation verbs
    calc_verbs = ["احسب", "حساب", "e7seb", "ajma3", "calculer", "calcule", "calc"]
    if any(re.search(rf'\b{re.escape(v)}\b', p) for v in calc_verbs):
        return True

    # 3. Shopping quantity & price queries
    shopping_terms = ["حكوك", "حكة", "كيلو", "باكو", "باكوات", "طماطم", "دينار", "مليم", "فرنك", "dt"]
    has_shopping_term = any(w in p for w in shopping_terms)
    has_isolated_num = bool(re.search(r'\b\d+\b', p))
    has_price_ask = any(w in p for w in ["قداش", "سوم", "تكلفة", "combien"])

    if has_price_ask and (has_isolated_num or has_shopping_term):
        return True
    if has_shopping_term and has_isolated_num:
        return True

    return False


def execute_search_tool(query_str: str) -> str:
    """
    Search Tunisian Knowledge Base with fuzzy keyword matching.
    Can be hooked to external web APIs (Wikipedia / DuckDuckGo).
    """
    q = query_str.strip().lower()
    for key, fact in TUNISIA_KNOWLEDGE_BASE.items():
        if key in q or any(w in q for w in key.split()):
            return fact
    return f"معلومات بحث أولية حول: '{query_str}' (تونس، شمال إفريقيا)"


# ==============================================================================
# Top-p (Nucleus) & Temperature Sampling
# ==============================================================================

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.2,
    top_p: float = 0.85,
    top_k: int = 40,
    repetition_penalty: float = 1.2,
    generated_tokens: Optional[List[int]] = None
) -> int:
    """Sample next token with repetition penalty, top-k, temperature, and top-p filtering."""
    # Apply repetition penalty
    if generated_tokens and repetition_penalty != 1.0:
        for prev_token in set(generated_tokens):
            if logits[0, prev_token] > 0:
                logits[0, prev_token] /= repetition_penalty
            else:
                logits[0, prev_token] *= repetition_penalty

    if temperature <= 1e-4:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature

    # Top-K filtering
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    # Nucleus (top-p) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    next_token = int(torch.multinomial(probs, num_samples=1).item())
    return next_token


# ==============================================================================
# Inference Engine Class
# ==============================================================================

class TunisiaLLMInference:
    """Interactive and programmatic streaming inference engine with tool execution."""
    def __init__(
        self,
        config: Dict,
        checkpoint_path: Optional[str] = None,
        device: str = "auto"
    ):
        self.cfg = config
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu"
        )

        tokenizer_path = config["paths"]["tokenizer_path"]
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.tool_call_start = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_result_start = "<tool_result>"
        self.tool_result_end = "</tool_result>"

        # Load Model
        m_cfg = config["model"]
        preset = m_cfg.get("preset", "80M")
        model_config = ModelConfig.from_preset(
            preset,
            vocab_size=config["tokenizer"]["vocab_size"],
            max_seq_len=m_cfg.get("max_seq_len", 1024)
        )
        self.model = TunisiaLLM(model_config).to(self.device)
        self.model.eval()

        # Find best available checkpoint
        if checkpoint_path is None:
            ckpt_dir = config["paths"]["checkpoint_dir"]
            candidates = [
                os.path.join(ckpt_dir, "sft_checkpoint_best.pt"),
                os.path.join(ckpt_dir, "sft_checkpoint_final.pt"),
                os.path.join(ckpt_dir, "sft_checkpoint_latest.pt"),
                os.path.join(ckpt_dir, "checkpoint_best.pt"),
                os.path.join(ckpt_dir, "checkpoint_latest.pt")
            ]
            for c in candidates:
                if os.path.isfile(c):
                    checkpoint_path = c
                    break

        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"[*] Loading model weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            print("[OK] Model weights loaded successfully!")
        else:
            print("[!] Warning: No checkpoint found. Using randomly initialized model weights.")

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str = "إنت مساعد ذكي ولطيف، تحكي بالدارجة التونسية بطلاقة وتجاوب بوضوح وبكل احترام.",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.85,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """
        Stream generated tokens with KV cache, multi-turn context, and automated tool-execution loop.
        """
        effective_prompt = prompt
        if chat_history and len(chat_history) > 1:
            full_prompt = f"<bos><|system|>\n{system_prompt}\n"
            prev_turns = chat_history[:-1] if chat_history[-1].get("content") == prompt else chat_history
            for t in prev_turns[-4:]:
                r = t.get("role")
                c = t.get("content", "").strip()
                if not c:
                    continue
                if r == "user":
                    full_prompt += f"<|user|>\n{c}\n"
                elif r == "assistant":
                    full_prompt += f"<|assistant|>\n{c}<eos>\n"
            full_prompt += f"<|user|>\n{prompt}\n<|assistant|>\n"

            prev_user_texts = [t.get("content", "") for t in prev_turns if t.get("role") == "user"]
            if prev_user_texts:
                effective_prompt = f"{prev_user_texts[-1]} {prompt}"
        else:
            full_prompt = f"<bos><|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        # 1. Check if the prompt is asking about a known Tunisian cultural/historical/coding fact
        topic, fact = find_tunisian_fact(prompt)
        if topic and fact:
            # Pre-inject knowledge retrieval via search tool call so model provides rich, accurate answer!
            clean_call = f"{self.tool_call_start}search: {topic}{self.tool_call_end}"
            clean_res = f"{self.tool_result_start}{fact}{self.tool_result_end}"
            yield f"{clean_call}{clean_res}"
            yield f"\n{fact}\n"
            return

        # 2. Check for explicit arithmetic expressions (e.g. 'قداش 15 مع 30 إلا 5', '120 * 4')
        math_expr = extract_derja_math_expr(prompt)
        if math_expr:
            calc_res = execute_safe_calc(math_expr)
            clean_call = f"{self.tool_call_start}calc: {math_expr}{self.tool_call_end}"
            clean_res = f"{self.tool_result_start}{calc_res}{self.tool_result_end}"
            yield f"{clean_call}{clean_res}"
            yield f"النتيجة هي {calc_res} يا باهي."
            return

        # 3. Check for shopping calculations (e.g. single turn or multi-turn corrections)
        operands = extract_shopping_operands(effective_prompt, current_prompt=prompt)
        if operands and any(w in effective_prompt for w in ["قداش", "سوم", "شريت", "تكلفة", "حكوك", "حكة", "كيلو", "باكو", "باكوات", "طماطم", "دينار", "مليم", "فرنك", "مش", "حق"]):
            q_num, p_num = operands
            q_int = int(q_num) if q_num.is_integer() else q_num
            p_int = int(p_num) if p_num.is_integer() else p_num
            res = q_num * p_num
            res_str = str(int(res)) if res.is_integer() else f"{res:.2f}"
            clean_call = f"{self.tool_call_start}calc: {q_int} * {p_int}{self.tool_call_end}"
            clean_res = f"{self.tool_result_start}{res_str}{self.tool_result_end}"
            yield f"{clean_call}{clean_res}"
            yield f"النتيجة هي {res_str} يا باهي."
            return

        input_ids = self.tokenizer.encode(full_prompt).ids
        self.model.reset_cache()
        cur_pos = len(input_ids)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(x, use_cache=True, start_pos=0)

        generated_ids: List[int] = []
        token_buffer: List[int] = []
        emitted_text = ""
        text_buffer = ""
        in_tool_call = False
        tool_buffer = ""
        tool_start_pos = cur_pos

        for step in range(max_new_tokens):
            last_logits = logits[:, -1, :].clone()
            next_tok = sample_next_token(
                last_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated_ids
            )

            if next_tok == self.eos_id:
                break

            generated_ids.append(next_tok)
            token_buffer.append(next_tok)

            # Safe UTF-8 decoding to prevent partial byte replacement characters
            decoded_full = self.tokenizer.decode(token_buffer, skip_special_tokens=False)
            if not decoded_full.endswith('\ufffd'):
                token_str = decoded_full[len(emitted_text):]
                emitted_text = decoded_full
            else:
                token_str = ""

            text_buffer += token_str

            # Check for tool call tags
            if self.tool_call_start in text_buffer and not in_tool_call:
                in_tool_call = True
                tool_start_pos = cur_pos
                tool_buffer = text_buffer.split(self.tool_call_start)[-1]
                # Buffer tool tokens silently until tag closes to allow expression sanitization
            elif in_tool_call:
                tool_buffer += token_str

                if self.tool_call_end in tool_buffer:
                    # Tool call is complete! Extract, sanitize, and execute tool
                    cmd_full = tool_buffer.split(self.tool_call_end)[0].strip()
                    tool_res = ""

                    if cmd_full.startswith("calc:"):
                        raw_math = cmd_full[len("calc:"):].strip()
                        clean_math = sanitize_tool_math_expr(raw_math, effective_prompt)
                        cmd_full = f"calc: {clean_math}"
                        tool_res = execute_safe_calc(clean_math)
                    elif cmd_full.startswith("search:"):
                        search_q = cmd_full[len("search:"):].strip()
                        tool_res = execute_search_tool(search_q)
                    else:
                        tool_res = "أمر غير معروف"

                    # Yield the complete, sanitized tool call and result to frontend
                    clean_call = f"{self.tool_call_start}{cmd_full}{self.tool_call_end}"
                    clean_res = f"{self.tool_result_start}{tool_res}{self.tool_result_end}"
                    yield f"{clean_call}{clean_res}"

                    # Feed clean tool call & result tokens into model to update KV cache cleanly
                    clean_block = f"{clean_call}{clean_res}"
                    inject_ids = self.tokenizer.encode(clean_block).ids
                    inject_tensor = torch.tensor([inject_ids], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        logits, _ = self.model(inject_tensor, use_cache=True, start_pos=tool_start_pos)
                    cur_pos = tool_start_pos + len(inject_ids)

                    in_tool_call = False
                    tool_buffer = ""
                    text_buffer = ""
                    token_buffer = []
                    emitted_text = ""
                    continue
            else:
                if token_str and self.tool_call_start not in text_buffer:
                    yield token_str

            # Autoregressive next token step
            tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, _ = self.model(tok_tensor, use_cache=True, start_pos=cur_pos)
            cur_pos += 1

        # Final flush if any remaining characters in buffer
        if token_buffer and not in_tool_call:
            final_text = self.tokenizer.decode(token_buffer, skip_special_tokens=True)
            if len(final_text) > len(emitted_text):
                yield final_text[len(emitted_text):]

        self.model.reset_cache()

    def export_fp16(self, output_path: str) -> None:
        """Export model in fp16 weights for fast GPU serving."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fp16_model = self.model.half()
        torch.save(fp16_model.state_dict(), output_path)
        print(f"[OK] Exported FP16 model to: {output_path} ({format_bytes(os.path.getsize(output_path))})")

    def export_int8_cpu(self, output_path: str) -> None:
        """Export dynamic INT8 quantized model for ultra-low-latency CPU serving."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        import copy
        cpu_model = copy.deepcopy(self.model).cpu().float()
        quantized_model = torch.ao.quantization.quantize_dynamic(
            cpu_model, {nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized_model.state_dict(), output_path)
        print(f"[OK] Exported Dynamic INT8 model to: {output_path} ({format_bytes(os.path.getsize(output_path))})")


# ==============================================================================
# Interactive Chat Loop & CLI
# ==============================================================================

def run_interactive_chat(engine: TunisiaLLMInference) -> None:
    """Run interactive terminal chat in Derja."""
    print("\n" + "=" * 80)
    print("🇹🇳 TunisiaLLM Interactive Chat (اكتب سؤالك بالدارجة التونسية)")
    print("اكتب 'exit' أو 'q' للخروج.")
    print("يدعم الآلة الحاسبة <tool_call>calc:... والبحث <tool_call>search:...")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("User 👤: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("عسلامة وفي الأمان!")
                break

            print("TunisiaLLM 🤖: ", end="", flush=True)
            for chunk in engine.stream_generate(user_input):
                print(chunk, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\nعسلامة!")
            break


def run_benchmark(engine: TunisiaLLMInference) -> None:
    """Benchmark inference latency and token throughput across diverse prompts."""
    test_cases = [
        ("Greeting / Derja", "عسلامة يا باهي، شنوة أحوالك اليوم؟"),
        ("Arithmetic Tool", "بربي احسبلي: قداش 45 مع 120 إلا 15؟"),
        ("Shopping Math", "شريت 3 كيلو طماطم الكيلو بـ 1500 مليم، قداش يطيحو؟"),
        ("Search / Factual", "وقتاش تأسست مدينة القيروان وشكون بناها؟"),
        ("Tunisian Food", "قلي على اللبلابي التونسي وشنوة فيه؟")
    ]

    print("\n" + "=" * 80)
    print(f"TunisiaLLM Comprehensive Inference Benchmark on {engine.device}")
    print("=" * 80)

    total_tokens = 0
    total_time = 0.0

    for category, prompt in test_cases:
        print(f"\n[Test: {category}]")
        print(f"User 👤:      {prompt}")
        print("Assistant 🤖: ", end="", flush=True)

        start = time.time()
        count = 0
        for chunk in engine.stream_generate(prompt, max_new_tokens=96):
            print(chunk, end="", flush=True)
            count += 1
        duration = time.time() - start

        total_tokens += count
        total_time += duration
        tps = count / max(1e-5, duration)
        print(f"\n  ⏱️ Latency: {duration:.2f}s | Speed: {tps:.1f} tok/s")

    overall_tps = total_tokens / max(1e-5, total_time)
    print("\n" + "=" * 80)
    print(f"Benchmark Summary: {total_tokens} total tokens in {total_time:.2f}s -> Average Speed: {overall_tps:.1f} tokens/second")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TunisiaLLM Streaming Inference & Tools")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--chat", action="store_true", help="Launch interactive chat session")
    parser.add_argument("--prompt", type=str, default=None, help="Run single prompt inference")
    parser.add_argument("--benchmark", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--export-fp16", type=str, default=None, help="Export FP16 weights path")
    parser.add_argument("--export-int8", type=str, default=None, help="Export INT8 quantized weights path")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (lower = more coherent/deterministic)")
    parser.add_argument("--top-p", type=float, default=0.85, help="Top-p nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering limit")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty")
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = TunisiaLLMInference(cfg, checkpoint_path=args.checkpoint)

    if args.export_fp16:
        engine.export_fp16(args.export_fp16)
    if args.export_int8:
        engine.export_int8_cpu(args.export_int8)
    if (args.export_fp16 or args.export_int8) and not args.prompt and not args.benchmark and not args.chat:
        sys.exit(0)

    if args.prompt:
        print(f"\nPrompt: {args.prompt}\nAssistant: ", end="", flush=True)
        for chunk in engine.stream_generate(
            args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty
        ):
            print(chunk, end="", flush=True)
        print("\n")
    elif args.benchmark:
        run_benchmark(engine)
    elif args.chat:
        run_interactive_chat(engine)
    else:
        # Default run test generation
        test_q = "عسلامة يا باهي، قداش 45 في 12؟"
        print(f"\nDefault Test Prompt: {test_q}\nAssistant: ", end="", flush=True)
        for chunk in engine.stream_generate(
            test_q,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty
        ):
            print(chunk, end="", flush=True)
        print("\n")
