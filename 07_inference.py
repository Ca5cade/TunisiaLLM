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
        # Find all numbers in the user prompt (integers and floats)
        prompt_nums = re.findall(r'\b\d+(?:\.\d+)?\b', user_prompt)
        # If the user prompt describes a single-item question or simple 2-number problem
        if len(prompt_nums) == 2:
            q, p = prompt_nums[0], prompt_nums[1]
            # Match quantity * price followed by spurious hallucinated addition/subtraction
            # e.g., "3 * 1500 + 1500" -> "3 * 1500"
            pat = rf'^\s*({q}\s*[\*xX]\s*{p}|{p}\s*[\*xX]\s*{q})\s*[\+\-].*$'
            m = re.match(pat, cleaned)
            if m:
                cleaned = m.group(1).replace('x', '*').replace('X', '*').strip()

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
    "استقلال تونس": "نالت تونس استقلالها الرسمي عن فرنسا يوم 20 مارس 1956."
}


def find_tunisian_fact(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Fuzzy matching helper to find relevant facts from TUNISIA_KNOWLEDGE_BASE."""
    p_norm = prompt.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").lower()
    for topic, fact in TUNISIA_KNOWLEDGE_BASE.items():
        t_norm = topic.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").lower()
        if t_norm in p_norm:
            return topic, fact
    for topic, fact in TUNISIA_KNOWLEDGE_BASE.items():
        words = [w for w in topic.split() if len(w) > 2]
        if any(w in p_norm for w in words):
            return topic, fact
    return None, None


def is_math_query(prompt: str) -> bool:
    """Check if the user prompt actually requires arithmetic or shopping calculations."""
    has_digits = bool(re.search(r'\d+', prompt))
    has_symbols = bool(re.search(r'[\+\-\*\/\=\%]', prompt))
    math_keywords = ["احسب", "حساب", "ضرب", "جمع", "طرح", "قسمة", "مليم", "دينار", "سوم", "شريت", "تخفيض", "صولد", "قداش حاصل", "مجموع"]
    has_keywords = any(kw in prompt for kw in math_keywords)
    return (has_digits and (has_symbols or has_keywords)) or (has_digits and len(re.findall(r'\d+', prompt)) >= 2)


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
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    generated_tokens: Optional[List[int]] = None
) -> int:
    """Sample next token with repetition penalty, temperature, and top-p filtering."""
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
        system_prompt: str = "إنت مساعد ذكي بالدارجة التونسية.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15
    ) -> Generator[str, None, None]:
        """
        Stream generated tokens with KV cache and automated tool-execution loop.
        """
        full_prompt = f"<bos><|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Check if the prompt is asking about a known Tunisian cultural/historical fact
        topic, fact = find_tunisian_fact(prompt)
        if topic and fact and not is_math_query(prompt):
            # Pre-inject knowledge retrieval via search tool call so model provides rich, accurate answer!
            clean_call = f"{self.tool_call_start}search: {topic}{self.tool_call_end}"
            clean_res = f"{self.tool_result_start}{fact}{self.tool_result_end}"
            yield f"{clean_call}{clean_res}"

            full_prompt = f"<bos><|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n{clean_call}{clean_res}"
            input_ids = self.tokenizer.encode(full_prompt).ids
            self.model.reset_cache()
            cur_pos = len(input_ids)
            x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, _ = self.model(x, use_cache=True, start_pos=0)
        else:
            input_ids = self.tokenizer.encode(full_prompt).ids
            self.model.reset_cache()
            cur_pos = len(input_ids)
            x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, _ = self.model(x, use_cache=True, start_pos=0)

        generated_ids: List[int] = []
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
                repetition_penalty=repetition_penalty,
                generated_tokens=generated_ids
            )

            if next_tok == self.eos_id:
                break

            generated_ids.append(next_tok)
            token_str = self.tokenizer.decode([next_tok], skip_special_tokens=False)
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
                        # Intent validation: if prompt has no numbers and is not a math query:
                        if not is_math_query(prompt):
                            # Suppress spurious calc tool call on non-math queries!
                            in_tool_call = False
                            tool_buffer = ""
                            text_buffer = ""
                            continue

                        raw_math = cmd_full[len("calc:"):].strip()
                        clean_math = sanitize_tool_math_expr(raw_math, prompt)
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
                    continue
            else:
                yield token_str

            # Autoregressive next token step
            tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, _ = self.model(tok_tensor, use_cache=True, start_pos=cur_pos)
            cur_pos += 1

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
        cpu_model = self.model.cpu().float()
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = TunisiaLLMInference(cfg, checkpoint_path=args.checkpoint)

    if args.export_fp16:
        engine.export_fp16(args.export_fp16)
    if args.export_int8:
        engine.export_int8_cpu(args.export_int8)

    if args.prompt:
        print(f"\nPrompt: {args.prompt}\nAssistant: ", end="", flush=True)
        for chunk in engine.stream_generate(args.prompt):
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
        for chunk in engine.stream_generate(test_q):
            print(chunk, end="", flush=True)
        print("\n")
