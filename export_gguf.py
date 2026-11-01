"""
export_gguf.py - High-Performance GGUF & Ollama Exporter for TunisiaLLM
Converts PyTorch checkpoint into GGUF format compatible with llama.cpp and Ollama.
Generates an Ollama Modelfile for 1-click local serving.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType

sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def export_to_gguf(
    checkpoint_path: str,
    tokenizer_path: str,
    output_gguf_path: str,
    dtype: str = "f16"
) -> None:
    print("=" * 80)
    print("TUNISIALLM: GGUF & OLLAMA EXPORT ENGINE")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output GGUF: {output_gguf_path}")
    print(f"Format: {dtype.upper()}")
    print("=" * 80)

    os.makedirs(os.path.dirname(os.path.abspath(output_gguf_path)), exist_ok=True)

    # 1. Load weights
    print("[*] Loading checkpoint weights...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # 2. Load tokenizer tokens
    print("[*] Loading vocabulary from tokenizer.json...")
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    vocab = tok_data.get("model", {}).get("vocab", {})
    # Sort tokens by ID
    tokens = [None] * len(vocab)
    for t, tid in vocab.items():
        if tid < len(tokens):
            tokens[tid] = t

    tokens = [t if t is not None else f"<unk_{i}>" for i, t in enumerate(tokens)]
    print(f"[+] Loaded {len(tokens)} tokens.")

    # 3. Detect architecture dimensions dynamically from checkpoint
    dim = state["tok_embeddings.weight"].shape[1]
    layer_indices = [int(k.split(".")[1]) for k in state if k.startswith("layers.") and ".attention." in k]
    n_layers = max(layer_indices) + 1 if layer_indices else 12
    hidden_dim = state["layers.0.feed_forward.w1.weight"].shape[0] if "layers.0.feed_forward.w1.weight" in state else 2176

    # Infer preset or heads
    if dim == 1280 and n_layers == 24:
        model_name = "TunisiaLLM-500M"
        n_heads = 20
        n_kv_heads = 4
    elif dim == 768:
        model_name = "TunisiaLLM-125M"
        n_heads = 12
        n_kv_heads = 4
    elif dim == 448:
        model_name = "TunisiaLLM-30M"
        n_heads = 7
        n_kv_heads = 1
    else:
        model_name = "TunisiaLLM-80M"
        n_heads = 12
        n_kv_heads = 4

    head_dim = dim // n_heads

    # Initialize GGUF Writer
    writer = GGUFWriter(output_gguf_path, "llama")

    # Metadata
    writer.add_name(model_name)
    writer.add_description(f"Tunisian Derja {model_name} LLaMA decoder with tool calling and DPO alignment")
    writer.add_architecture()
    writer.add_block_count(n_layers)
    writer.add_context_length(1024)
    writer.add_embedding_length(dim)
    writer.add_feed_forward_length(hidden_dim)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(10000.0)

    # Special tokens
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(0)

    # Tokenizer model
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)

    # Map state dict tensor names to GGML tensor names
    tensor_map = {
        "tok_embeddings.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight"
    }

    # Add output weight (tied or explicit)
    if "output.weight" in state:
        tensor_map["output.weight"] = "output.weight"

    for i in range(n_layers):
        tensor_map[f"layers.{i}.attention.wq.weight"] = f"blk.{i}.attn_q.weight"
        tensor_map[f"layers.{i}.attention.wk.weight"] = f"blk.{i}.attn_k.weight"
        tensor_map[f"layers.{i}.attention.wv.weight"] = f"blk.{i}.attn_v.weight"
        tensor_map[f"layers.{i}.attention.wo.weight"] = f"blk.{i}.attn_output.weight"
        tensor_map[f"layers.{i}.feed_forward.w1.weight"] = f"blk.{i}.ffn_gate.weight"
        tensor_map[f"layers.{i}.feed_forward.w2.weight"] = f"blk.{i}.ffn_down.weight"
        tensor_map[f"layers.{i}.feed_forward.w3.weight"] = f"blk.{i}.ffn_up.weight"
        tensor_map[f"layers.{i}.attention_norm.weight"] = f"blk.{i}.attn_norm.weight"
        tensor_map[f"layers.{i}.ffn_norm.weight"] = f"blk.{i}.ffn_norm.weight"

    print(f"[*] Packing {len(tensor_map)} tensors into GGUF container...")
    for orig_name, ggml_name in tensor_map.items():
        if orig_name in state:
            t = state[orig_name].float().numpy()
            if dtype == "f16":
                t = t.astype(np.float16)
            writer.add_tensor(ggml_name, t)

    # If tied embeddings, also write output.weight
    if "output.weight" not in state and "tok_embeddings.weight" in state:
        t = state["tok_embeddings.weight"].float().numpy()
        if dtype == "f16":
            t = t.astype(np.float16)
        writer.add_tensor("output.weight", t)

    print("[*] Writing GGUF binary file to disk...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_gguf_path) / (1024 ** 2)
    print(f"[OK] GGUF Model exported successfully: {output_gguf_path} ({size_mb:.2f} MB)")

    # 4. Generate Ollama Modelfile
    modelfile_path = os.path.join(os.path.dirname(output_gguf_path), "Modelfile")
    modelfile_content = f"""# Modelfile for TunisiaLLM (Tunisian Derja 80M LLM)
FROM {os.path.basename(output_gguf_path)}

TEMPLATE \"\"\"<bos><|system|>
{{{{ .System }}}}<|user|>
{{{{ .Prompt }}}}<|assistant|>
{{{{ .Response }}}}<eos>\"\"\"

SYSTEM \"\"\"إنت مساعد ذكي ولطيف بالدارجة التونسية، تعاون بوضوح وطلاقة وتجاوب ديما على قد السؤال.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<eos>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|system|>"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"[OK] Generated Ollama Modelfile: {modelfile_path}")
    print("\nTo create and run with Ollama:")
    print(f"  cd {os.path.dirname(output_gguf_path)}")
    print("  ollama create tunisiallm -f Modelfile")
    print("  ollama run tunisiallm\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TunisiaLLM GGUF Exporter")
    parser.add_argument("--checkpoint", type=str, default="sft_checkpoint_best.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--output", type=str, default="output/tunisiallm-80m-f16.gguf")
    parser.add_argument("--dtype", type=str, default="f16", choices=["f16", "f32"])
    args = parser.parse_args()

    export_to_gguf(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_gguf_path=args.output,
        dtype=args.dtype
    )

