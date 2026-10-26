"""
02_tokenizer.py - Byte-Level BPE Tokenizer Trainer & Binary Sharder
Trains a Byte-Level BPE tokenizer on cleaned text using HuggingFace `tokenizers`.
Supports special tokens for chat roles, tool calling, and encodes text into uint16 .bin shards.
"""

import os
import sys
import argparse
from typing import List, Dict, Optional

import numpy as np
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders, processors

from utils import load_config, set_seed, format_bytes


def get_special_tokens_list(cfg: Dict) -> List[str]:
    """Extract ordered special tokens from config."""
    spec_dict = cfg["tokenizer"]["special_tokens"]
    return [
        spec_dict["pad"],
        spec_dict["bos"],
        spec_dict["eos"],
        spec_dict["user"],
        spec_dict["assistant"],
        spec_dict["system"],
        spec_dict["tool_call_start"],
        spec_dict["tool_call_end"],
        spec_dict["tool_result_start"],
        spec_dict["tool_result_end"],
    ]


def train_byte_level_bpe(
    train_files: List[str],
    output_tokenizer_path: str,
    vocab_size: int = 16384,
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None
) -> Tokenizer:
    """
    Train a Byte-Level BPE tokenizer.
    Byte-level BPE never creates <unk> tokens because any byte sequence can be represented.
    """
    print(f"\n[*] Initializing Byte-Level BPE Tokenizer...")
    print(f"    - Target Vocab Size: {vocab_size}")
    print(f"    - Min Frequency:     {min_frequency}")
    print(f"    - Special Tokens:    {special_tokens}")
    print(f"    - Training Files:    {train_files}")

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # ByteLevel pre-tokenizer: splits on whitespace & handles byte fallbacks
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens or [],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    print("\n[*] Training tokenizer on cleaned corpus...")
    tokenizer.train(files=train_files, trainer=trainer)

    # Save tokenizer to JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_tokenizer_path)), exist_ok=True)
    tokenizer.save(output_tokenizer_path)
    print(f"[OK] Tokenizer successfully trained and saved to: {output_tokenizer_path}")
    print(f"     Actual Vocab Size: {tokenizer.get_vocab_size()}")

    return tokenizer


def verify_tokenizer(tokenizer: Tokenizer, cfg: Dict) -> None:
    """Run round-trip tests on Derja, Arabizi, French/English code-switching, and tool calls."""
    print("\n" + "=" * 70)
    print("Verifying Tokenizer Round-Trip & Special Token Preservation")
    print("=" * 70)

    test_sentences = [
        "عسلامة يا باهي! شنوة حوالك اليوم؟ إن شاء الله لاباس وكل شي مريڤل.",
        "ahla bik ya sahbi, cava elyoum? nchallah kol chay labes.",
        "Bonjour خويا, c'est bon kamalt el projet mte3i hier soir.",
        "<|user|>\nقداش 12 * 15؟<|assistant|>\n<tool_call>calc: 12 * 15</tool_call><tool_result>180</tool_result>النتيجة هي 180.<eos>",
        "الكسكسي التونسي بالقرنيط والحوت بنين برشا وما كيفو شي في البحر الأبيض المتوسط."
    ]

    for sent in test_sentences:
        encoding = tokenizer.encode(sent)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        print(f"\nOriginal: {sent}")
        print(f"Tokens:   {len(encoding.ids)} tokens | IDs: {encoding.ids[:12]}...")
        print(f"Decoded:  {decoded}")
        # Verify lossless round-trip (preserving special tokens)
        if decoded.strip() != sent.strip():
            print(f"[!] Warning: Minor mismatch in roundtrip: {decoded}")
        else:
            print("Round-trip status: EXACT MATCH [PASSED]")

    # Verify special token IDs
    spec_tokens = get_special_tokens_list(cfg)
    print("\nSpecial Token ID Mapping:")
    for token in spec_tokens:
        tok_id = tokenizer.token_to_id(token)
        print(f"  {token:<20} -> ID: {tok_id}")
        assert tok_id is not None, f"Special token {token} was not found in vocabulary!"


def encode_corpus_to_shards(
    text_path: str,
    output_prefix: str,
    tokenizer: Tokenizer,
    tokens_per_shard: int = 5000000,
    eos_token: str = "<eos>"
) -> List[str]:
    """Encode lines from text_path into uint16 .bin shards."""
    if not os.path.isfile(text_path):
        print(f"[!] Corpus file {text_path} does not exist.")
        return []

    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        eos_id = 2

    print(f"\n[*] Encoding {text_path} into uint16 .bin shards...")
    shard_files = []
    shard_idx = 0
    buffer: List[int] = []
    total_tokens = 0

    def write_shard(token_list: List[int], s_idx: int) -> str:
        out_name = f"{output_prefix}_shard_{s_idx:03d}.bin"
        arr = np.array(token_list, dtype=np.uint16)
        arr.tofile(out_name)
        print(f"    [+] Saved shard {s_idx:03d}: {out_name} ({len(token_list):,} tokens, {format_bytes(arr.nbytes)})")
        return out_name

    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            enc = tokenizer.encode(line).ids
            buffer.extend(enc)
            buffer.append(eos_id)
            total_tokens += len(enc) + 1

            if len(buffer) >= tokens_per_shard:
                to_save = buffer[:tokens_per_shard]
                shard_files.append(write_shard(to_save, shard_idx))
                buffer = buffer[tokens_per_shard:]
                shard_idx += 1

    if buffer:
        shard_files.append(write_shard(buffer, shard_idx))

    print(f"[OK] Total tokens encoded: {total_tokens:,} into {len(shard_files)} shards.")
    return shard_files


def run_tokenizer_pipeline(cfg_path: Optional[str] = None, smoke_test: Optional[bool] = None) -> None:
    """Run full tokenizer training and sharding."""
    cfg = load_config(cfg_path, smoke_test_override=smoke_test)
    set_seed(cfg.get("seed", 42))

    data_dir = cfg["paths"]["data_dir"]
    train_txt = os.path.join(data_dir, "cleaned_train.txt")
    val_txt = os.path.join(data_dir, "cleaned_val.txt")
    tokenizer_path = cfg["paths"]["tokenizer_path"]

    # If cleaned_train.txt doesn't exist, create a sample so pipeline is runnable standalone
    if not os.path.isfile(train_txt):
        print(f"[!] {train_txt} not found. Running data prep first or creating starter corpus...")
        import importlib
        data_prep_mod = importlib.import_module("01_data_prep")
        data_prep_mod.run_data_prep(cfg_path, smoke_test=cfg.get("smoke_test", False))

    special_tokens = get_special_tokens_list(cfg)
    vocab_size = cfg["tokenizer"]["vocab_size"]
    min_freq = cfg["tokenizer"]["min_frequency"]

    # 1. Train Tokenizer
    tokenizer = train_byte_level_bpe(
        train_files=[train_txt],
        output_tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens
    )

    # 2. Verify
    verify_tokenizer(tokenizer, cfg)

    # 3. Shard cleaned text to uint16 .bin shards
    tokens_per_shard = cfg["data"]["tokens_per_shard"]
    if cfg.get("smoke_test"):
        tokens_per_shard = 200000  # Smaller shards for quick verification

    train_prefix = os.path.join(data_dir, "train")
    val_prefix = os.path.join(data_dir, "val")

    encode_corpus_to_shards(train_txt, train_prefix, tokenizer, tokens_per_shard=tokens_per_shard)
    encode_corpus_to_shards(val_txt, val_prefix, tokenizer, tokens_per_shard=tokens_per_shard)

    print("\n[OK] Tokenizer training & dataset sharding complete! Ready for pretraining.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TunisiaLLM Tokenizer Trainer")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny smoke test")
    args = parser.parse_args()

    run_tokenizer_pipeline(args.config, smoke_test=args.smoke_test if args.smoke_test else None)
