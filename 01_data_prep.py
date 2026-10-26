"""
01_data_prep.py - Data Collection, Cleaning, Deduplication, and Sharding Pipeline
Downloads Tunisian Derja corpora and general Wikipedia mix, applies normalization,
exact and MinHash near-deduplication, language/quality filtering, and saves train/val sets.
"""

import os
import sys
import re
import math
import json
import hashlib
import argparse
import unicodedata
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Iterator, Optional

import numpy as np

# Try importing HuggingFace datasets, provide fallback if offline
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from utils import load_config, set_seed, format_bytes, format_num_params


# ==============================================================================
# Arabic & Derja Normalization & Cleaning Helpers
# ==============================================================================

# Arabic diacritics (tashkeel) regex
ARABIC_DIACRITICS_REGEX = re.compile(r'[\u064B-\u0652\u0657-\u065F\u0670]')
# Arabic tatweel (kashida)
ARABIC_TATWEEL_REGEX = re.compile(r'\u0640')
# URLs and emails
URL_REGEX = re.compile(r'https?://\S+|www\.\S+|ftp://\S+')
EMAIL_REGEX = re.compile(r'\S+@\S+\.\S+')
# HTML tags
HTML_TAG_REGEX = re.compile(r'<[^>]+>')
# Repeated characters (e.g. ههههههه -> ههه, looooool -> lool)
REPEATED_CHARS_REGEX = re.compile(r'(.)\1{3,}')
# Emojis and miscellaneous symbols (Unicode blocks)
EMOJI_REGEX = re.compile(
    r'[\U00010000-\U0010ffff]'
    r'|[\u2600-\u27BF]'
    r'|[\uD83C-\uDBFF\uDC00-\uDFFF]'
)
# Zero-width characters & non-printables
CONTROL_CHARS_REGEX = re.compile(r'[\u200B-\u200D\uFEFF\u0000-\u0008\u000B\u000C\u000E-\u001F]')


def normalize_text(text: str) -> str:
    """Normalize Unicode, strip HTML, URLs, emojis, tatweel, diacritics, and collapse repeats."""
    if not text or not isinstance(text, str):
        return ""

    # NFKC Unicode Normalization
    text = unicodedata.normalize("NFKC", text)

    # Strip HTML and URLs
    text = HTML_TAG_REGEX.sub(" ", text)
    text = URL_REGEX.sub(" ", text)
    text = EMAIL_REGEX.sub(" ", text)

    # Remove zero-width & control characters
    text = CONTROL_CHARS_REGEX.sub("", text)

    # Remove Arabic tatweel (ـ)
    text = ARABIC_TATWEEL_REGEX.sub("", text)

    # Remove Arabic diacritics / tashkeel for casual Derja consistency
    text = ARABIC_DIACRITICS_REGEX.sub("", text)

    # Collapse character repetitions (e.g., "هههههههههه" -> "ههه")
    text = REPEATED_CHARS_REGEX.sub(r"\1\1", text)

    # Remove excessive emojis & symbols
    text = EMOJI_REGEX.sub(" ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def quality_filter(text: str, min_chars: int = 15, max_chars: int = 15000, min_words: int = 3) -> bool:
    """Filter out documents that are too short, too long, or mostly non-linguistic noise."""
    length = len(text)
    if length < min_chars or length > max_chars:
        return False

    words = text.split()
    if len(words) < min_words:
        return False

    # Check alphanumeric script ratio (reject binary garbage or symbol soup)
    alpha_chars = sum(1 for c in text if c.isalnum())
    if alpha_chars / max(1, length) < 0.40:
        return False

    return True


# ==============================================================================
# Deduplication: Exact Hash + MinHash Near-Deduplication
# ==============================================================================

class MinHashDeduplicator:
    """
    Self-contained pure-Python/NumPy MinHash LSH deduplicator.
    Eliminates exact and near-duplicate documents.
    """
    def __init__(self, num_perm: int = 64, threshold: float = 0.85, ngram_size: int = 5, use_minhash: bool = False, seed: int = 42):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.use_minhash = use_minhash
        self.seen_exact: Set[int] = set()

        if self.use_minhash:
            rng = np.random.RandomState(seed)
            # Random linear hash parameters: (a * x + b) % p
            self.large_prime = 2147483647  # 2^31 - 1
            self.a = rng.randint(1, self.large_prime - 1, size=num_perm, dtype=np.int64)
            self.b = rng.randint(0, self.large_prime - 1, size=num_perm, dtype=np.int64)

            # LSH index: bucket bands
            self.num_bands = 8
            self.rows_per_band = num_perm // self.num_bands
            self.buckets = [defaultdict(list) for _ in range(self.num_bands)]
            self.minhashes: List[np.ndarray] = []

    def _get_shingles(self, text: str) -> Set[int]:
        """Extract word shingles (capped to 3000 chars) and hash to 32-bit integers."""
        shingles = set()
        words = text.lower()[:3000].split()
        if len(words) < 3:
            shingles.add(abs(hash(" ".join(words))) % self.large_prime)
            return shingles
        for i in range(len(words) - 2):
            shingle = f"{words[i]} {words[i+1]} {words[i+2]}"
            shingles.add(abs(hash(shingle)) % self.large_prime)
        return shingles

    def _compute_minhash(self, shingles: Set[int]) -> np.ndarray:
        """Compute the MinHash signature vector for a set of shingles."""
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.int64)
        shingle_arr = np.array(list(shingles), dtype=np.int64).reshape(-1, 1)
        # Vectorized min-hash computation
        hashes = (shingle_arr * self.a + self.b) % self.large_prime
        return np.min(hashes, axis=0)

    def is_duplicate(self, text: str) -> bool:
        """Return True if document is an exact duplicate or near-duplicate."""
        # 1. Exact hash check (lightning fast MD5)
        exact_h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:16], 16)
        if exact_h in self.seen_exact:
            return True

        if not self.use_minhash:
            self.seen_exact.add(exact_h)
            return False

        # 2. MinHash signature
        shingles = self._get_shingles(text)
        if not shingles:
            return True
        sig = self._compute_minhash(shingles)

        # 3. LSH Query
        matched = False
        candidates = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_key = tuple(sig[start:end])
            for prev_idx in self.buckets[band_idx].get(band_key, []):
                candidates.add(prev_idx)

        # Check Jaccard similarity against candidate matches
        for cand_idx in candidates:
            cand_sig = self.minhashes[cand_idx]
            sim = np.mean(sig == cand_sig)
            if sim >= self.threshold:
                matched = True
                break

        if matched:
            return True

        # Register document
        doc_idx = len(self.minhashes)
        self.seen_exact.add(exact_h)
        self.minhashes.append(sig)
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_key = tuple(sig[start:end])
            self.buckets[band_idx][band_key].append(doc_idx)

        return False


# ==============================================================================
# Synthetic Fallback Corpus for Offline / Smoke Test Execution
# ==============================================================================

SYNTHETIC_DERJA_FALLBACK = [
    # Common conversational Derja
    "عسلامة يا جماعة، شنوة أحوالكم اليوم؟ إن شاء الله تكونوا لاباس وبخير ونعمة.",
    "تونس بلاد مزيانة برشا، فيها بحر وبساتين وصحراء وتاريخ كبير من عهد قرطاج.",
    "الكسكسي بالعصبان والمقرونة الجارية واللبلابي في الشتاء سخون يدفّي البدن.",
    "بربي وقتاش يحل البوسطة غدوة؟ نحب نخلّص فاتورة الستاغ ونبعث حوالة.",
    "ماتش الدربي التونسي بين الترجي والإفريقي ديما فيه جو وحماس كبير في رادس.",
    "سامحني يا خويا، تنجم تدلني على أقرب محطة مترو وإلا لواجات؟",
    "شنوة رايك في البرمجة بالبايثون والذكاء الاصطناعي؟ توا المستقبل ماشي غادي.",
    "البحر في الحمامات وسوسة وجربة في الصيف عالمي، السياح يجيو من كل بلاصة.",
    "عيشك ويرحم والديك، ما قصرتش معايا وعاونتني برشا.",
    "الطقس اليوم في تونس العاصمة سخون برشا والشهيلي يهب من الصباح.",
    "توا وقتاش يبدا الماتش متع المنتخب؟ إن شاء الله نربحو ونتعداو للدور الجاي.",
    "خدمة المرمة والعمارة في الصيف صعيبة والرجال يخدمو تحت الشمس ربي يعينهم.",
    "فطرت الصباح كسكروت كفتاجي وباتيسري وكاس تاي بنين يعدل الكيف.",
    "شكون يعرف حماص باهي يبيع كاوكاو ولوز سخون في لافايات؟",
    "السيستام طايح اليوم في الإدارة، قالولي ارجع غدوة الصباح بكري.",
    "كي تبدا ماشي لصفاقس ما تنساش تاكل الحوت المشوي والصيد البنين.",
    "قريت كتاب باهي برشا على تاريخ شمال إفريقيا وكيفاش كانت تونس مركز للتجارة.",
    "بربي قداش سوم التيكي متع الطيارة لفرنسا في الصيف؟ غالي برشا توا.",
    "يا باهي، ديما خليك متفائل وما تيأسش، ربي يحلها في وجه كل إنسان صابر.",
    "القهوة العربي بالزهر وكعيبات المقروض القيرواني ما كيفهم شي في الدنيا."
]

SYNTHETIC_WIKI_FALLBACK = [
    "تونس أو رسمياً الجمهورية التونسية، هي دولة عربية تقع في شمال أفريقيا. عاصمتها مدينة تونس، وتبلغ مساحتها حوالي 163,610 كيلومتر مربع.",
    "تأسست مدينة قرطاج عام 814 قبل الميلاد على يد الأميرة الفينيقية أليسا، وأصبحت قوة بحرية وتجارية كبرى في البحر الأبيض المتوسط.",
    "La Tunisie est un pays d'Afrique du Nord bordé au nord et à l'est par la mer Méditerranée. Sa capitale est Tunis.",
    "L'histoire de la Tunisie est marquée par la civilisation carthaginoise, l'Empire romain, et l'ère arabo-musulmane qui a façonné sa culture.",
    "Tunisia is the northernmost country in Africa, situated along the Mediterranean coast with a rich cultural heritage spanning millennia.",
    "Computer science and artificial intelligence are rapidly transforming software development, education, and economic growth globally."
]


# ==============================================================================
# Dataset Download and Processing
# ==============================================================================

def download_tunisian_datasets(cfg: Dict, force_offline: bool = False) -> Dict[str, List[str]]:
    """Download Tunisian Derja corpora from HuggingFace, with offline fallback."""
    corpus_texts: Dict[str, List[str]] = defaultdict(list)
    smoke_test = cfg.get("smoke_test", False)

    for item in cfg["data"]["tunisian_datasets"]:
        name = item["name"]
        split = item.get("split", "train")
        col = item.get("text_column", "text")

        print(f"\n[*] Fetching Tunisian dataset: {name} (split: {split})...")
        downloaded = False
        if HF_DATASETS_AVAILABLE and not force_offline:
            try:
                split_spec = f"{split}[:1000]" if smoke_test else split
                try:
                    ds = load_dataset(name, split=split_spec)
                except TypeError:
                    ds = load_dataset(name, split=split_spec, trust_remote_code=True)

                for row in ds:
                    val = row.get(col) or row.get("text") or row.get("content") or row.get("derja")
                    if val and isinstance(val, str):
                        corpus_texts[name].append(val)
                print(f"    [+] Successfully loaded {len(corpus_texts[name])} rows from {name}")
                downloaded = True
            except Exception as e:
                print(f"    [!] HF download failed or timed out ({e}). Using offline synthetic fallback.")

        if not downloaded:
            # Fallback to high-quality synthetic expansion
            mult = 50 if smoke_test else 1000
            corpus_texts[name] = [
                f"{line} ({name.split('/')[-1]} sample {idx})"
                for idx, line in enumerate(SYNTHETIC_DERJA_FALLBACK * mult)
            ]
            print(f"    [+] Generated {len(corpus_texts[name])} offline fallback samples for {name}")

    return corpus_texts


def shard_text_to_bin(
    text_file: str,
    output_prefix: str,
    tokenizer_path: str,
    tokens_per_shard: int = 5000000,
    eos_token: str = "<eos>"
) -> List[str]:
    """
    Encode raw text file into uint16 numpy binary shards.
    uint16 supports vocabularies up to 65,535 tokens (our vocab is 16k-24k).
    """
    from tokenizers import Tokenizer

    if not os.path.isfile(tokenizer_path):
        print(f"[!] Tokenizer {tokenizer_path} not found. Cannot create .bin shards yet.")
        return []

    print(f"\n[*] Sharding {text_file} into uint16 binary files with {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        eos_id = 2  # Standard fallback

    shard_files = []
    shard_idx = 0
    buffer = []
    total_tokens = 0

    def flush_shard(tokens: List[int], s_idx: int) -> str:
        s_name = f"{output_prefix}_shard_{s_idx:03d}.bin"
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(s_name)
        print(f"    [+] Saved shard {s_idx:03d}: {s_name} ({len(tokens):,} tokens, {format_bytes(arr.nbytes)})")
        return s_name

    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            encoded = tokenizer.encode(line).ids
            buffer.extend(encoded)
            buffer.append(eos_id)
            total_tokens += len(encoded) + 1

            if len(buffer) >= tokens_per_shard:
                shard_to_save = buffer[:tokens_per_shard]
                shard_files.append(flush_shard(shard_to_save, shard_idx))
                buffer = buffer[tokens_per_shard:]
                shard_idx += 1

    if buffer:
        shard_files.append(flush_shard(buffer, shard_idx))

    print(f"[OK] Sharded {total_tokens:,} tokens into {len(shard_files)} binary files.")
    return shard_files


def download_wikipedia_mix(cfg: Dict, force_offline: bool = False) -> Dict[str, List[str]]:
    """Download general Arabic, French, and English Wikipedia samples for multilingual coherence."""
    corpus_texts: Dict[str, List[str]] = defaultdict(list)
    wiki_cfg = cfg["data"]["wikipedia"]
    if not wiki_cfg.get("enabled", True):
        return corpus_texts

    smoke_test = cfg.get("smoke_test", False)
    target_samples = 500 if smoke_test else wiki_cfg.get("samples_per_lang", 15000)

    for lang in wiki_cfg.get("languages", ["ar", "fr", "en"]):
        name = f"wikipedia_{lang}"
        print(f"\n[*] Fetching Wikipedia mix for [{lang}] (target: {target_samples} samples)...")
        downloaded = False

        if HF_DATASETS_AVAILABLE and not force_offline:
            try:
                # Try wikimedia/wikipedia or wikipedia dataset with streaming
                ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
                count = 0
                for row in ds:
                    text = row.get("text", "")
                    if len(text) > 200:
                        paras = text.split("\n\n")[:3]
                        corpus_texts[name].append("\n\n".join(paras))
                        count += 1
                        if count >= target_samples:
                            break
                print(f"    [+] Streamed {count} Wikipedia samples for {lang}")
                downloaded = True
            except Exception as e:
                print(f"    [!] Wikipedia streaming failed ({e}). Using offline multilingual fallback.")

        if not downloaded:
            mult = 20 if smoke_test else 200
            corpus_texts[name] = [
                f"{line} [Lang: {lang} sample {idx}]"
                for idx, line in enumerate(SYNTHETIC_WIKI_FALLBACK * mult)
            ]
            print(f"    [+] Generated {len(corpus_texts[name])} offline fallback samples for {name}")

    return corpus_texts


# ==============================================================================
# Pipeline Orchestrator
# ==============================================================================

def run_data_prep(cfg_path: Optional[str] = None, smoke_test: Optional[bool] = None, force_offline: bool = False) -> None:
    """Run full data preparation pipeline."""
    cfg = load_config(cfg_path, smoke_test_override=smoke_test)
    set_seed(cfg.get("seed", 42))

    data_dir = cfg["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 80)
    print(f"TunisiaLLM Data Preparation Pipeline")
    print(f"Mode: {'SMOKE TEST' if cfg.get('smoke_test') else 'FULL RUN'} | Offline: {force_offline}")
    print(f"Target Output Directory: {data_dir}")
    print("=" * 80)

    # 1. Fetch raw datasets
    tunisian_corpora = download_tunisian_datasets(cfg, force_offline=force_offline)
    wiki_corpora = download_wikipedia_mix(cfg, force_offline=force_offline)

    all_corpora = {**tunisian_corpora, **wiki_corpora}

    # 2. Clean & Deduplicate
    use_minhash = cfg["data"]["dedup"].get("minhash", False)
    deduplicator = MinHashDeduplicator(
        num_perm=cfg["data"]["dedup"].get("minhash_num_perm", 64),
        threshold=cfg["data"]["dedup"].get("minhash_threshold", 0.85),
        ngram_size=cfg["data"]["dedup"].get("ngram_size", 5),
        use_minhash=use_minhash,
        seed=cfg.get("seed", 42)
    )

    cleaned_by_source: Dict[str, List[str]] = defaultdict(list)
    stats: Dict[str, Dict[str, int]] = {}

    total_tokens_est = 0
    tunisian_tokens_est = 0

    mode_str = "Exact MD5 + MinHash LSH" if use_minhash else "Exact MD5 (Fast)"
    print("\n" + "=" * 80)
    print(f"Cleaning, Normalizing, and Deduplicating ({mode_str})...")
    print("=" * 80)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    for source_name, raw_docs in all_corpora.items():
        is_tunisian = source_name in tunisian_corpora
        initial_count = len(raw_docs)
        cleaned_docs = []
        source_tokens = 0

        doc_iter = tqdm(raw_docs, desc=f"  [*] {source_name[:35]}", total=len(raw_docs)) if has_tqdm else raw_docs
        for doc in doc_iter:
            norm = normalize_text(doc)
            if not quality_filter(
                norm,
                min_chars=cfg["data"]["min_char_len"],
                max_chars=cfg["data"]["max_char_len"],
                min_words=cfg["data"]["min_words"]
            ):
                continue

            # Deduplication
            if cfg["data"]["dedup"].get("exact", True) or use_minhash:
                if deduplicator.is_duplicate(norm):
                    continue

            cleaned_docs.append(norm)
            est_tokens = int(len(norm.split()) * 1.3)
            source_tokens += est_tokens

        cleaned_by_source[source_name] = cleaned_docs
        total_tokens_est += source_tokens
        if is_tunisian:
            tunisian_tokens_est += source_tokens

        stats[source_name] = {
            "raw_docs": initial_count,
            "clean_docs": len(cleaned_docs),
            "est_tokens": source_tokens
        }

    # 3. Print Token and Size Statistics
    print("\n" + "-" * 80)
    print(f"{'Source Name':<45} | {'Raw Docs':<10} | {'Clean Docs':<10} | {'Est. Tokens':<12}")
    print("-" * 80)
    for src, st in stats.items():
        print(f"{src:<45} | {st['raw_docs']:<10} | {st['clean_docs']:<10} | {format_num_params(st['est_tokens']):<12}")
    print("-" * 80)
    tunisian_pct = (tunisian_tokens_est / max(1, total_tokens_est)) * 100
    print(f"Total Cleaned Documents: {sum(st['clean_docs'] for st in stats.values())}")
    print(f"Total Estimated Tokens:  {format_num_params(total_tokens_est)}")
    print(f"Tunisian Derja Ratio:    {tunisian_pct:.1f}% (Target: ~{cfg['data']['target_tunisian_ratio']*100:.0f}%)")
    print("-" * 80)

    # 4. Partition into Train (99%) and Held-out Val (1%)
    val_ratio = cfg["data"]["val_ratio"]
    train_docs: List[str] = []
    val_docs: List[str] = []

    rng = np.random.RandomState(cfg.get("seed", 42))

    for src, docs in cleaned_by_source.items():
        shuffled = list(docs)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_docs.extend(shuffled[:n_val])
        train_docs.extend(shuffled[n_val:])

    rng.shuffle(train_docs)
    rng.shuffle(val_docs)

    print(f"\nFinal Split: {len(train_docs)} Train documents, {len(val_docs)} Validation documents.")

    # 5. Save Cleaned Corpus Text Files
    train_txt_path = os.path.join(data_dir, "cleaned_train.txt")
    val_txt_path = os.path.join(data_dir, "cleaned_val.txt")

    print(f"Saving training text to: {train_txt_path}")
    with open(train_txt_path, "w", encoding="utf-8") as f:
        for doc in train_docs:
            f.write(doc + "\n\n")

    print(f"Saving validation text to: {val_txt_path}")
    with open(val_txt_path, "w", encoding="utf-8") as f:
        for doc in val_docs:
            f.write(doc + "\n\n")

    # Save summary metadata
    meta_path = os.path.join(data_dir, "data_prep_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats": stats,
            "total_tokens_est": total_tokens_est,
            "tunisian_tokens_est": tunisian_tokens_est,
            "tunisian_ratio": tunisian_pct,
            "train_docs_count": len(train_docs),
            "val_docs_count": len(val_docs)
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Data preparation completed successfully. Metadata saved to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TunisiaLLM Data Preparation")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny smoke test")
    parser.add_argument("--offline", action="store_true", help="Use local synthetic corpus without network access")
    args = parser.parse_args()

    run_data_prep(args.config, smoke_test=args.smoke_test if args.smoke_test else None, force_offline=args.offline)
