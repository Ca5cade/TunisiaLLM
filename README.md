# 🇹🇳 TunisiaLLM: From-Scratch Tunisian Derja LLM Pipeline

A complete, production-grade pipeline to pretrain and instruction-tune an **~80M parameter** decoder-only chat LLM for **Tunisian Arabic (Derja)** from absolute scratch.

- **Zero Pretrained Weights**: Trained strictly from random initialization ($\mathcal{N}(0, 0.02)$).
- **Pure PyTorch**: No Hugging Face `Trainer` or external training frameworks.
- **Optimized for Kaggle**: Runs within free Kaggle limits (2x T4 16GB GPUs, 12h session limit, ~30 GPU hours/week, `/kaggle/working`).
- **Low Latency**: Lightweight LLaMA-style decoder with RoPE, RMSNorm, SwiGLU, Grouped-Query Attention (GQA), Tied Embeddings, and native PyTorch SDPA (no flash-attn compilation issues on T4).
- **Tool Calling**: Emits and processes `<tool_call>calc: ...</tool_call>` (safe AST-based calculator) and `<tool_call>search: ...</tool_call>` (pluggable search).

---

## 📁 Repository Structure

```
TunisiaLLM/
├── config.yaml          # Master configuration (paths, model presets, training params)
├── utils.py             # Shared environment detection, seed setting, and formatting
├── 01_data_prep.py      # Download, clean, MinHash near-dedup, and train/val splits
├── 02_tokenizer.py      # Train 16k Byte-Level BPE + uint16 .bin sharder
├── 03_model.py          # LLaMA decoder (30M, 80M, 125M presets, RoPE, SwiGLU, GQA)
├── 04_pretrain.py       # DDP 2x T4 pretraining, memmap dataloader, 20m checkpointing
├── 05_make_sft_data.py  # SFT dialogue builder (tools, smalltalk, QA, loss masking)
├── 06_sft.py            # Supervised fine-tuning with assistant-only loss masking
├── 07_inference.py      # Streaming generation with KV cache, tools, INT8/FP16 export
└── README.md            # Detailed workflow and Kaggle session chaining guide
```

---

## 🏗️ Model Architecture (80M Preset)

| Component | Specification | Rationale |
| :--- | :--- | :--- |
| **Hidden Dimension ($d_{\text{model}}$)** | 672 | Low latency, compact representation |
| **Layers ($n_{\text{layers}}$)** | 12 | Deep enough for grammar & dialect structure |
| **Query Heads ($n_{\text{heads}}$)** | 12 (head dim = 56) | Expressive multi-head attention |
| **KV Heads ($n_{\text{kv\_heads}}$)** | 4 (Grouped-Query Attn) | 3x memory reduction during KV-cached generation |
| **Feed-Forward ($d_{\text{hidden}}$)** | 2176 (SwiGLU) | $\sim 3.2 \times d_{\text{model}}$ for non-linear reasoning |
| **Position Embeddings** | RoPE ($\theta = 10000$) | Relative positional representation up to 1024 tokens |
| **Layer Normalization** | RMSNorm ($\epsilon = 10^{-5}$) | Faster than LayerNorm, no mean centering/bias |
| **Weight Tying** | `lm_head == tok_embeddings` | Saves **11.01M parameters** (16,384 $\times$ 672) |
| **Total Trainable Params** | **78.12M parameters** | Fits effortlessly in fast GPU VRAM |
| **Attention Kernel** | PyTorch Native SDPA | Native Tensor Core speedup on NVIDIA T4 without compilation |

---

## ⚡ Quick Smoke Test (< 2 Minutes)

Verify the entire pipeline locally or on a new Kaggle session before launching long pretraining:

```bash
# 1. Clean & prepare sample dataset
python 01_data_prep.py --smoke-test --offline

# 2. Train tokenizer and create binary shards
python 02_tokenizer.py --smoke-test

# 3. Verify model architecture presets
python 03_model.py

# 4. Pretrain 5 steps (verifies loss, DDP/single device, and checkpointing)
python 04_pretrain.py --smoke-test

# 5. Generate 1,000 SFT dialogues with loss masking
python 05_make_sft_data.py --smoke-test

# 6. SFT fine-tune 5 steps from checkpoint
python 06_sft.py --smoke-test

# 7. Test streaming inference & model quantization
python 07_inference.py --prompt "عسلامة يا باهي"
python 07_inference.py --export-fp16 output/model_fp16.pt --export-int8 output/model_int8.pt
```

---

## 📓 Kaggle Execution Guide (Chaining Sessions)

Kaggle provides **2x T4 GPUs (16GB each)** with a **12-hour session limit** and **~30 GPU hours per week**. Because pretraining 80M tokens to convergence requires ~15-20 hours on 2x T4, the workflow is split into **chainable sessions**.

### Prerequisites in Kaggle Notebook Settings:
- **Accelerator**: `GPU T4 x2`
- **Internet**: `On` (needed for 01_data_prep to download HF datasets)
- **Environment**: `Python 3` (PyTorch 2.x comes preinstalled)

---

### Phase 1: Data Preparation & Tokenizer (CPU or P100/T4)
*Runtime: ~20 minutes | Output: `data/`, `tokenizer.json`*

Run in **Notebook 1**:
```bash
# Clone or upload repository scripts to /kaggle/working
cd /kaggle/working

# 1. Download Tunisian corpora + Arabic/French/English Wikipedia subsets
python 01_data_prep.py

# 2. Train 16k Byte-Level BPE and shard into uint16 .bin files
python 02_tokenizer.py
```

> **Save Output as Kaggle Dataset:**
> 1. In Kaggle notebook, click **Save Version** -> **Quick Save**.
> 2. Open the completed version, scroll to Output, and click **Create Dataset** (name it `tunisiallm-data`).

---

### Phase 2: Pretraining Part 1 (2x T4 GPUs)
*Runtime: ~11 hours (automatic graceful shutdown)*

In **Notebook 2**:
1. Click **+ Add Input** and attach `tunisiallm-data` (it mounts at `/kaggle/input/tunisiallm-data`).
2. Run distributed pretraining with `torchrun`:

```bash
cd /kaggle/working

# Symlink or copy data from input dataset to working directory
mkdir -p data checkpoints
cp /kaggle/input/tunisiallm-data/tokenizer.json ./tokenizer.json
cp -r /kaggle/input/tunisiallm-data/data/* ./data/

# Launch DDP over both T4 GPUs:
torchrun --nproc_per_node=2 04_pretrain.py
```

#### What happens during pretraining:
- **Zero RAM Overhead**: Reads tokens directly from disk via `np.memmap(dtype=np.uint16)`.
- **Checkpoint Every 20 Mins**: Saves `checkpoint_latest.pt`, `checkpoint_best.pt`, and rolling snapshots.
- **Graceful Kaggle Exit**: At **11.0 hours** of execution, the script automatically saves `checkpoint_session_end.pt` and exits with code 0 before Kaggle forcibly kills the container!

> **Save Checkpoint Dataset:**
> Save the notebook version and create a Kaggle dataset named `tunisiallm-ckpt-p1` from `/kaggle/working/checkpoints`.

---

### Phase 3: Pretraining Part 2 (Auto-Resume Chaining)
*Runtime: ~8-10 hours*

In **Notebook 3**:
1. Click **+ Add Input** -> Attach `tunisiallm-data` AND `tunisiallm-ckpt-p1`.
2. Setup and resume:

```bash
cd /kaggle/working
mkdir -p data checkpoints
cp /kaggle/input/tunisiallm-data/tokenizer.json ./tokenizer.json
cp -r /kaggle/input/tunisiallm-data/data/* ./data/

# Copy checkpoint from previous session
cp -r /kaggle/input/tunisiallm-ckpt-p1/checkpoints/* ./checkpoints/

# Launch training: 04_pretrain.py automatically detects checkpoint_latest.pt,
# restores weights, optimizer, scaler, RNG, and exact data offset, and continues!
torchrun --nproc_per_node=2 04_pretrain.py
```

---

### Phase 4: SFT Instruction Tuning & Tool Alignment
*Runtime: ~2.5 hours on 2x T4*

In **Notebook 4**:
1. Attach final pretrain checkpoint.
2. Build chat dataset and instruction-tune:

```bash
cd /kaggle/working

# 1. Synthesize 150k SFT chat examples (smalltalk, QA, calc tool, search tool, refusals)
python 05_make_sft_data.py

# 2. Train SFT with assistant-only loss masking (3 epochs, low LR 3.5e-5)
torchrun --nproc_per_node=2 06_sft.py
```

---

### Phase 5: Inference, Tools, & Model Export

Test interactive chat or export optimized models:

```bash
cd /kaggle/working

# 1. Interactive terminal chat with Derja assistant
python 07_inference.py --chat

# 2. Export FP16 model (149 MB) for GPU serving
python 07_inference.py --export-fp16 output/tunisiallm_80m_fp16.pt

# 3. Export Dynamic INT8 quantized model (116 MB) for ultra-fast CPU inference
python 07_inference.py --export-int8 output/tunisiallm_80m_int8.pt
```

---

## ⏱️ Expected Timing & Throughput Breakdown

Measured on **2x NVIDIA T4 16GB (Kaggle)**:

| Stage | Script | Hardware | Duration | Throughput / Output |
| :--- | :--- | :--- | :--- | :--- |
| **Data Cleaning** | `01_data_prep.py` | 4-core CPU | ~15-20 min | ~200M tokens cleaned & deduplicated |
| **Tokenizer Training** | `02_tokenizer.py` | 4-core CPU | ~5 min | 16,384 vocab Byte-Level BPE |
| **Binary Sharding** | `02_tokenizer.py` | 4-core CPU | ~8 min | uint16 `.bin` files (~10MB/shard) |
| **Pretraining (80M)** | `04_pretrain.py` | 2x T4 (DDP) | ~18-20 hrs | **~22,000 - 26,000 tok/sec** |
| **SFT Data Gen** | `05_make_sft_data.py` | 4-core CPU | ~10 min | 150,000 chat dialogues |
| **SFT Training** | `06_sft.py` | 2x T4 (DDP) | ~2.5 hrs | 3 epochs with assistant loss masking |
| **Streaming Inference** | `07_inference.py` | 1x T4 GPU | Realtime | **~55 tokens/sec** (KV Cache) |
| **CPU Quantized Inference**| `07_inference.py` | Intel Xeon CPU | Realtime | **~28 tokens/sec** (INT8) |

---

## 🛠️ Tool Calling Syntax & Behavior

TunisiaLLM uses special tokens `<tool_call>` and `</tool_call>` to execute external tools and waits for `<tool_result>` before continuing:

### 1. Math Calculation (`calc: EXPR`)
- **User**: `"قداش 45 في 12 مع 150؟"`
- **Assistant Generation**: `<tool_call>calc: 45 * 12 + 150</tool_call>`
- **Tool Interceptor**: Executes Python AST Safe Calculator $\to$ `690`.
- **Context Injection**: `<tool_result>690</tool_result>`
- **Assistant Completion**: `"النتيجة هي 690 يا باهي."`

### 2. Fact Search (`search: QUERY`)
- **User**: `"وقتاش تأسست جامعة الزيتونة؟"`
- **Assistant Generation**: `<tool_call>search: جامعة الزيتونة تاريخ التأسيس</tool_call>`
- **Tool Interceptor**: Queries knowledge base / search API $\to$ returns verified snippet.
- **Context Injection**: `<tool_result>تأسست جامعة الزيتونة بتونس عام 737 ميلادي...</tool_result>`
- **Assistant Completion**: `"تأسست جامعة الزيتونة سنة 737 ميلادي في عهد الدولة الأموية..."`

### 3. Refusal / Honest Uncertainty
- **User**: `"شكون باش يربح كأس إفريقيا 2034؟"`
- **Assistant**: `"سامحني، هذي حاجة في المستقبل وما يعلم الغيب إلا ربي سبحانه، ما نجمش نتكهن بحاجة ما زالت ما صارتش."`

---

## ⚙️ Configuration (`config.yaml`) Reference

All scripts share the unified `config.yaml`:
- **`model.preset`**: Choose `"30M"`, `"80M"`, or `"125M"`.
- **`pretrain.save_interval_mins`**: Default `20` (wall-clock timer).
- **`pretrain.max_session_hours`**: Default `11.0` (graceful Kaggle timeout guard).
- **`smoke_test`**: Set to `true` (or pass `--smoke-test` on any CLI) for instantaneous dry runs.
