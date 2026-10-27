"""
03_model.py - LLaMA-style Decoder Architecture for Small Derja LLM
Features:
  - Rotary Position Embeddings (RoPE)
  - Root Mean Square Layer Normalization (RMSNorm)
  - SwiGLU Feed-Forward Networks
  - Grouped-Query Attention (GQA) with KV Cache
  - Tied Word Embeddings (tok_embeddings == lm_head)
  - Native PyTorch scaled_dot_product_attention (SDPA) for fast T4 execution without FlashAttention
  - Presets for 30M, 80M, and 125M parameter configurations
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import format_num_params


@dataclass
class ModelConfig:
    dim: int = 768
    n_layers: int = 14
    n_heads: int = 12
    n_kv_heads: int = 4
    hidden_dim: int = 2688
    max_seq_len: int = 1024
    vocab_size: int = 16384
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False

    @classmethod
    def from_preset(cls, preset_name: str = "80M", vocab_size: int = 16384, max_seq_len: int = 1024) -> "ModelConfig":
        """Factory method for predefined architecture presets matching exact parameter counts."""
        presets = {
            "30M": {
                "dim": 448,
                "n_layers": 8,
                "n_heads": 7,
                "n_kv_heads": 1,
                "hidden_dim": 1408,
            },
            "80M": {
                "dim": 672,
                "n_layers": 12,
                "n_heads": 12,
                "n_kv_heads": 4,
                "hidden_dim": 2176,
            },
            "125M": {
                "dim": 768,
                "n_layers": 14,
                "n_heads": 12,
                "n_kv_heads": 4,
                "hidden_dim": 2688,
            }
        }
        if preset_name not in presets:
            raise ValueError(f"Unknown preset {preset_name}. Available: {list(presets.keys())}")

        cfg_dict = presets[preset_name]
        return cls(
            dim=cfg_dict["dim"],
            n_layers=cfg_dict["n_layers"],
            n_heads=cfg_dict["n_heads"],
            n_kv_heads=cfg_dict["n_kv_heads"],
            hidden_dim=cfg_dict["hidden_dim"],
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            tie_word_embeddings=True,
            gradient_checkpointing=False
        )


# ==============================================================================
# Normalization: RMSNorm
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization without bias."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ==============================================================================
# Positional Embeddings: RoPE
# ==============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    # xq shape: (B, seq_len, n_heads, head_dim)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis: (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, xq.shape[-1] // 2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat Key/Value heads when n_kv_heads < n_heads (Grouped-Query Attention)."""
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ==============================================================================
# Grouped-Query Attention (GQA) with KV Cache & Native PyTorch SDPA
# ==============================================================================

class Attention(nn.Module):
    """Grouped-Query Attention with optional KV cache and PyTorch SDPA."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        # KV cache tensors for inference
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        """Reset KV cache for new inference generation."""
        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # Linear projections
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape to (bsz, seqlen, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # KV Cache logic for autoregressive decoding
        if use_cache:
            if self.k_cache is None or self.k_cache.shape[0] != bsz:
                self.k_cache = torch.zeros(
                    (bsz, 2048, self.n_kv_heads, self.head_dim),
                    device=x.device,
                    dtype=x.dtype
                )
                self.v_cache = torch.zeros(
                    (bsz, 2048, self.n_kv_heads, self.head_dim),
                    device=x.device,
                    dtype=x.dtype
                )
            self.k_cache[:bsz, start_pos:start_pos + seqlen] = xk
            self.v_cache[:bsz, start_pos:start_pos + seqlen] = xv

            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]
        else:
            keys = xk
            values = xv

        # Repeat KV heads for Grouped-Query Attention
        keys = repeat_kv(keys, self.n_rep)      # (bsz, total_seqlen, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bsz, total_seqlen, n_heads, head_dim)

        # Transpose to (bsz, n_heads, seqlen, head_dim) for SDPA
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled Dot-Product Attention (SDPA)
        # In PyTorch, SDPA automatically uses optimized math/mem-efficient kernels on T4
        is_causal = (seqlen > 1 and mask is None)
        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal
        )

        # Transpose back to (bsz, seqlen, dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


# ==============================================================================
# Feed-Forward: SwiGLU
# ==============================================================================

class FeedForward(nn.Module):
    """SwiGLU feed-forward network: w2(SiLU(w1(x)) * w3(x))."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)  # down
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ==============================================================================
# Transformer Block
# ==============================================================================

class TransformerBlock(nn.Module):
    """Single LLaMA transformer layer with pre-norm RMSNorm."""
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            start_pos=start_pos,
            freqs_cis=freqs_cis,
            mask=mask,
            use_cache=use_cache
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ==============================================================================
# Full Decoder Model: TunisiaLLM
# ==============================================================================

class TunisiaLLM(nn.Module):
    """
    LLaMA-style autoregressive decoder language model tailored for Tunisian Derja.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len

        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id=i, config=config)
            for i in range(config.n_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with output projection
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE complex frequencies
        head_dim = config.dim // config.n_heads
        freqs_cis = precompute_freqs_cis(
            dim=head_dim,
            end=config.max_seq_len * 2,
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_cache(self) -> None:
        """Reset KV cache in all attention layers."""
        for layer in self.layers:
            layer.attention.reset_cache()

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            tokens: (bsz, seqlen) input token IDs
            targets: (bsz, seqlen) target token IDs (optional)
            start_pos: current token position for KV caching
            use_cache: whether to use KV cache during generation
        Returns:
            logits: (bsz, seqlen, vocab_size)
            loss: scalar CrossEntropyLoss (if targets is not None)
        """
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # RoPE frequencies slice for current sequence position
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        # Attention mask for caching or padding
        mask = None
        if seqlen > 1 and use_cache:
            total_len = start_pos + seqlen
            q_pos = torch.arange(start_pos, total_len, device=tokens.device).unsqueeze(1)
            k_pos = torch.arange(total_len, device=tokens.device).unsqueeze(0)
            mask = torch.zeros((seqlen, total_len), device=tokens.device)
            mask = mask.masked_fill(q_pos < k_pos, float("-inf")).type_as(h)

        # Pass through transformer layers
        for layer in self.layers:
            if self.training and getattr(self.config, "gradient_checkpointing", False) and not use_cache:
                h = torch.utils.checkpoint.checkpoint(
                    layer,
                    h,
                    start_pos,
                    freqs_cis,
                    mask,
                    use_cache,
                    use_reentrant=False
                )
            else:
                h = layer(
                    h,
                    start_pos=start_pos,
                    freqs_cis=freqs_cis,
                    mask=mask,
                    use_cache=use_cache
                )

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            # Shift tokens for next-token prediction if not already aligned
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def get_num_params(self) -> Dict[str, int]:
        """Compute total, tied, and unique trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        unique_params = sum(
            p.numel() for name, p in self.named_parameters()
            if not (self.config.tie_word_embeddings and name == "lm_head.weight")
        )
        return {
            "total_parameters": total_params,
            "unique_trainable_parameters": unique_params,
            "tied_savings": total_params - unique_params if not self.config.tie_word_embeddings else self.tok_embeddings.weight.numel()
        }


# ==============================================================================
# Model Architecture Unit Test & Preset Summary
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TunisiaLLM Architecture Inspection & Presets")
    print("=" * 80)

    for preset in ["30M", "80M", "125M"]:
        cfg = ModelConfig.from_preset(preset, vocab_size=16384, max_seq_len=1024)
        model = TunisiaLLM(cfg)
        p_stats = model.get_num_params()
        print(f"\nPreset: [{preset}]")
        print(f"  - Layers: {cfg.n_layers} | Dim: {cfg.dim} | Heads: {cfg.n_heads} | KV-Heads: {cfg.n_kv_heads} (GQA)")
        print(f"  - Hidden Dim (SwiGLU): {cfg.hidden_dim}")
        print(f"  - Unique Trainable Params: {format_num_params(p_stats['unique_trainable_parameters'])} ({p_stats['unique_trainable_parameters']:,})")
        print(f"  - Tied Embedding Savings:  {format_num_params(p_stats['tied_savings'])}")

    # Run forward pass test with default 80M
    print("\n" + "=" * 80)
    print("Running Forward Pass & Loss Unit Test on 80M Model...")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_80m = ModelConfig.from_preset("80M", vocab_size=16384, max_seq_len=1024)
    model = TunisiaLLM(cfg_80m).to(device)

    # Dummy batch
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, cfg_80m.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, cfg_80m.vocab_size, (batch_size, seq_len), device=device)

    logits, loss = model(x, targets=y)
    print(f"[+] Input shape:  {x.shape}")
    print(f"[+] Logits shape: {logits.shape}")
    print(f"[+] Loss scalar:  {loss.item():.4f}")
    assert logits.shape == (batch_size, seq_len, cfg_80m.vocab_size), "Logits shape mismatch!"
    assert loss is not None and not torch.isnan(loss), "Loss computation failed!"

    # Test KV cache step
    print("\n[+] Testing Autoregressive KV Cache Step...")
    model.reset_cache()
    # Prompt phase
    logits_prompt, _ = model(x[:, :10], use_cache=True, start_pos=0)
    # Generation step (single token)
    next_token = torch.argmax(logits_prompt[:, -1:, :], dim=-1)
    logits_step, _ = model(next_token, use_cache=True, start_pos=10)
    print(f"    KV Cache step output shape: {logits_step.shape}")
    assert logits_step.shape == (batch_size, 1, cfg_80m.vocab_size), "KV cache step shape mismatch!"

    print("\n[PASSED] TunisiaLLM Model Architecture verified successfully!")
