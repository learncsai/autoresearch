"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage:
  uv run train.py                  # default experiment from DEFAULT_EXPERIMENT
  EXPERIMENT_ID=baseline uv run train.py
  EXPERIMENT_ID=exp01 uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    depth: int = 12
    aspect_ratio: int = 40
    target_dim: int | None = None
    head_dim: int = 128
    window_pattern: str = "SSSL"
    short_window_divisor: int = 2
    total_batch_size: int = 2**18
    device_batch_size: int = 128
    embedding_lr: float = 0.8
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.05
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_betas: tuple[float, float] = (0.8, 0.95)
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0
    rope_base: int = 10000
    rope_seq_len_multiplier: int = 10
    qk_norm: bool = True
    logits_softcap: float = 15.0
    relu_squared: bool = True
    x0_injection: bool = True
    x0_init: float = 0.1
    x0_to_every_block: bool = False  # explicit experimental flag; baseline already has x0 residual path
    zero_init_proj: bool = True
    extra_value_embed_streams: int = 0
    block_skip_from: int | None = None
    block_skip_to: int | None = None

    @property
    def base_dim(self) -> int:
        return self.depth * self.aspect_ratio

    def realized_dim(self) -> int:
        if self.target_dim is not None:
            return self.target_dim
        return ((self.base_dim + self.head_dim - 1) // self.head_dim) * self.head_dim


def _make_experiments() -> dict[str, ExperimentConfig]:
    exps: list[ExperimentConfig] = []
    add = exps.append

    # Baseline from the provided current best-family log origin.
    add(ExperimentConfig(
        name="baseline",
        description="Baseline family from log: depth=12 dim≈512, batch=2^18, matrix_lr=0.05, embedding_lr=0.8",
    ))

    # Ordered 20-run plan from the markdown file.
    add(ExperimentConfig(
        name="exp01",
        description="Experiment 1 — depth 13 at realized dim 512",
        depth=13,
        aspect_ratio=39,  # 13*39=507 -> rounds to 512 with head_dim 128
    ))
    add(ExperimentConfig(
        name="exp02",
        description="Experiment 2 — short attention window = MAX_SEQ_LEN // 4",
        short_window_divisor=4,
    ))
    add(ExperimentConfig(
        name="exp03",
        description='Experiment 3 — WINDOW_PATTERN = "SSSSL"',
        window_pattern="SSSSL",
    ))
    add(ExperimentConfig(
        name="exp04",
        description='Experiment 4 — best short window + "SSSSL"',
        short_window_divisor=4,
        window_pattern="SSSSL",
    ))
    add(ExperimentConfig(
        name="exp05",
        description="Experiment 5 — RoPE base = 50k",
        rope_base=50_000,
    ))
    add(ExperimentConfig(
        name="exp06",
        description="Experiment 6 — RoPE base = 200k",
        rope_base=200_000,
    ))
    add(ExperimentConfig(
        name="exp07",
        description="Experiment 7 — depth 12 at realized dim 576",
        depth=12,
        aspect_ratio=48,
        target_dim=576,
    ))
    add(ExperimentConfig(
        name="exp08",
        description="Experiment 8 — stronger x0 injection to every block",
        x0_to_every_block=True,
        x0_init=0.2,
    ))
    add(ExperimentConfig(
        name="exp09",
        description="Experiment 9 — zero-init c_proj in attention and MLP",
        zero_init_proj=True,
    ))
    add(ExperimentConfig(
        name="exp10",
        description="Experiment 10 — x0 injection + zero-init projections",
        x0_to_every_block=True,
        x0_init=0.2,
        zero_init_proj=True,
    ))
    add(ExperimentConfig(
        name="exp11",
        description="Experiment 11 — ReLU^2 activation",
        relu_squared=True,
    ))
    add(ExperimentConfig(
        name="exp12",
        description="Experiment 12 — extra value embeddings: +1 stream",
        extra_value_embed_streams=1,
    ))
    add(ExperimentConfig(
        name="exp13",
        description="Experiment 13 — best surgery combo seed: x0 injection + zero-init + one extra value embedding stream",
        x0_to_every_block=True,
        x0_init=0.2,
        zero_init_proj=True,
        extra_value_embed_streams=1,
    ))
    add(ExperimentConfig(
        name="exp14",
        description="Experiment 14 — depth 11 at realized dim 640",
        depth=11,
        aspect_ratio=58,  # 11*58=638 -> rounds to 640
    ))
    add(ExperimentConfig(
        name="exp15",
        description="Experiment 15 — depth 13 at realized dim 576",
        depth=13,
        aspect_ratio=44,  # 13*44=572 -> rounds to 640 with head_dim=128? not desired
        head_dim=64,      # 13*44=572 -> rounds to 576 with head_dim=64
        qk_norm=True,
    ))
    add(ExperimentConfig(
        name="exp16",
        description="Experiment 16 — short attention window = MAX_SEQ_LEN // 8",
        short_window_divisor=8,
    ))
    add(ExperimentConfig(
        name="exp17",
        description="Experiment 17 — block skip 3 -> 6",
        block_skip_from=3,
        block_skip_to=6,
    ))
    add(ExperimentConfig(
        name="exp18",
        description="Experiment 18 — QK-norm (explicit on)",
        qk_norm=True,
    ))
    add(ExperimentConfig(
        name="exp19",
        description="Experiment 19 — extra value embeddings: +2 streams",
        extra_value_embed_streams=2,
    ))
    add(ExperimentConfig(
        name="exp20",
        description="Experiment 20 — logit softcap with tanh",
        logits_softcap=30.0,
    ))

    return {cfg.name: cfg for cfg in exps}


EXPERIMENTS = _make_experiments()
DEFAULT_EXPERIMENT = "baseline"
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID", DEFAULT_EXPERIMENT)
assert EXPERIMENT_ID in EXPERIMENTS, f"Unknown EXPERIMENT_ID={EXPERIMENT_ID}. Available: {sorted(EXPERIMENTS)}"
EXP = EXPERIMENTS[EXPERIMENT_ID]

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    short_window_divisor: int = 2
    rope_base: int = 10000
    rope_seq_len_multiplier: int = 10
    qk_norm: bool = True
    logits_softcap: float = 15.0
    relu_squared: bool = True
    x0_injection: bool = True
    x0_to_every_block: bool = False
    x0_init: float = 0.1
    zero_init_proj: bool = True
    extra_value_embed_streams: int = 0
    block_skip_from: int | None = None
    block_skip_to: int | None = None


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.extra_value_embed_streams = config.extra_value_embed_streams
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.qk_norm = config.qk_norm
        self.ve_gate_channels = 32
        gate_heads = self.n_kv_head * (1 + self.extra_value_embed_streams)
        self.ve_gate = nn.Linear(self.ve_gate_channels, gate_heads, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve_list, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve_list:
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            gate = gate.view(B, T, len(ve_list), self.n_kv_head)
            for j, ve in enumerate(ve_list):
                ve = ve.view(B, T, self.n_kv_head, self.head_dim)
                v = v + gate[:, :, j].unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if self.qk_norm:
            q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relu_squared = config.relu_squared
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        if self.relu_squared:
            x = F.relu(x).square()
        else:
            x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve_list, cos_sin, window_size):
        x = x + self.attn(norm(x), ve_list, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.x0_layer_gates = nn.Parameter(torch.zeros(config.n_layer)) if config.x0_to_every_block else None
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.ModuleList([nn.Embedding(config.vocab_size, kv_dim) for _ in range(1 + config.extra_value_embed_streams)])
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * config.rope_seq_len_multiplier
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, base=config.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            if self.config.zero_init_proj:
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            else:
                torch.nn.init.uniform_(block.attn.c_proj.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            if self.config.zero_init_proj:
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            else:
                torch.nn.init.uniform_(block.mlp.c_proj.weight, -s, s)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(self.config.x0_init)
        if self.x0_layer_gates is not None:
            self.x0_layer_gates.fill_(0.0)
        for ve_modules in self.value_embeds.values():
            for ve in ve_modules:
                torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, base=self.config.rope_base)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve_modules in self.value_embeds.values():
            for ve in ve_modules:
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = max(1, long_window // config.short_window_divisor)
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve_modules in self.value_embeds.values() for ve in ve_modules)
        x0_gate_numel = 0 if self.x0_layer_gates is None else self.x0_layer_gates.numel()
        nparams_exclude = (
            self.transformer.wte.weight.numel() + value_embeds_numel +
            self.resid_lambdas.numel() + self.x0_lambdas.numel() + x0_gate_numel
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        if self.x0_layer_gates is not None:
            scalars += self.x0_layer_gates.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        scalar_params = resid_params + x0_params
        if self.x0_layer_gates is not None:
            scalar_params.append(self.x0_layer_gates)
        total_params = len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(scalar_params)
        assert len(list(self.parameters())) == total_params
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if self.x0_layer_gates is not None:
            param_groups.append(dict(kind='adamw', params=[self.x0_layer_gates], lr=scalar_lr * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        saved_skip = None
        for i, block in enumerate(self.transformer.h):
            if self.config.x0_injection:
                x0_term = self.x0_lambdas[i] * x0
                if self.x0_layer_gates is not None:
                    x0_term = x0_term * torch.sigmoid(self.x0_layer_gates[i]) * 2.0
                x = self.resid_lambdas[i] * x + x0_term
            else:
                x = self.resid_lambdas[i] * x

            if str(i) in self.value_embeds:
                ve_modules = self.value_embeds[str(i)]
                ve_list = [ve(idx) for ve in ve_modules]
            else:
                ve_list = []
            x = block(x, ve_list, cos_sin, self.window_sizes[i])

            if self.config.block_skip_from is not None and i == self.config.block_skip_from:
                saved_skip = x
            if self.config.block_skip_to is not None and i == self.config.block_skip_to and saved_skip is not None:
                x = x + saved_skip
        x = norm(x)

        logits = self.lm_head(x).float()
        if self.config.logits_softcap > 0:
            softcap = self.config.logits_softcap
            logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'], self._adamw_step_t, self._adamw_lr_t,
                             self._adamw_beta1_t, self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params, state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Experiment: {EXPERIMENT_ID}")
print(f"Description: {EXP.description}")
print(f"Vocab size: {vocab_size:,}")


def build_model_config(exp: ExperimentConfig):
    model_dim = exp.realized_dim()
    assert model_dim % exp.head_dim == 0, f"realized dim {model_dim} must be divisible by head_dim {exp.head_dim}"
    num_heads = model_dim // exp.head_dim
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=exp.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=exp.window_pattern,
        short_window_divisor=exp.short_window_divisor,
        rope_base=exp.rope_base,
        rope_seq_len_multiplier=exp.rope_seq_len_multiplier,
        qk_norm=exp.qk_norm,
        logits_softcap=exp.logits_softcap,
        relu_squared=exp.relu_squared,
        x0_injection=exp.x0_injection,
        x0_to_every_block=exp.x0_to_every_block,
        x0_init=exp.x0_init,
        zero_init_proj=exp.zero_init_proj,
        extra_value_embed_streams=exp.extra_value_embed_streams,
        block_skip_from=exp.block_skip_from,
        block_skip_to=exp.block_skip_to,
    )


config = build_model_config(EXP)
print(f"Model config: {asdict(config)}")
print(f"Requested base_dim={EXP.base_dim}, target_dim={EXP.target_dim}, realized dim={config.n_embd}, n_head={config.n_head}, head_dim={EXP.head_dim}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = EXP.device_batch_size * MAX_SEQ_LEN
assert EXP.total_batch_size % tokens_per_fwdbwd == 0
grad_accum_steps = EXP.total_batch_size // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=EXP.unembedding_lr,
    embedding_lr=EXP.embedding_lr,
    scalar_lr=EXP.scalar_lr,
    adam_betas=EXP.adam_betas,
    matrix_lr=EXP.matrix_lr,
    weight_decay=EXP.weight_decay,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, EXP.device_batch_size, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")


def get_lr_multiplier(progress):
    if progress < EXP.warmup_ratio:
        return progress / EXP.warmup_ratio if EXP.warmup_ratio > 0 else 1.0
    elif progress < 1.0 - EXP.warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / EXP.warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * EXP.final_lr_frac


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return EXP.weight_decay * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(EXP.total_batch_size / dt)
    mfu = 100 * num_flops_per_token * EXP.total_batch_size / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
        f"dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | "
        f"remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * EXP.total_batch_size
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, EXP.device_batch_size)

t_end = time.time()
steady_state_mfu = 100 * num_flops_per_token * EXP.total_batch_size * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"experiment_id:     {EXPERIMENT_ID}")
print(f"description:       {EXP.description}")
print(f"val_bpb:           {val_bpb:.6f}")
print(f"training_seconds:  {total_training_time:.1f}")
print(f"total_seconds:     {t_end - t_start:.1f}")
print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
print(f"mfu_percent:       {steady_state_mfu:.2f}")
print(f"total_tokens_M:    {total_tokens / 1e6:.1f}")
print(f"num_steps:         {step}")
print(f"num_params_M:      {num_params / 1e6:.1f}")
print(f"depth:             {config.n_layer}")
print(f"realized_dim:      {config.n_embd}")
