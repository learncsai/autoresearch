# 20-Run Experiment Plan for Autoresearch

**Starting point:** current best recipe family, not the original baseline.

**Current best known config**
- `depth=12`
- realized `dim=512`
- `head_dim=128`
- `TOTAL_BATCH_SIZE=2^18`
- `matrix_lr=0.05`
- `embedding_lr=0.8`

**Guiding logic**
- Your log shows the biggest gains came from more optimizer steps, then better depth/width tradeoffs, while `HEAD_DIM=64`, too-small device batch, and overly large models hurt.
- The closest public H100 autoresearch runs suggest strong upside from shorter sliding windows, more sliding-window layers (`SSSSL`), and larger RoPE base up to around `200k`.
- The modded-nanogpt speedrun stack adds higher-risk but real ideas such as `x0` injection, skip connections, and zero-init projections.

---

## Experiment 1 — Depth 13 at realized dim 512
- **What to run**
  - Increase depth from 12 to 13
  - Keep realized model dimension at 512
  - Keep all other current best settings unchanged
- **Why**
  - This is the cleanest missing point in your architecture frontier.
  - You already know `depth=12, dim=512` is your best and `depth=14, dim=512` is worse.
  - So `depth=13, dim=512` is the most obvious place where the optimum may actually sit.
- **Expectation**
  - One of the highest-probability runs in the whole plan.

## Experiment 2 — Short attention window = `MAX_SEQ_LEN // 4`
- **What to run**
  - Keep architecture the same as best-so-far
  - Reduce the short sliding attention window from half-context to quarter-context
- **Why**
  - Public H100 autoresearch results found that reducing the short window from `1/2` to `1/4` gave a meaningful gain, likely because it cuts attention cost while preserving enough local structure.
- **Expectation**
  - A strong candidate for immediate improvement, especially in a 5-minute budget.

## Experiment 3 — `WINDOW_PATTERN = "SSSSL"`
- **What to run**
  - Change the attention layer pattern from `SSSL` to `SSSSL`
  - Keep short window at the current setting unless Experiment 2 already won
- **Why**
  - The same public H100 autoresearch run reported that using more sliding-window layers was one of the clearest wins.
- **Expectation**
  - Likely positive, especially when paired with shorter short-window attention.

## Experiment 4 — Best short window + `SSSSL`
- **What to run**
  - Combine the best result from Experiment 2 with Experiment 3
  - If `1/4` helped, run `1/4 + SSSSL`
  - If `1/4` did not help, use the best-so-far window setting
- **Why**
  - This is the first deliberate stack of two high-confidence attention wins.
  - The public run suggests these effects can compound.
- **Expectation**
  - One of the most likely combo wins in the full campaign.

## Experiment 5 — RoPE base = 50k
- **What to run**
  - Increase RoPE base from the standard setting to `50_000`
  - Keep best-so-far architecture and attention pattern
- **Why**
  - The public H100 autoresearch sweep found improvement moving RoPE base upward, with gains continuing through higher values before flattening.
- **Expectation**
  - Likely positive or at least neutral.

## Experiment 6 — RoPE base = 200k
- **What to run**
  - Increase RoPE base to `200_000`
  - Use the best-so-far stack from earlier experiments
- **Why**
  - The strongest public signal suggests the sweet spot was around `200k`, so this is a high-upside destination in the sweep.
- **Expectation**
  - Potentially better than 50k and one of the most likely global wins.

## Experiment 7 — Depth 12 at realized dim 576
- **What to run**
  - Keep depth at 12
  - Increase realized dimension from 512 to 576
  - Keep the rest fixed
- **Why**
  - You have evidence that `12×512` works very well and `12×640` OOMs.
  - So `12×576` is the natural middle point.
- **Expectation**
  - A real chance of beating `12×512` if the extra width fits without destroying step count.

## Experiment 8 — `x0` injection to every block
- **What to run**
  - Add normalized input embedding signal into every transformer block
  - Use a learned scalar or per-block scalar gate
- **Why**
  - This is one of the most important aggressive ideas from modded-nanogpt: shorten the gradient path from output back to early representations.
  - The public modded-nanogpt recipe explicitly cites embedding-to-every-block skips as part of the speedrun architecture.
- **Expectation**
  - High upside, but more invasive than the conservative runs.

## Experiment 9 — Zero-init `c_proj` in attention and MLP
- **What to run**
  - Initialize projection outputs in attention and MLP residual branches to zero
  - Keep everything else fixed
- **Why**
  - This is another publicly documented modded-nanogpt trick.
  - It makes blocks start closer to identity and can stabilize very short-budget optimization.
- **Expectation**
  - Moderate-to-high upside, especially when paired with other architectural shortcuts.

## Experiment 10 — Combine `x0` injection + zero-init projections
- **What to run**
  - Stack Experiment 8 and Experiment 9
- **Why**
  - These two ideas are complementary:
    - `x0` injection helps gradient flow
    - zero-init projections stabilize residual learning
- **Expectation**
  - A strong aggressive combo candidate.

## Experiment 11 — ReLU² activation
- **What to run**
  - Replace the current MLP activation with `relu(x) ** 2`
- **Why**
  - ReLU² is part of the modern speedrun transformer recipe in modded-nanogpt.
- **Expectation**
  - Uncertain, but plausible upside in a short training budget.

## Experiment 12 — Extra value embeddings: +1 stream
- **What to run**
  - Add one extra embedding stream mixed into the attention value pathway
- **Why**
  - This is one of the more unusual but real modded-nanogpt ideas: cheap parameter additions that improve learning speed in a low-step regime.
- **Expectation**
  - High upside if your branch benefits from richer early representations.

## Experiment 13 — Best surgery combo
- **What to run**
  - Combine the best two winners among Experiments 8–12
  - Most likely candidates:
    - `x0` injection + zero-init
    - `x0` injection + extra value embeddings
    - zero-init + ReLU²
- **Why**
  - Once you find one or two surgical wins, stacking them is the fastest way to jump beyond conservative tuning.
- **Expectation**
  - Potentially the single highest-upside run in the aggressive block.

## Experiment 14 — Depth 11 at realized dim 640
- **What to run**
  - Set depth to 11
  - Set realized dimension to 640
  - Keep best optimizer settings
- **Why**
  - This is the missing point between strong `depth=10, dim=640` and OOM `depth=12, dim=640`.
- **Expectation**
  - Good upside, but slightly riskier on steps and VRAM than depth 13 at 512.

## Experiment 15 — Depth 13 at realized dim 576
- **What to run**
  - Set depth to 13
  - Set realized dimension to 576
- **Why**
  - This is the aggressive architecture frontier:
    - deeper than current best
    - wider than current best
    - still short of the fully failed region
- **Expectation**
  - High ceiling, but also meaningful risk of losing steps.

## Experiment 16 — Short attention window = `MAX_SEQ_LEN // 8`
- **What to run**
  - Only do this if quarter-window attention helped
  - Reduce short window further to one-eighth of context
- **Why**
  - The public H100 run found gains continuing from `1/4` to `1/8`, but not indefinitely.
- **Expectation**
  - Could beat `1/4`, but should be treated as conditional.

## Experiment 17 — Block skip: 3 → 6
- **What to run**
  - Add a direct residual skip from an early-middle block to a later-middle block, such as block 3 to block 6
- **Why**
  - The modded-nanogpt recipe explicitly mentions block skip connections like this as part of its speedrun stack.
- **Expectation**
  - Another genuine high-upside surgery, but more invasive than the conservative path.

## Experiment 18 — QK-norm
- **What to run**
  - If not already present in your branch, add RMS-like normalization to queries and keys before the attention dot product
- **Why**
  - QK-norm is a major modern stabilization trick and is specifically part of the public modded-nanogpt architecture.
- **Expectation**
  - Could be very good, but only if your current branch truly does not already include its equivalent.

## Experiment 19 — Extra value embeddings: +2 streams
- **What to run**
  - Only do this if +1 stream helped
  - Increase extra value embeddings from one additional stream to two
- **Why**
  - This tests whether the benefit of extra embeddings continues or saturates immediately.
- **Expectation**
  - Conditional experiment only. Skip if +1 was neutral or bad.

## Experiment 20 — Logit softcap with tanh
- **What to run**
  - Apply a soft cap to logits, for example:
    - `logits = s * tanh(logits / s)`
    - with `s ≈ 30`
- **Why**
  - This is a more speculative stabilization trick seen in the broader modded-nanogpt ecosystem, but it is less core than the other surgery ideas.
- **Expectation**
  - Medium-risk, medium-upside. Worth trying only after more grounded ideas.

---

## Recommended execution notes
- Run these **from best-so-far**, not all from the original baseline.
- Skip conditional experiments if their parent experiment failed:
  - Experiment 6 depends loosely on Experiment 5 but can still be run directly
  - Experiment 16 depends on Experiment 2 helping
  - Experiment 19 depends on Experiment 12 helping
- Use a simple keep rule:
  - **keep** if improvement is at least about `0.0003`
  - **discard** if worse by more than `0.001`
- Be careful with experiments that increase VRAM or reduce step count too much; in your log, step loss often killed otherwise promising larger models.

## Highest-confidence runs in this order
- Experiment 1
- Experiment 4
- Experiment 6
- Experiment 7
- Experiment 8
- Experiment 10
