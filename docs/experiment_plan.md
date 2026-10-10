# Experiment Plan: KV Cache and Meaning

English translation of `kv_experiment_plan.md` (the Korean original is the source of truth). Implementation notes that deviate from or refine the proposal are marked **[impl]**.

## 0. Background and research questions

- **RQ1 (priming / persona):** When the same body text is read (a) with no preceding context, (b) after a hint, (c) under an assigned stance or persona, how, where (layer/head), and how much do the body tokens' K/V differ?
- **RQ2 (sequence trajectory / topic shift):** When a conversation on topic A switches with "let's talk about B from now on", is there a metric over the KV cache that quantifies and visualizes the shift?

Core premise: under causal attention, **a past token's KV never changes due to later input.** A "change of direction" can therefore only appear in (1) the K/V of new tokens after the shift and (2) attention patterns from new tokens' Q to past K. RQ1 differences are likewise the result of the prefix being mixed in via attention while body tokens are computed.

## 1. Environment

| Item | Setting |
|---|---|
| Hardware | MacBook M2, 16GB unified memory (MPS) |
| Model | `Qwen/Qwen3-4B-Instruct-2507` (non-thinking; no `<think>` issues). All layers full attention, GQA (36 layers, 32 Q heads, 8 KV heads, head_dim 128) |
| Framework | HF transformers + PyTorch MPS, bf16 |
| Other | numpy, scikit-learn, umap-learn, ruptures, matplotlib |

- Weights ≈ 8GB, so keep sequences under 2k tokens.
- Do not use `output_attentions=True` (36 × 32 × n² at once → OOM). Compute attention per layer from hooks (§2).
- Validate the pipeline on `Qwen3-0.6B` first, then move to 4B.
- **[impl]** Config values are asserted at load (`src/kvmeaning/model.py`). 0.6B: 28 layers, 16 Q heads, 8 KV heads, head_dim 128.

## 2. What to extract

For the same tokens, extract all four; which carries the cleanest semantic signal is itself a result.

1. **V** (`v_proj` output): no RoPE, least positional contamination. First priority.
2. **pre-RoPE K**: Qwen3 applies per-head RMSNorm (`k_norm`) to K before RoPE; hook the `k_norm` output.
3. **post-RoPE K** (what `past_key_values` actually caches): contains position; the difference from (2) gauges positional effect.
4. **Residual stream** (`hidden_states`): baseline; most interpretability work uses this space.

Additionally:
- **Attention mass:** compute softmax(QKᵀ/√d) per layer from hooked post-RoPE Q and cached K, keep only needed statistics (per-span sums), discard the matrix. Account for GQA (4 Q heads share one KV head). **[impl]** Post-RoPE Q is rebuilt from the `q_norm` output plus the `position_embeddings` kwarg, because RoPE is applied inline with no hookable submodule.
- **Storage:** float16 on disk (safetensors). One sample (1k tokens, all layers K+V) ≈ 150MB; subsample layers or sentence-pool if needed. **[impl]** `--layer-stride`.

## 3. Common preprocessing and pitfalls

- **Attention sink / massive activations:** the first token and some special tokens dominate norms and attention. Exclude from analysis; renormalize attention mass after removing the sink.
- **Token alignment:** boundary tokenization may vary with prefix. Put a fixed delimiter (`\n\n`) before the body and assert body token ids are identical across conditions. **[impl]** Segments are tokenized independently and concatenated as ids.
- **Chat template:** exclude template special tokens (`<|im_start|>` etc.) from analysis spans; record body span indices exactly. **[impl]** ChatML rendered by hand (0.6B template injects an empty `<think>` block into the last assistant turn).
- **Normalization:** head norms differ widely; standardize per head before comparing. Report both cosine and L2.
- **Noise floor:** first measure the size of "meaningless difference" (condition F) and report all distances as a ratio to it.
- **Determinism:** run the same input twice and confirm identical results; fold any MPS numeric difference into the noise floor.

## 4. Experiment 1: priming / persona → KV change

### Data
30–50 short passages (200–400 tokens, one language to start), open to interpretation (contentious editorials, ambiguous event descriptions). **[impl]** English; 5 seed passages in `data/exp1_passages.jsonl`.

### Conditions (same body, different prefix)

| Code | Prefix |
|---|---|
| F1, F2 | Different neutral fillers (length-matched). Position control and noise floor |
| H+ | Correct hint ("The following passage is about X") |
| H− | Misleading hint (wrong topic) |
| P1 | Persona A ("You are a skeptical lawyer; read looking for holes") |
| P2 | Persona B ("You are an empathetic counselor") |
| N | No prefix (reference only; position differs) |

All prefixes are **token-length matched** so the body's absolute position is identical, removing RoPE confounds.

### Analysis
1. **Layer×head heatmap:** mean distance between condition pairs on body tokens divided by the noise floor (F1 vs F2).
2. **Layer curve:** difference vs depth. Layer 0 should be ≈0.
3. **Within-body decay curve:** difference vs body token position — does the prefix effect persist or dilute?
4. **Direction vectors:** per layer, mean(P1 − F) as the "persona direction". Cosine between directions from passage A and passage B to test generalization; per-token projection curves.
5. **Linear probe:** classify condition (P1/P2/F) from per-token V; per-layer accuracy curve. Split train/test by passage to prevent leakage.
6. **Behavioral link:** KL between conditions of next-token distribution at body end — does KV difference reach the output?
7. *(Stretch)* **KV patching:** transplant P1 body KV into an F run; does output move toward P1? Causal test.

## 5. Experiment 2: quantifying and visualizing topic shifts

### Data
Chat-template dialogues: 6–10 turns on A → shift → 6–10 turns on B. Vary topic distance: near (soccer↔baseball) to far (soccer↔quantum mechanics).

### Conditions (key: separate declaration from actual content change)

| Code | Construction |
|---|---|
| S1 | Explicit declaration + actual shift |
| S2 | Abrupt shift without declaration |
| S3 | Declaration only, A continues |
| S4 | No shift (A only) — control |
| S5 | Gradual drift (A → intermediate → B) |
| S6 | A → B → back to A (return to original cluster?) |

Record ground-truth indices per sequence: declaration start token, first B-content token.

### Metrics
- **M1. Adjacent distance:** token-level cosine distance between neighbors with moving average (window 16–32); turn-level mean-pooled distance (more stable).
- **M2. Attention mass:** after the shift, fraction of each query's attention on A-span tokens (sink excluded), per layer/head. The most direct metric computable from the KV cache alone.
- **M3. Trajectory:** mid-layer sentence vectors projected to 2D by PCA (and UMAP), connected in time order with a color gradient.
- **M4. Topic probe:** logistic regression trained on separate pure A/B text, applied to the sequence for per-token P(B).
- **M5. Change-point detection:** ruptures (PELT); error vs ground truth in tokens.

### What to look for
- Mark declaration and B-content onset as vertical lines on every curve.
- Does the signal bend at the **declaration or at the content** (S1 vs S3)?
- Which representation (V / pre-RoPE K / post-RoPE K / residual) and layer captures the shift most sharply?
- Correlation between topic distance and signal size.

## 6. Deliverables
- `extract.py`: model load, hooks, per-condition runs, activation storage
- `exp1_priming.ipynb`, `exp2_topic_shift.ipynb`: analysis and figures
- Key figures — Exp1: layer×head heatmap, layer curve, decay curve, probe accuracy curve. Exp2: shift timeline (M1/M2/M4 overlaid), PCA trajectory.
- Short results summary: which representation/layer carries semantic signal best; effect size relative to noise floor.

## 7. Order of work
1. Sanity check on 0.6B: shapes, determinism, F1 vs F1 distance = 0, token alignment assert
2. Noise floor on 4B
3. Exp1 small (5 passages) → check figures → full scale
4. Exp2: S1/S4 first → remaining conditions
5. *(Optional)* Apply the same pipeline to Qwen3.5 (hybrid) and compare attention-layer KV with DeltaNet state changes
