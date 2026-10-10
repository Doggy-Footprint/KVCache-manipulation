# References mapped to design choices

Section numbers refer to `docs/experiment_plan.md`. Metadata (title, authors, year) was checked against the arXiv API. Venue is stated only where a venue page or journal_ref was found. Tier = scored-web-search verdict (academic mode) of the best URL. SKIM-tier entries are preprints with few indexed citations; treat their claims as single-source.

## A. Mechanics of what is being measured (§0, §1, §2)

| Ref | Supports |
|---|---|
| Vaswani et al., "Attention Is All You Need", NeurIPS 2017. https://arxiv.org/abs/1706.03762 (PRIMARY) | Q/K/V decomposition and causal masking. This is why past-token KV is fixed under later input (core premise, §0). |
| Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023. https://aclanthology.org/2023.emnlp-main.298/ (PRIMARY) | Grouped-query attention: groups of Q heads share one KV head. It requires the Q→KV head mapping in attention-mass computation (`attention_mass.py`). |
| Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021. https://arxiv.org/abs/2104.09864 (SUPPORT) | RoPE rotates Q/K by position and leaves V untouched. This justifies V as the least position-contaminated signal, and pre- vs post-RoPE K as a measure of positional effect (§2). |
| Yang et al., "Qwen3 Technical Report", 2025. https://arxiv.org/abs/2505.09388 (SUPPORT) | Model under study, including the QK-norm (per-head RMSNorm before RoPE) that defines the pre-RoPE K hook point. |
| Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories", EMNLP 2021. https://aclanthology.org/2021.emnlp-main.446/ (PRIMARY) | Context for why the residual stream (§2.4) carries content beyond attention K/V. It is the baseline comparison space. |

## B. Preprocessing pitfalls (§3)

| Ref | Supports |
|---|---|
| Xiao et al., "Efficient Streaming Language Models with Attention Sinks", 2023 (ICLR 2024). https://arxiv.org/abs/2309.17453 (SUPPORT) | Initial tokens absorb disproportionate attention. This justifies excluding the sink and renormalizing attention mass (`span_mass(sink=1)`). |
| Sun et al., "Massive Activations in Large Language Models", 2024. https://arxiv.org/abs/2402.17762 (SKIM) | A few activations are orders of magnitude larger and are tied to specific tokens. Motivates excluding special tokens and standardizing per head. |
| Timkey & van Schijndel, "All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality", EMNLP 2021. https://aclanthology.org/2021.emnlp-main.372/ (PRIMARY) | A few dimensions dominate cosine similarity. This justifies standardization before distance and reporting both cosine and L2. |
| Ethayarajh, "How Contextual are Contextualized Word Representations?", EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1006/ (PRIMARY) | Anisotropy: raw cosine between contextual vectors is inflated and varies by layer. Distances need a baseline, which is the noise-floor ratio. |
| Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019. https://proceedings.mlr.press/v97/kornblith19a.html (PRIMARY) | CKA is an invariance-aware similarity measure. A candidate complement to per-token L2/cosine for layer-level comparison. |

## C. Experiment 1 analyses (§4)

| Ref | Supports |
|---|---|
| Alain & Bengio, "Understanding intermediate layers using linear classifier probes", 2016. https://arxiv.org/abs/1610.01644 (SUPPORT) | Per-layer linear probes as a readout of what information is linearly available (analysis 5). |
| Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks", EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1275/ (PRIMARY) | Probe accuracy needs a control/selectivity baseline. This supports keeping probes linear and reporting against F-condition controls with passage-level splits. |
| Turner et al., "Steering Language Models With Activation Engineering", 2023. https://arxiv.org/abs/2308.10248 (SKIM) | Prompt-pair activation differences form usable directions. This is the basis for mean(P1 − F) direction vectors (analysis 4). |
| Panickssery (Rimsky) et al., "Steering Llama 2 via Contrastive Activation Addition", ACL 2024. https://aclanthology.org/2024.acl-long.828/ (PRIMARY) | Mean-difference vectors over contrastive pairs generalize across inputs. This supports the cross-passage cosine test of direction generality. |
| Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency", 2023. https://arxiv.org/abs/2310.01405 (SUPPORT) | Population-level reading of concept directions from contrastive stimuli. A framework for persona/stance directions. |
| Chen et al., "Persona Vectors: Monitoring and Controlling Character Traits in Language Models", 2025. https://arxiv.org/abs/2507.21509 (SUPPORT) | Persona/trait directions extracted from system-prompt contrasts in activation space. This is the closest prior work to P1/P2 conditions. |
| Meng et al., "Locating and Editing Factual Associations in GPT", NeurIPS 2022. https://arxiv.org/abs/2202.05262 (SUPPORT) | Causal tracing by restoring hidden states from a clean run. The template for KV patching (analysis 7). |
| Heimersheim & Nanda, "How to use and interpret activation patching", 2024. https://arxiv.org/abs/2404.15255 (SKIM) | Practical guidance on patching metrics and denoising vs noising direction for the KV-patching stretch goal. |
| Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens", 2023. https://arxiv.org/abs/2303.08112 (SKIM) | Layer-wise decoding of predictions. An optional bridge between per-layer KV differences and output KL (analysis 6). |

## D. Experiment 2 change-point detection (§5, M5)

| Ref | Supports |
|---|---|
| Killick, Fearnhead & Eckley, "Optimal Detection of Changepoints With a Linear Computational Cost", JASA 107(500), 2012. https://doi.org/10.1080/01621459.2012.737745 (PRIMARY) | The PELT algorithm used for automatic shift detection. |
| Truong, Oudre & Vayatis, "Selective review of offline change point detection methods", Signal Processing 167:107299, 2020. https://arxiv.org/abs/1801.00718 (SKIM; journal version confirmed via arXiv journal_ref) | Review and reference for the `ruptures` library, cost functions, and penalty selection. |

29 sources collected → 29 scored (18 citable at PRIMARY/SUPPORT, rest SKIM preprints) → 21 verified via arXiv metadata.
