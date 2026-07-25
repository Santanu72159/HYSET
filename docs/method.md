# HYSET Method Overview
---

## 1. Problem

ToolBench exposes 13,860 callable API endpoints after its official filtering. No agent can place every tool schema into a prompt, so a retriever has to shortlist first.

Existing retrievers score each tool on its own. BM25 matches text. Dense bi-encoders such as Contriever and ToolLLaMA-Retriever compare a query embedding against a tool embedding. Graph methods add pairwise tool edges. Generative retrievers emit tool identifiers token by token. All of them return a ranked list, and a ranked list has no way to say that flights, hotel, and weather are useful precisely because they arrive together.

Real queries are usually resolved by several APIs invoked jointly, so the object worth retrieving is a set.

---

## 2. Formulation

HYSET treats tool retrieval as query-conditioned hyperedge prediction on a tool co-invocation hypergraph. Every tool is a node, every ground truth tool set is a hyperedge, and the query is the condition. Scoring a candidate set `E` against a query `x` is additive.

```
F(x, E) = F_set(E) + F_align(x, E)
```

Written this way, most existing retrieval paradigms reduce to restricted instances. Independent scoring is what is left when `F_set` is removed and `E` is a singleton.

The trainable parameters are the tool embedding matrix `Z`, the cardinality-specific interaction matrices `{M_m}`, and the projection `P`. The query encoder stays frozen throughout.

---

## 3. Set-level scoring

Each tool `j` carries a learnable embedding `z_j`. For a candidate set of `m` tools, HYSET sums the interaction of every tool pair through a matrix that belongs to that particular cardinality.

```
F_set(E) = Σ_{a < b}  z_{j_a}ᵀ  M_m  z_{j_b}
```

Pairs go through `M_2`, triples through `M_3`, and so on up to the maximum cardinality `M`. Because `M_m` depends on `m`, the same pair of tools can contribute differently in a set of two and in a set of four. That is the point of the design. Compatibility between two tools is not a fixed quantity, it depends on how large the surrounding set is.

A singleton has no pair to sum over, so `F_set` vanishes and the score falls back to alignment alone.

---

## 4. Query-set alignment

A frozen pre-trained encoder maps the query to `r(x)`. The paper reports two backbones, a BERT-base retriever and a tuned Qwen2.5-1.5B retriever, and the gains hold for both.

Tool embeddings are projected into the query space by `P`, and single-head attention with `r(x)` as the query over `{P z_j}` produces a query-conditioned representation of the set.

```
F_align(x, E) = Σ_k  α_k(x, E) · ℓ(x, t_{j_k})     where  ℓ(x, t_j) = r(x)ᵀ P z_j
```

Every attention weight depends on all tools in `E`, so the alignment term is a property of the set rather than a sum of independent matches. For a singleton this collapses to `r(x)ᵀ P z_j`, which is ordinary dense query-tool matching.

---

## 5. Training

Two signals drive learning.

**Set-level retrieval loss.** For each query the ground truth set is contrasted against `K - 1` sampled negative sets in a softmax. The negatives come from three sources in a 50/30/20 mixture. Half are size-matched sets drawn uniformly, thirty percent are ground truth sets belonging to other queries in the same minibatch, and twenty percent are hard negatives built by swapping one or two tools of the ground truth set for their nearest neighbours under the tool embeddings.

**Reward-weighted self-training loss.** The highest scoring candidate is handed to a frozen agent, which runs DFSDT for at most `R` steps. The resulting answer receives an execution reward `ρ = ExecScore(x, ŷ)` in `[0, 1]`. Successful selections are reinforced in proportion to that reward, and a selection with zero reward contributes no gradient. This lets the model reinforce sets that work even when they are not the annotated ones.

The full objective adds the two losses and regularises the interaction matrices, under a unit-norm constraint on every tool embedding.

```
L = L_ret + η · L_self + λ · Σ_m ‖M_m‖_F²
```

---

## 6. Inference

Enumerating every subset of the corpus is impossible, so HYSET exploits the additive form of `F` and splits inference into two stages.

**Stage 1.** Score each tool on its own with `r(x)ᵀ P z_j` and keep the `K_pool` highest as a shortlist `S`.

**Stage 2.** Build the reduced space of subsets of `S` with cardinality between 1 and `M`, score each one with the full `F(x, E)`, and return the best.

`K_pool` and `M` are chosen so that the reduced space stays inside the inference budget. Stage 1 finds plausible tools and Stage 2 finds a combination that works together. The selected set then goes to the downstream agent unchanged, which is why HYSET needs no modification anywhere else in the pipeline.

---

## 7. Data and evaluation

The corpus holds 13,860 tools in `data/hyset_corpus.json`. Evaluation uses the six official held-out ToolBench splits in `data/test_query_ids/`, which is 600 queries in total. Retrieval quality is measured by Recall@K and NDCG@K, set completeness by COMP@K, and downstream utility by Pass Rate under the official ToolEval protocol. Baselines are BM25, Contriever, ToolLLaMA-Retriever, COLT, and ToolGen. Training ran on two NVIDIA RTX 4090 cards with 24 GB each.

Hyperparameters live at the top of the shell scripts in the repository root, which are the source of truth for what was actually run. See [`README.md`](../README.md) for the commands.

---

## 8. Design take-away

HYSET keeps two decisions apart. HYSET decides which tools should be available together, and the agent decides how to reason over them. Everything above exists to make the first decision at the level of a set instead of one tool at a time.
