import math
import os
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer



def _standardize(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).lower().strip("_")
    return ("get_" + s) if s and s[0].isdigit() else s


def _change_name(n: str) -> str:
    return ("is_" + n) if n in {"from", "class", "return", "false", "true", "id", "and"} else n


def func_name_of(tool: str, api: str) -> str:
    return f"{_change_name(_standardize(api))}_for_{_standardize(tool)}"

class HYSETModel(nn.Module):
    """
    Full HYSET model (encoder frozen).

    Trainable parameters: U (=Z), W (=P^T), log_tau, {M_m}_{m=2..m_max}
    Encoder g_psi is frozen; only W, U, M_m are updated during training.

    encoder_type: "bert" uses mean-pool (bidirectional);
                  "qwen" uses last-non-pad-token pool (causal LM).
    """

    def __init__(
        self,
        encoder_path: str,
        func_names: List[str],
        d: int = 256,
        m_max: int = 4,
        encoder_type: str = "bert",
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── query encoder ─────────────────────────────────────────────────────
        self.encoder_type = encoder_type
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        encoder_dtype = "auto" if encoder_type == "qwen" else None
        self.encoder_model = AutoModel.from_pretrained(
            encoder_path, torch_dtype=encoder_dtype,
        )
        d_q = self.encoder_model.config.hidden_size

        self.func_names = func_names
        self.fn_to_idx  = {fn: i for i, fn in enumerate(func_names)}
        N_T             = len(func_names)
        self.U       = nn.Embedding(N_T, d)          # Z in paper
        self.W       = nn.Linear(d_q, d, bias=False) 
        if encoder_type == "bert" and d == d_q:
            nn.init.eye_(self.W.weight)
        else:
            nn.init.xavier_uniform_(self.W.weight)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))

        self.D = nn.ParameterDict({
            str(m): nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.01)
            for m in range(2, m_max + 1)
        })

        self.dropout_u = nn.Dropout(dropout)
        self.dropout_q = nn.Dropout(dropout)

        self.d     = d
        self.d_q   = d_q
        self.m_max = m_max
        self.N_T   = N_T

    @torch.no_grad()
    def init_api_embeddings(self, api_texts: List[str], batch_size: int = 256):
        """Initialise U by projecting encoder (BERT/Qwen) embeddings."""
        device = next(self.parameters()).device
        target_dtype = self.W.weight.dtype
        all_embs = []
        for start in range(0, len(api_texts), batch_size):
            batch = api_texts[start: start + batch_size]
            enc   = self.tokenizer(batch, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt").to(device)
            out   = self.encoder_model(**enc)
            emb   = self._pool(out.last_hidden_state, enc["attention_mask"])
            all_embs.append(emb.to(target_dtype))   # cast bf16→fp32 if needed
        embs      = torch.cat(all_embs, dim=0)   # (N_T, d_q) in W's dtype
        projected = self.W(embs)                  # (N_T, d)
        self.U.weight.data.copy_(projected)
        print(f"[HYSET] Initialized U from encoder: {projected.shape}")


    @torch.no_grad()
    def init_api_embeddings_from_toolgen(
        self,
        toolgen_path: str,
        corpus: dict,
        n_base_vocab: int = 151_665,
        api_texts_fallback: Optional[List[str]] = None,
    ) -> int:
        try:
            from safetensors import safe_open
        except ImportError:
            print("[HYSET] safetensors not installed. Run: pip install safetensors")
            return 0

        from transformers import AutoTokenizer as AT

        device = next(self.parameters()).device

        print(f"[HYSET] Loading ToolGen tokenizer from {toolgen_path} …")
        tg_tok = AT.from_pretrained(toolgen_path)
        vocab  = tg_tok.get_vocab()

        fn_to_local_idx: Dict[str, int] = {}
        for fn, entry in corpus.items():
            if fn not in self.fn_to_idx:
                continue
            aj        = entry.get("full_api_json", {})
            tool_name = aj.get("tool_name", "")
            api_name  = aj.get("api_name", "")
            tok_str   = f"<<{tool_name}&&{api_name}>>"
            tid       = vocab.get(tok_str)
            if tid is not None and tid >= n_base_vocab:
                fn_to_local_idx[fn] = tid - n_base_vocab

        n_matched = len(fn_to_local_idx)
        if n_matched == 0:
            print("[HYSET] No ToolGen virtual tokens matched.")
            return 0
        print(f"[HYSET] Matched {n_matched}/{self.N_T} APIs to ToolGen virtual tokens.")
        st_path = os.path.join(toolgen_path, "model.safetensors")
        print(f"[HYSET] Reading ToolGen embed_tokens from {st_path} …")
        with safe_open(st_path, framework="pt", device="cpu") as f:
            tg_emb = f.get_tensor("model.embed_tokens.weight").float()

        d_toolgen   = tg_emb.shape[1]
        all_virtual = tg_emb[n_base_vocab:]  
        if self.d == d_toolgen:
            print(f"[HYSET] d ({self.d}) == d_toolgen ({d_toolgen}); "
                  f"copying virtual-token embeddings directly (no PCA).")
            for fn, local_idx in fn_to_local_idx.items():
                u_idx = self.fn_to_idx[fn]
                self.U.weight.data[u_idx] = all_virtual[local_idx].to(device)
        else:
            print(f"[HYSET] d ({self.d}) < d_toolgen ({d_toolgen}); "
                  f"PCA-projecting {d_toolgen} → {self.d}.")
            matched_local_idxs = list(fn_to_local_idx.values())
            matched_embs       = all_virtual[matched_local_idxs]   # (N_matched, d_tg)
            mu                 = matched_embs.mean(0)
            centered           = matched_embs - mu
            _, _, Vt           = torch.linalg.svd(centered, full_matrices=False)
            proj_mat           = Vt[: self.d].T                    # (d_tg, d)
            for fn, local_idx in fn_to_local_idx.items():
                u_idx = self.fn_to_idx[fn]
                emb_d = ((all_virtual[local_idx] - mu) @ proj_mat).to(device)
                self.U.weight.data[u_idx] = emb_d

        unmatched_fns = [fn for fn in self.fn_to_idx if fn not in fn_to_local_idx]
        if unmatched_fns and api_texts_fallback is not None:
            print(f"[HYSET] {len(unmatched_fns)} unmatched APIs — encoding their text "
                  f"with frozen g_psi for U init …")
            # Build idx → text map for unmatched ones
            fn_text_pairs = [(fn, api_texts_fallback[self.fn_to_idx[fn]])
                             for fn in unmatched_fns]
            texts = [t for _, t in fn_text_pairs]

            embs_all: List[torch.Tensor] = []
            batch_size = 64
            target_dtype = self.W.weight.dtype
            for s in range(0, len(texts), batch_size):
                batch = texts[s: s + batch_size]
                enc   = self.tokenizer(batch, padding=True, truncation=True,
                                       max_length=128, return_tensors="pt").to(device)
                out   = self.encoder_model(**enc)
                emb   = self._pool(out.last_hidden_state, enc["attention_mask"])
                embs_all.append(emb.to(target_dtype))   # bf16→fp32 if needed
            unm_embs = torch.cat(embs_all, dim=0)  # (N_unmatched, d_q) in fp32

            if self.d == self.d_q:
                for (fn, _), emb in zip(fn_text_pairs, unm_embs):
                    self.U.weight.data[self.fn_to_idx[fn]] = emb
            else:
                proj = self.W(unm_embs)  # (N_unmatched, d)
                for (fn, _), emb in zip(fn_text_pairs, proj):
                    self.U.weight.data[self.fn_to_idx[fn]] = emb
            print(f"[HYSET] Unmatched APIs initialized via frozen g_psi encoding.")
        elif unmatched_fns:
            print(f"[HYSET][WARN] {len(unmatched_fns)} unmatched APIs but no "
                  f"api_texts_fallback provided — those U entries remain randomly init.")

        print(f"[HYSET] U init done: {n_matched} from ToolGen vocab, "
              f"{len(unmatched_fns)} from g_psi fallback.")
        return n_matched


    def _pool(self, hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "qwen":
            lengths = attn_mask.sum(dim=1) - 1               # (B,) last real token pos
            idx     = lengths.clamp(min=0)
            return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        else:
            mask = attn_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    # ── query encoding (with gradient) ────────────────────────────────────────

    def freeze_encoder(self):
        for p in self.encoder_model.parameters():
            p.requires_grad_(False)
        self._encoder_frozen = True

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        enc = self.tokenizer(
            queries, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        # Eval mode OR explicit freeze → run under no_grad and detach result.
        no_grad = getattr(self, "_encoder_frozen", False) or (not self.training)
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            out = self.encoder_model(**enc)
            pooled = self._pool(out.last_hidden_state, enc["attention_mask"])
            # Cast to W.weight dtype so downstream W(q) doesn't dtype-mismatch
            # when encoder is loaded in bf16 (Qwen) but W is fp32.
            pooled = pooled.to(self.W.weight.dtype)
            return pooled.detach() if no_grad else pooled
    def score_sets(
        self,
        q: torch.Tensor,       # (d_q,)  single query vector
        api_idx: torch.Tensor, # (B, m)  API indices, all same size m
    ) -> torch.Tensor:         # (B,)    scores
        B, m = api_idx.shape

        u      = self.U(api_idx)                              # (B, m, d)
        u      = self.dropout_u(u)                            # (B, m, d) dropout on U
        q_proj = self.dropout_q(self.W(q))                    # (d,) dropout on W(q)
        q_proj = F.normalize(q_proj, dim=-1)                  # (d,) unit norm
        u_norm = F.normalize(u, dim=-1)                       # (B, m, d) unit norm

        attn  = torch.einsum("d,bmd->bm", q_proj, u_norm)
        alpha = torch.softmax(attn, dim=-1)
        h     = torch.einsum("bm,bmd->bd", alpha, u_norm)
        f2    = torch.einsum("d,bd->b",   q_proj, h)

        if m < 2:
            return f2

        m_str   = str(m)
        D_m     = self.D[m_str]
        Du      = torch.einsum("de,bme->bmd", D_m, u_norm)
        gram    = torch.einsum("bmd,bnd->bmn", u_norm, Du)
        mask    = torch.triu(torch.ones(m, m, device=q.device), diagonal=1).bool()
        f1      = gram[:, mask].sum(dim=-1)

        return f2 + f1


    @torch.no_grad()
    def dense_scores(self, q: torch.Tensor) -> torch.Tensor:
        """Cosine similarities between projected query and all API embeddings."""
        q_proj = F.normalize(self.W(q), dim=-1)
        U_norm = F.normalize(self.U.weight, dim=-1)
        return q_proj @ U_norm.T


    @torch.no_grad()
    def _greedy_retrieve(
        self,
        q: torch.Tensor,
        pool_idx: List[int],
        m_max: int,
        k_return: int,
        device: str,
    ) -> List[int]:
        selected: List[int] = []
        remaining = list(pool_idx)

        for step in range(min(m_max, k_return)):
            if not remaining:
                break
            sets    = [selected + [c] for c in remaining]     # (|rem|, step+1)
            sets_t  = torch.tensor(sets, device=device)
            scores  = self.score_sets(q, sets_t)               # (|rem|,)
            best    = int(scores.argmax().item())
            selected.append(remaining[best])
            remaining.pop(best)

        return selected

    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        target_size: Optional[int] = None,
        k_pool: int = 50,
        k_return: int = 5,
        return_predicted_set: bool = False,
        normalize_across_orders: bool = False,
        use_greedy: bool = False,
    ) -> List[str]:
        device = next(self.parameters()).device

        # Force greedy when m_max ≥ 5: exhaustive enumeration is infeasible.
        if self.m_max >= 5 and not use_greedy:
            use_greedy = True

        q = self.encode_queries([query])[0]             # (d_q,)
        sims = self.dense_scores(q)                     # (N_T,)
        pool_idx = torch.topk(sims, k=min(k_pool, self.N_T)).indices.tolist()

        if use_greedy:
            m_greedy    = target_size or min(self.m_max, k_return)
            best_set_idx = self._greedy_retrieve(q, pool_idx, m_greedy, k_return, device)
        else:
            orders = [target_size] if target_size else list(range(1, min(self.m_max, k_return) + 1))

            best_score   = float("-inf")
            best_set_idx = pool_idx[:k_return]

            for m in orders:
                if m > len(pool_idx):
                    continue
                if m == 1:
                    api_t  = torch.tensor(pool_idx, device=device).unsqueeze(1)
                    scores = self.score_sets(q, api_t)
                else:
                    combos = list(combinations(range(len(pool_idx)), m))
                    if not combos:
                        continue
                    api_t  = torch.tensor(
                        [[pool_idx[j] for j in c] for c in combos], device=device
                    )
                    scores = self.score_sets(q, api_t)

                top_s, top_pos = scores.max(dim=0)

                if normalize_across_orders:
                    mu      = scores.mean()
                    sigma   = scores.std().clamp(min=1e-9)
                    compare = ((top_s - mu) / sigma).item()
                else:
                    compare = top_s.item()

                if compare > best_score:
                    best_score = compare
                    if m == 1:
                        best_set_idx = [pool_idx[top_pos.item()]]
                    else:
                        best_set_idx = [pool_idx[j] for j in combos[top_pos.item()]]

        set_sims = sims[best_set_idx]
        rank_ord = torch.argsort(set_sims, descending=True).tolist()
        ranked   = [best_set_idx[i] for i in rank_ord]

        if return_predicted_set:
            if len(ranked) < 2:
                used = set(ranked)
                for idx in pool_idx:
                    if idx not in used:
                        ranked.append(idx)
                    if len(ranked) >= 2:
                        break
            return [self.func_names[i] for i in ranked]

        # Pad to k_return
        if len(ranked) < k_return:
            used = set(ranked)
            for idx in pool_idx:
                if idx not in used:
                    ranked.append(idx)
                if len(ranked) >= k_return:
                    break

        return [self.func_names[i] for i in ranked[:k_return]]

    def regularisation(self, lambda_D: float) -> torch.Tensor:
        """λ Σ_m ||M_m||_F^2."""
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in range(2, self.m_max + 1):
            reg = reg + lambda_D * self.D[str(m)].pow(2).sum()
        return reg


    def save(self, path: str, extra: dict = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Temporarily unwrap DataParallel so state_dict keys stay clean.
        enc = self.encoder_model
        if isinstance(enc, nn.DataParallel):
            self.encoder_model = enc.module
        payload = {
            "model_state":  self.state_dict(),
            "func_names":   self.func_names,
            "d":            self.d,
            "m_max":        self.m_max,
            "encoder_type": self.encoder_type,
            **(extra or {}),
        }
        torch.save(payload, path)
        if isinstance(enc, nn.DataParallel):
            self.encoder_model = enc  

    @classmethod
    def load(cls, path: str, encoder_path: str, device: str = "cpu") -> "HYSETModel":
        payload      = torch.load(path, map_location=device)
        func_names   = payload["func_names"]
        d            = payload["d"]
        m_max        = payload["m_max"]
        encoder_type = payload.get("encoder_type", "bert")
        model        = cls(encoder_path, func_names, d=d, m_max=m_max,
                           encoder_type=encoder_type)
        missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
        if missing:
            print(f"[HYSET] New params initialized with defaults: {missing}")
        if unexpected:
            print(f"[HYSET] Unexpected params in checkpoint (ignored): {unexpected}")
        model.to(device)
        return model
