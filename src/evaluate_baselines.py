import argparse
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

SPLITS = {
    "I1-Inst": ("G1_query.json", "G1_instruction_test_query_ids.json"),
    "I1-Tool": ("G1_query.json", "G1_tool_test_query_ids.json"),
    "I1-Cate": ("G1_query.json", "G1_category_test_query_ids.json"),
    "I2-Inst": ("G2_query.json", "G2_instruction_test_query_ids.json"),
    "I2-Cate": ("G2_query.json", "G2_category_test_query_ids.json"),
    "I3-Inst": ("G3_query.json", "G3_instruction_test_query_ids.json"),
}
def _std(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).lower().strip("_")
    return ("get_" + s) if s and s[0].isdigit() else s


def _chname(n: str) -> str:
    return ("is_" + n) if n in {"from", "class", "return", "false", "true", "id", "and"} else n


def func_name_of(tool: str, api: str) -> str:
    return f"{_chname(_std(api))}_for_{_std(tool)}"

def recall_at_k(retrieved: List[str], gt: List[str], k: int) -> float:
    if not gt:
        return 0.0
    return len(set(gt) & set(retrieved[:k])) / len(gt)


def ndcg_at_k(retrieved: List[str], gt: List[str], k: int) -> float:
    gt_set = set(gt)
    dcg  = sum(1.0 / math.log2(r + 2) for r, fn in enumerate(retrieved[:k]) if fn in gt_set)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(k, len(gt_set))))
    return dcg / idcg if idcg > 0 else 0.0


def comp_at_k(retrieved: List[str], gt: List[str], k: int) -> float:
    return 1.0 if set(gt) <= set(retrieved[:k]) else 0.0


def compute_split_metrics(
    retrieved_list: List[List[str]],
    gt_list: List[List[str]],
) -> dict:
    results: dict = {}
    for k in (3, 5):
        r_vals = [recall_at_k(r, g, k) for r, g in zip(retrieved_list, gt_list)]
        n_vals = [ndcg_at_k(r, g, k)   for r, g in zip(retrieved_list, gt_list)]
        c_vals = [comp_at_k(r, g, k)   for r, g in zip(retrieved_list, gt_list)]
        results[f"Recall@{k}"] = sum(r_vals) / len(r_vals)
        results[f"NDCG@{k}"]   = sum(n_vals) / len(n_vals)
        results[f"COMP@{k}"]   = sum(c_vals) / len(c_vals)
    return results

def load_split(query_file: str, id_file: str) -> List[dict]:
    with open(query_file) as f:
        all_queries = json.load(f)
    with open(id_file) as f:
        test_ids = {str(x) for x in json.load(f)}
    return [q for q in all_queries if str(q.get("query_id", "")) in test_ids]

class RandomRetriever:
    def __init__(self, fn_keys: List[str], seed: int = 42):
        import random
        self._keys = fn_keys
        self._rng  = random.Random(seed)

    def retrieve(self, query: str, k: int) -> List[str]:
        shuffled = list(self._keys)
        self._rng.shuffle(shuffled)
        return shuffled[:k]


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()


class BM25Retriever:

    def __init__(self, fn_keys: List[str], docs: List[str], k1: float = 1.5, b: float = 0.75):
        import math
        from collections import Counter

        self._fn_keys = fn_keys
        self._k1 = k1
        self._b  = b

        tokenized = [_tokenize(d) for d in docs]
        self._dl = [len(t) for t in tokenized]
        self._avgdl = sum(self._dl) / max(len(self._dl), 1)

        N = len(docs)
        term_docs: Dict[str, dict] = {}
        for i, toks in enumerate(tokenized):
            for tok, cnt in Counter(toks).items():
                if tok not in term_docs:
                    term_docs[tok] = {}
                term_docs[tok][i] = cnt

        # Precompute IDF per term
        self._idf: Dict[str, float] = {}
        for tok, posting in term_docs.items():
            df = len(posting)
            self._idf[tok] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        self._term_docs = term_docs

    def retrieve(self, query: str, k: int) -> List[str]:
        q_toks = _tokenize(query)
        scores = [0.0] * len(self._fn_keys)
        for tok in q_toks:
            if tok not in self._term_docs:
                continue
            idf = self._idf[tok]
            for i, freq in self._term_docs[tok].items():
                dl    = self._dl[i]
                tf    = freq * (self._k1 + 1) / (freq + self._k1 * (1 - self._b + self._b * dl / self._avgdl))
                scores[i] += idf * tf

        ranked = sorted(range(len(self._fn_keys)), key=lambda i: -scores[i])
        return [self._fn_keys[i] for i in ranked[:k]]

class DenseRetriever:

    def __init__(
        self,
        fn_keys: List[str],
        docs: List[str],
        model_path: str,
        device: str = "cpu",
        batch_size: int = 256,
        cache_path: str = "",
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"  [dense] WARNING: CUDA requested but not available — falling back to CPU.")
            device = "cpu"
        self._fn_keys = fn_keys
        self._device  = device

        print(f"  [dense] loading encoder from {model_path} (device={device}) …")
        self._tok   = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path).to(device)
        self._model.eval()

        if cache_path and os.path.exists(cache_path):
            print(f"  [dense] loading cached corpus embeddings from {cache_path} …")
            self._corpus_emb = torch.load(cache_path, map_location=device)
        else:
            print(f"  [dense] encoding {len(docs)} corpus docs …")
            self._corpus_emb = self._encode_batch(docs, batch_size)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                torch.save(self._corpus_emb, cache_path)
                print(f"  [dense] saved corpus embeddings → {cache_path}")

    def _encode_batch(self, texts: List[str], batch_size: int):
        import torch
        import torch.nn.functional as F

        all_emb = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            enc   = self._tok(batch, padding=True, truncation=True,
                              max_length=128, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model(**enc).last_hidden_state
                attn_mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out * attn_mask).sum(1) / attn_mask.sum(1).clamp(min=1e-9)
                emb = F.normalize(emb, dim=-1)
            all_emb.append(emb.cpu())
            if (start // batch_size + 1) % 20 == 0:
                print(f"    encoded {start + len(batch)}/{len(texts)} …")
        return torch.cat(all_emb, dim=0)

    def retrieve(self, query: str, k: int) -> List[str]:
        import torch
        import torch.nn.functional as F

        enc = self._tok([query], padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out  = self._model(**enc).last_hidden_state
            attn = enc["attention_mask"].unsqueeze(-1).float()
            q_emb = (out * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
            q_emb = F.normalize(q_emb, dim=-1)

        sims   = (self._corpus_emb.to(self._device) @ q_emb.T).squeeze(-1)
        ranked = sims.topk(k).indices.tolist()
        return [self._fn_keys[i] for i in ranked]


# ── ToolGen generative retriever ─────────────────────────────────────────────

class ToolGenRetriever:

    def __init__(
        self,
        model_path: str,
        corpus_fn_set: set,
        device: str = "cpu",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("  [toolgen] WARNING: CUDA not available — falling back to CPU.")
            device = "cpu"
        self._device = device

        print(f"  [toolgen] loading tokenizer from {model_path} …")
        self._tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"  [toolgen] loading model from {model_path} (device={device}) …")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()
        print("  [toolgen] building virtual-token → func_name mapping …")
        added_vocab = self._tok.get_added_vocab()  # {token_str: token_id}
        self._candidate_ids:   List[int] = []
        self._candidate_fns:   List[str] = []

        for tok_str, tok_id in added_vocab.items():
            if not (tok_str.startswith("<<") and "&&" in tok_str and tok_str.endswith(">>")):
                continue
            fn = _vtoken_to_fn(tok_str)
            if fn in corpus_fn_set:
                self._candidate_ids.append(tok_id)
                self._candidate_fns.append(fn)

        import torch
        self._candidate_ids_t = torch.tensor(self._candidate_ids, dtype=torch.long)
        print(f"  [toolgen] {len(self._candidate_fns)} tool tokens matched corpus "
              f"(out of {sum(1 for t in added_vocab if '&&' in t)} total virtual tokens)")

    def retrieve(self, query: str, k: int) -> List[str]:
        import torch

        messages = [{"role": "user", "content": query}]
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self._tok(prompt, return_tensors="pt", add_special_tokens=False).to(self._device)

        with torch.no_grad():
            logits = self._model(**enc).logits[0, -1]  # [vocab_size]

        candidate_scores = logits[self._candidate_ids_t.to(self._device)]
        topk_local = candidate_scores.topk(min(k, len(self._candidate_fns))).indices.tolist()
        return [self._candidate_fns[i] for i in topk_local]


def _vtoken_to_fn(vtoken: str) -> str:
    """Convert '<<ServiceName&&OperationName>>' → 'operation_for_service' func_name."""
    inner = vtoken[2:-2]  # strip << and >>
    parts = inner.split("&&", 1)
    if len(parts) != 2:
        return vtoken
    tool_name, api_name = parts
    fn = _chname(_std(api_name)) + "_for_" + _std(tool_name)
    return fn[-64:]  

def main():
    parser = argparse.ArgumentParser(description="Offline baseline evaluation on 6 ToolBench splits")

    parser.add_argument("--method", required=True,
                        choices=["random", "bm25", "dense", "toolgen"],
                        help="Retrieval method")
    parser.add_argument("--label",  required=True,
                        help="Experiment label (used as result directory name)")
    parser.add_argument("--encoder",
                        default=os.path.join(ROOT, "models", "ToolBench_IR_bert_based_uncased"),
                        help="Path to HF encoder model (dense only)")
    parser.add_argument("--corpus",
                        default=os.path.join(ROOT, "data", "hyset_corpus.json"))
    parser.add_argument("--result_dir",
                        default=os.path.join(ROOT, "result"))
    parser.add_argument("--k_pool",  type=int, default=50,
                        help="Number of top-k to retrieve (must be >= 5)")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for dense encoding")
    parser.add_argument("--bm25_k1",  type=float, default=1.5)
    parser.add_argument("--bm25_b",   type=float, default=0.75)
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (random baseline)")
    parser.add_argument("--cache_dir",
                        default=os.path.join(ROOT, "data", "cache"),
                        help="Directory for dense embedding caches")
    parser.add_argument("--toolgen_model",
                        default=os.path.join(ROOT, "models", "ToolGen-Qwen2.5-1.5B-Tool-Retriever"),
                        help="Path to ToolGen model (toolgen method only)")
    args = parser.parse_args()

    assert args.k_pool >= 5, "--k_pool must be >= 5"
    print(f"Loading corpus from {args.corpus} …")
    with open(args.corpus) as f:
        corpus: dict = json.load(f)
    fn_keys: List[str] = list(corpus.keys())
    fn_set  = set(fn_keys)
    print(f"  {len(fn_keys)} APIs in corpus.")

    def _api_text(entry: dict) -> str:
        aj   = entry.get("full_api_json", {})
        desc = aj.get("api_description", "") or ""
        return f"{aj.get('category_name','')} {aj.get('tool_name','')} {aj.get('api_name','')} {desc}"

    docs = [_api_text(corpus[k]) for k in fn_keys]
    print(f"Building {args.method} retriever …")
    t0 = time.time()

    if args.method == "random":
        retriever = RandomRetriever(fn_keys, seed=args.seed)

    elif args.method == "bm25":
        retriever = BM25Retriever(fn_keys, docs, k1=args.bm25_k1, b=args.bm25_b)

    elif args.method == "dense":
        os.makedirs(args.cache_dir, exist_ok=True)
        safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "_", args.label)
        cache_path = os.path.join(args.cache_dir, f"{safe_label}_corpus_emb.pt")
        retriever  = DenseRetriever(
            fn_keys    = fn_keys,
            docs       = docs,
            model_path = args.encoder,
            device     = args.device,
            batch_size = args.batch_size,
            cache_path = cache_path,
        )

    elif args.method == "toolgen":
        retriever = ToolGenRetriever(
            model_path    = args.toolgen_model,
            corpus_fn_set = fn_set,
            device        = args.device,
        )

    print(f"  Retriever ready in {time.time()-t0:.1f}s")

    label_dir = os.path.join(args.result_dir, args.label)
    os.makedirs(label_dir, exist_ok=True)

    config = {
        "label":         args.label,
        "method":        args.method,
        "encoder":       args.encoder       if args.method == "dense"    else None,
        "toolgen_model": args.toolgen_model if args.method == "toolgen"  else None,
        "corpus":        args.corpus,
        "k_pool":        args.k_pool,
        "bm25_k1":       args.bm25_k1       if args.method == "bm25"     else None,
        "bm25_b":        args.bm25_b         if args.method == "bm25"     else None,
        "seed":          args.seed           if args.method == "random"   else None,
    }
    with open(os.path.join(label_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    INST_DIR = os.path.join(ROOT, "data", "instruction")
    ID_DIR   = os.path.join(ROOT, "data", "test_query_ids")

    all_split_metrics: Dict[str, dict] = {}

    for split_name, (qfile, idfile) in SPLITS.items():
        split_dir = os.path.join(label_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"  Evaluating {split_name} …")

        queries = load_split(
            os.path.join(INST_DIR, qfile),
            os.path.join(ID_DIR,   idfile),
        )
        print(f"  {len(queries)} test queries.")

        retrieved_list: List[List[str]] = []
        gt_list:        List[List[str]] = []

        for q in queries:
            # Ground-truth: filter to APIs that exist in the corpus
            gt_fns = [
                func_name_of(e[0], e[1])
                for e in q.get("relevant APIs", [])
                if len(e) >= 2 and e[0] and e[1]
            ]
            gt_fns = [fn for fn in gt_fns if fn in fn_set]
            gt_list.append(gt_fns)

            raw_q = q["query"]
            q_str = " ".join(raw_q) if isinstance(raw_q, list) else raw_q

            ret = retriever.retrieve(q_str, k=args.k_pool)
            retrieved_list.append(ret)

        metrics = compute_split_metrics(retrieved_list, gt_list)
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        split_summary = {"split": split_name, "n_queries": len(queries), **metrics}
        with open(os.path.join(split_dir, "summary.json"), "w") as f:
            json.dump(split_summary, f, indent=2)

        all_split_metrics[split_name] = split_summary

    overall_dir = os.path.join(label_dir, "overall")
    os.makedirs(overall_dir, exist_ok=True)

    offline_keys = [k for k in list(all_split_metrics.values())[0]
                    if k not in ("split", "n_queries")]
    overall: dict = {"label": args.label, "method": args.method, "splits": list(SPLITS.keys())}
    for k in offline_keys:
        vals = [v[k] for v in all_split_metrics.values() if k in v]
        overall[k] = sum(vals) / len(vals) if vals else 0.0

    with open(os.path.join(overall_dir, "summary.json"), "w") as f:
        json.dump(overall, f, indent=2)
    metrics_cols = ["Recall@3", "Recall@5", "NDCG@3", "NDCG@5", "COMP@3", "COMP@5"]
    print(f"\n{'=' * 80}")
    print(f"  {args.label} ({args.method})")
    print(f"{'=' * 80}")
    print(f"  {'Split':<12}" + "".join(f"{m:>12}" for m in metrics_cols))
    print("  " + "-" * (12 + 12 * len(metrics_cols)))
    for sp in SPLITS:
        d = all_split_metrics.get(sp, {})
        row = f"  {sp:<12}"
        for m in metrics_cols:
            v = d.get(m)
            row += f"{v:>12.4f}" if isinstance(v, float) else f"{'—':>12}"
        print(row)
    print("  " + "-" * (12 + 12 * len(metrics_cols)))
    row = f"  {'Overall':<12}"
    for m in metrics_cols:
        v = overall.get(m)
        row += f"{v:>12.4f}" if isinstance(v, float) else f"{'—':>12}"
    print(row)
    print(f"\nAll results saved under: {label_dir}")


if __name__ == "__main__":
    main()
