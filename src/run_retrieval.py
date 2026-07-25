import argparse
import json
import os
import re
import sys
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "stabletoolbench"))


def standardize(string: str) -> str:
    res = re.compile(r"[^一-龥^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    string = string.strip("_")
    if string and string[0].isdigit():
        string = "get_" + string
    return string


def change_name(name: str) -> str:
    if name in {"from", "class", "return", "false", "true", "id", "and"}:
        return "is_" + name
    return name


def relevant_api_to_func(tool_name: str, api_name: str) -> str:
    return f"{change_name(standardize(api_name))}_for_{standardize(tool_name)}"


# ── Offline metrics ───────────────────────────────────────────────────────────

def compute_offline_metrics(retrieved: List[str], relevant_apis: List[List[str]]) -> dict:
    """
    retrieved      : list of func_names returned by retriever (top-K)
    relevant_apis  : list of [tool_name, api_name] pairs from query JSON

    Returns dict with:
      recall@K      : fraction of relevant APIs present in retrieved set
      precision@K   : fraction of retrieved set that are relevant
    """
    retrieved_set = set(retrieved)
    relevant_set  = {relevant_api_to_func(tn, an) for tn, an in relevant_apis}

    K     = len(retrieved)
    R     = len(relevant_set)
    hit   = len(relevant_set & retrieved_set)

    recall    = hit / R if R > 0 else 0.0
    precision = hit / K if K > 0 else 0.0
    return {"recall_at_k": recall, "precision_at_k": precision,
            "hits": hit, "n_relevant": R, "n_retrieved": K}


# ── Build modified query JSON ─────────────────────────────────────────────────

def build_query_with_retrieved_apis(query: dict, retrieved: List[str], corpus: dict) -> dict:
    """
    Replace query['api_list'] with the full API JSONs of the retrieved func_names.
    Keeps original query fields (query, relevant APIs, query_id) intact.
    """
    new_api_list = []
    for fname in retrieved:
        if fname in corpus:
            new_api_list.append(corpus[fname]["full_api_json"])
    new_query = dict(query)
    new_query["api_list"] = new_api_list
    return new_query


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["random", "bm25", "dense", "ied"],
                        help="Retrieval method")
    parser.add_argument("--label", required=True,
                        help="Experiment label, e.g. A1_bm25")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of APIs to retrieve")
    parser.add_argument("--corpus",
                        default=os.path.join(ROOT, "results", "end2end", "corpus.json"))
    parser.add_argument("--queries",
                        default=os.path.join(ROOT, "stabletoolbench", "solvable_queries",
                                             "test_instruction", "G1_instruction.json"))
    parser.add_argument("--out_dir",
                        default=os.path.join(ROOT, "results", "end2end"))
    # Dense / IED options
    parser.add_argument("--dense_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--ied_model",
                        default=os.path.join(ROOT, "results", "task2", "ied", "ied_model.pt"))
    parser.add_argument("--ied_device", default="cuda")
    parser.add_argument("--ied_candidate_k", type=int, default=50)
    args = parser.parse_args()

    # ── Load corpus ──────────────────────────────────────────────────────────
    print(f"Loading corpus from {args.corpus} …")
    with open(args.corpus) as f:
        corpus = json.load(f)
    print(f"  {len(corpus)} API functions in corpus.")

    # ── Load queries ─────────────────────────────────────────────────────────
    with open(args.queries) as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries.")

    # ── Build retriever ───────────────────────────────────────────────────────
    from retrievers import build_retriever
    label_dir  = os.path.join(args.out_dir, args.label)
    cache_path = os.path.join(label_dir, "dense_emb.pt")
    os.makedirs(label_dir, exist_ok=True)
    a2_cache = os.path.join(args.out_dir, "A2_dense", "dense_emb.pt")
    full_cache = a2_cache if os.path.exists(a2_cache) else os.path.join(label_dir, "dense_full_emb.pt")

    cfg = {
        "dense_model":      args.dense_model,
        "dense_cache":      cache_path,           
        "dense_full_cache": full_cache,           
        "ied_model":        args.ied_model,
        "ied_device":       args.ied_device,
        "ied_candidate_k":  args.ied_candidate_k,
        "ied_supplement":   True,
        "random_seed":      42,
    }
    retriever = build_retriever(args.method, corpus, cfg)
    print(f"Retriever ready: {args.method}")
    modified_queries = []
    per_query_metrics = {}

    for q in queries:
        qid   = str(q["query_id"])
        query = q["query"]
        rel   = q.get("relevant APIs", [])

        retrieved = retriever.retrieve(query, top_k=args.top_k)
        metrics   = compute_offline_metrics(retrieved, rel)
        per_query_metrics[qid] = {"retrieved": retrieved, **metrics}

        modified_q = build_query_with_retrieved_apis(q, retrieved, corpus)
        modified_queries.append(modified_q)

    # Aggregate
    recalls    = [v["recall_at_k"]    for v in per_query_metrics.values()]
    precisions = [v["precision_at_k"] for v in per_query_metrics.values()]
    agg = {
        "method":          args.method,
        "label":           args.label,
        "top_k":           args.top_k,
        "n_queries":       len(queries),
        "recall_at_k_mean":    sum(recalls)    / len(recalls),
        "precision_at_k_mean": sum(precisions) / len(precisions),
        "per_query":       per_query_metrics,
    }

    queries_path  = os.path.join(label_dir, "queries_retrieved.json")
    metrics_path  = os.path.join(label_dir, "offline_metrics.json")

    with open(queries_path, "w") as f:
        json.dump(modified_queries, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n[{args.label}] Offline metrics:")
    print(f"  Recall@{args.top_k}    = {agg['recall_at_k_mean']:.4f}")
    print(f"  Precision@{args.top_k} = {agg['precision_at_k_mean']:.4f}")
    print(f"Saved → {queries_path}")
    print(f"Saved → {metrics_path}")


if __name__ == "__main__":
    main()
