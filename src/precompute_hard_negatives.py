import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CORPUS_PATH   = os.path.join(ROOT, "data", "hyset_corpus.json")
HARD_NEG_PATH = os.path.join(ROOT, "data", "hard_neg_cache.json")
ENCODER_PATH  = os.path.join(ROOT, "models", "ToolBench_IR_bert_based_uncased")

INSTRUCTION_FILES = [
    os.path.join(ROOT, "data", "instruction", "G1_query.json"),
    os.path.join(ROOT, "data", "instruction", "G2_query.json"),
    os.path.join(ROOT, "data", "instruction", "G3_query.json"),
]
TEST_ID_DIR = os.path.join(ROOT, "data", "test_query_ids")


def load_test_ids() -> set:
    ids: set = set()
    for fname in os.listdir(TEST_ID_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(TEST_ID_DIR, fname)) as f:
                ids.update(str(x) for x in json.load(f))
    return ids


def func_name_of(tool: str, api: str) -> str:
    import re
    def std(s):
        s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
        s = re.sub(r"_+", "_", s).lower().strip("_")
        return ("get_" + s) if s and s[0].isdigit() else s
    def chg(n):
        return ("is_" + n) if n in {"from","class","return","false","true","id","and"} else n
    return f"{chg(std(api))}_for_{std(tool)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",      default=CORPUS_PATH)
    parser.add_argument("--out_path",    default=HARD_NEG_PATH)
    parser.add_argument("--encoder",     default=ENCODER_PATH)
    parser.add_argument("--model_ckpt",  default=None,
                        help="HYSET checkpoint to use U embeddings instead of encoder")
    parser.add_argument("--top_k",       type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Loading corpus …")
    with open(args.corpus) as f:
        corpus = json.load(f)
    func_names = list(corpus.keys())
    fn_to_idx  = {fn: i for i, fn in enumerate(func_names)}
    N_T = len(func_names)
    print(f"  {N_T} API endpoints.")

    # Load encoder once (used for both API and query embeddings when no checkpoint)
    from transformers import AutoModel, AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(args.encoder)
    _enc = AutoModel.from_pretrained(args.encoder).to(args.device).eval()

    def _mean_pool(texts_list, verbose=False):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts_list), args.batch_size):
                batch = texts_list[i: i + args.batch_size]
                inp = _tok(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(args.device)
                out = _enc(**inp)
                mask = inp["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                all_embs.append(emb)
                if verbose and i % (args.batch_size * 20) == 0:
                    print(f"  encoded {i + len(batch)}/{len(texts_list)}")
        return torch.cat(all_embs, dim=0)

    def encode_queries(texts_batch):
        return F.normalize(_mean_pool(texts_batch, verbose=True), dim=-1)

    # Build API embedding matrix
    if args.model_ckpt and os.path.exists(args.model_ckpt):
        # Use trained U from HYSET checkpoint
        print(f"Loading API embeddings from checkpoint: {args.model_ckpt}")
        ckpt = torch.load(args.model_ckpt, map_location="cpu")
        U = ckpt["model_state"]["U.weight"].float().numpy()  # (N_T, d)
        print(f"  U shape: {U.shape}")
    else:
        # Use ToolBench_IR encoder to build API embeddings
        print(f"Encoding API corpus with {args.encoder} …")
        texts = [corpus[fn]["text"] for fn in func_names]
        U = _mean_pool(texts, verbose=True).cpu().float().numpy()
        print(f"  U shape: {U.shape}")

    # Normalize for cosine similarity
    norms = np.linalg.norm(U, axis=1, keepdims=True).clip(min=1e-9)
    U_norm = U / norms
    U_tensor = torch.from_numpy(U_norm).to(args.device)  # (N_T, d)

    # Load training queries
    test_ids = load_test_ids()
    print(f"Test IDs to exclude: {len(test_ids)}")

    hard_neg_cache = {}

    for inst_file in INSTRUCTION_FILES:
        print(f"\nProcessing {os.path.basename(inst_file)} …")
        with open(inst_file) as f:
            queries = json.load(f)

        # Batch-encode queries
        train_queries = [q for q in queries if str(q["query_id"]) not in test_ids]
        print(f"  {len(train_queries)} training queries")

        query_texts = [" ".join(q["query"]) if isinstance(q["query"], list) else q["query"] for q in train_queries]
        q_embs = encode_queries(query_texts).to(args.device)

        # Batch similarity: (n_queries, N_T)
        # Process in chunks to avoid OOM
        chunk = 512
        for start in range(0, len(train_queries), chunk):
            end = min(start + chunk, len(train_queries))
            q_chunk = q_embs[start:end]                    # (chunk, d)
            sims    = torch.mm(q_chunk, U_tensor.T)        # (chunk, N_T)

            for local_i, q_item in enumerate(train_queries[start:end]):
                qid = str(q_item["query_id"])
                gt_fns = {func_name_of(e[0], e[1])
                          for e in q_item.get("relevant APIs", []) if len(e) >= 2 and e[0] and e[1]}

                sim_row = sims[local_i]
                # Zero out GT API scores so they don't appear as hard negatives
                for fn in gt_fns:
                    idx = fn_to_idx.get(fn)
                    if idx is not None:
                        sim_row[idx] = -1.0

                top_k_idx = torch.topk(sim_row, k=min(args.top_k, N_T)).indices.tolist()
                hard_neg_cache[qid] = [func_names[i] for i in top_k_idx]

            if start % (chunk * 10) == 0:
                with open(args.out_path, "w") as f:
                    json.dump(hard_neg_cache, f)

    with open(args.out_path, "w") as f:
        json.dump(hard_neg_cache, f, indent=2)
    print(f"\nSaved {len(hard_neg_cache)} entries → {args.out_path}")


if __name__ == "__main__":
    main()
