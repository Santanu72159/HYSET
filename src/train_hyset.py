import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from hyset_model import HYSETModel, func_name_of


def query_hash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


def make_cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_test_ids(test_id_dir: str) -> set:
    ids: set = set()
    for fname in os.listdir(test_id_dir):
        if fname.endswith(".json"):
            with open(os.path.join(test_id_dir, fname)) as f:
                ids.update(str(x) for x in json.load(f))
    return ids


def stratified_val_split(
    items: List[dict],
    target_val: int,
    rng: random.Random,
    m_max: int,
) -> Tuple[List[dict], List[dict]]:
    def bucket(n: int) -> str:
        return str(n) if n < m_max else f">={m_max}"

    # Group items by stratum, recording indices into `items`.
    strata: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        strata[(it["gx"], bucket(len(it["gt"])))].append(idx)

    total = len(items)
    val_indices: List[int] = []

    quotas: Dict[Tuple[str, str], int] = {}
    for key, idxs in strata.items():
        q = int(round(target_val * len(idxs) / total))
        q = max(0, min(q, len(idxs)))
        quotas[key] = q

    cur = sum(quotas.values())
    keys_by_size = sorted(strata.keys(), key=lambda k: -len(strata[k]))
    while cur < target_val:
        progressed = False
        for key in keys_by_size:
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                cur += 1
                progressed = True
                if cur == target_val:
                    break
        if not progressed: 
            break
    while cur > target_val:
        for key in reversed(keys_by_size):
            if quotas[key] > 0:
                quotas[key] -= 1
                cur -= 1
                if cur == target_val:
                    break
    for key, idxs in strata.items():
        q = quotas[key]
        if q == 0:
            continue
        chosen = rng.sample(idxs, q)
        val_indices.extend(chosen)

    val_set = set(val_indices)
    val_items   = [items[i] for i in val_indices]
    train_items = [items[i] for i in range(len(items)) if i not in val_set]

    print(f"[stratified val] target={target_val}  realized={len(val_items)}  "
          f"strata={len(strata)}")
    for key in sorted(strata.keys()):
        gx, mb = key
        print(f"  {gx} m={mb:<6}  pool={len(strata[key]):>6}  "
              f"val={quotas[key]:>4}")
    return val_items, train_items


# ── dataset ───────────────────────────────────────────────────────────────────

class ToolSetDataset(Dataset):
    def __init__(self, instruction_files: List[Tuple[str, str]], test_ids: set,
                 fn_to_idx: dict):
        self.items: List[dict] = []
        for gx_tag, path in instruction_files:
            with open(path) as f:
                queries = json.load(f)
            for q in queries:
                qid = str(q.get("query_id", ""))
                if qid in test_ids:
                    continue
                gt_fns = [
                    func_name_of(e[0], e[1])
                    for e in q.get("relevant APIs", [])
                    if len(e) >= 2 and e[0] and e[1]
                ]
                gt_fns = [fn for fn in gt_fns if fn in fn_to_idx]
                if not gt_fns:
                    continue
                raw_q = q["query"]
                query_str = " ".join(raw_q) if isinstance(raw_q, list) else raw_q
                self.items.append({
                    "query":  query_str,
                    "gt":     list(dict.fromkeys(gt_fns)),  # deduplicated
                    "qid":    qid,
                    "qhash":  query_hash(query_str),
                    "gx":     gx_tag,
                })
        print(f"[Dataset] {len(self.items)} training instances loaded.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class NegativeSampler:
    def __init__(
        self,
        func_names: List[str],
        fn_to_idx: Dict[str, int],
        hard_neg_cache: Dict[str, List[str]],
        B: int = 10,
        rng_seed: int = 42,
        m_max: int = 4,
    ):
        self.func_names     = func_names
        self.fn_to_idx      = fn_to_idx
        self.hard_neg_cache = hard_neg_cache
        self.B              = B
        self.rng            = random.Random(rng_seed)
        self.N_T            = len(func_names)
        self.m_max          = m_max

    def _random_size_m(self, m: int, exclude: set) -> Optional[List[int]]:
        candidates = [i for i in range(self.N_T) if i not in exclude]
        if len(candidates) < m:
            return None
        return self.rng.sample(candidates, m)

    def _hard_neg_indices(self, qid: str, m: int, exclude: set) -> Optional[List[int]]:
        pool = [self.fn_to_idx[fn]
                for fn in self.hard_neg_cache.get(qid, [])
                if fn in self.fn_to_idx and self.fn_to_idx[fn] not in exclude]
        if len(pool) < m:
            return None
        return self.rng.sample(pool, m)

    def build_candidates(
        self,
        batch: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        by_m: Dict[int, List[int]] = defaultdict(list)
        for i, item in enumerate(batch):
            m = min(len(item["gt"]), self.m_max)
            by_m[m].append(i)

        results: List[Optional[Tuple]] = [None] * len(batch)

        for m, idxs in by_m.items():
            # Candidate GT indices for in-batch pool (same size m, capped)
            ib_pool: List[List[int]] = [
                [self.fn_to_idx[fn] for fn in batch[i]["gt"]][:m]
                for i in idxs
            ]

            for pos, i in enumerate(idxs):
                item       = batch[i]
                gt_idx     = [self.fn_to_idx[fn] for fn in item["gt"]][:m]  # cap to m_max
                gt_set     = set(gt_idx)
                n_neg      = self.B - 1
                n_rand  = max(1, int(round(n_neg * 0.50)))
                n_ib    = max(0, int(round(n_neg * 0.30)))
                n_hard  = n_neg - n_rand - n_ib

                negs: List[List[int]] = []

                for _ in range(n_rand * 3): 
                    s = self._random_size_m(m, gt_set)
                    if s is not None:
                        negs.append(s)
                    if len(negs) >= n_rand:
                        break

                ib_added = 0
                for j, other_gt in enumerate(ib_pool):
                    if j == pos:
                        continue
                    if ib_added >= n_ib:
                        break
                    if set(other_gt) & gt_set:
                        continue
                    negs.append(list(other_gt))
                    ib_added += 1
                for _ in range(n_hard):
                    s = self._hard_neg_indices(item["qid"], m, gt_set)
                    if s is not None:
                        negs.append(s)
                while len(negs) < n_neg:
                    s = self._random_size_m(m, gt_set)
                    if s is not None:
                        negs.append(s)

                negs = negs[:n_neg]

                gt_t  = torch.tensor([gt_idx], dtype=torch.long)       # (1, m)
                neg_t = torch.tensor(negs,     dtype=torch.long)       # (n_neg, m)
                results[i] = (gt_t, neg_t)

        return results

def train_step(
    model:     HYSETModel,
    batch:     List[dict],
    candidates: List[Tuple],
    reward_cache: Dict[str, dict],
    eta:       float,
    lambda_D:  float,
    device:    str,
) -> Tuple[torch.Tensor, dict]:
    queries = [item["query"] for item in batch]
    q_vecs  = model.encode_queries(queries)    # (batch, d_q)

    l_tool_total  = torch.tensor(0.0, device=device)
    l_agent_total = torch.tensor(0.0, device=device)
    n_valid       = 0

    tau = model.log_tau.exp().clamp(min=0.01, max=1.0)

    for i, (item, cand) in enumerate(zip(batch, candidates)):
        if cand is None:
            continue
        gt_t, neg_t = cand                     
        m = gt_t.shape[1]

        gt_t  = gt_t.to(device)
        neg_t = neg_t.to(device)

        all_cand = torch.cat([gt_t, neg_t], dim=0)  
        q         = q_vecs[i]                          

        scores = model.score_sets(q, all_cand)        
        log_prob  = F.log_softmax(scores / tau, dim=0) 
        l_tool    = -log_prob[0]
        l_tool_total = l_tool_total + l_tool

        if eta > 0:
            qhash     = item["qhash"]
            gt_apis   = set(item["gt"])

            with torch.no_grad():
                t_hat_idx = scores.argmax().item()
            t_hat_apis = set(model.func_names[j] for j in all_cand[t_hat_idx].tolist())
            recall_proxy = len(t_hat_apis & gt_apis) / max(len(gt_apis), 1)

            if qhash in reward_cache:
                cached_score = float(reward_cache[qhash].get("score", 0.0))
                R_i = cached_score * recall_proxy
            else:
                R_i = recall_proxy

            if R_i > 0:
                log_p_that  = F.log_softmax(scores / tau, dim=0)[t_hat_idx]
                l_agent_total = l_agent_total + (-R_i * log_p_that)

        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, requires_grad=True, device=device), {}

    l_tool_mean  = l_tool_total  / n_valid
    l_agent_mean = l_agent_total / n_valid
    reg          = model.regularisation(lambda_D)

    loss = l_tool_mean + eta * l_agent_mean + reg

    return loss, {
        "l_tool":  l_tool_mean.item(),
        "l_agent": l_agent_mean.item(),
        "reg":     reg.item(),
        "loss":    loss.item(),
    }


@torch.no_grad()
def validate(model: HYSETModel, val_items: List[dict], k: int = 5,
             device: str = "cuda") -> float:
    """Quick Recall@K on a small validation subset."""
    model.eval()
    recalls = []
    for item in val_items:
        retrieved = model.retrieve(item["query"], k_return=k)
        gt_set    = set(item["gt"])
        hit       = len(gt_set & set(retrieved))
        recalls.append(hit / max(len(gt_set), 1))
    model.train()
    return float(sum(recalls) / len(recalls)) if recalls else 0.0
def main():
    parser = argparse.ArgumentParser(description="Train HYSET model")

    # paths
    parser.add_argument("--corpus",       default=os.path.join(ROOT, "data", "hyset_corpus.json"))
    parser.add_argument("--reward_cache", default=os.path.join(ROOT, "data", "reward_cache.json"))
    parser.add_argument("--hard_neg",     default=os.path.join(ROOT, "data", "hard_neg_cache.json"))
    parser.add_argument("--ckpt_dir",     default=os.path.join(ROOT, "checkpoints", "hyset"))
    parser.add_argument("--encoder",      default=os.path.join(ROOT, "models", "ToolBench_IR_bert_based_uncased"))
    parser.add_argument("--resume",       default=None, help="Resume from checkpoint path")
    parser.add_argument("--d",            type=int,   default=128,  help="API embedding dim")
    parser.add_argument("--m_max",        type=int,   default=4,    help="Max set order")
    parser.add_argument("--B",            type=int,   default=10,   help="Candidate set size")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4, help="HYSET-specific params LR")
    parser.add_argument("--lr_encoder",   type=float, default=1e-5, help="BERT fine-tuning LR")
    parser.add_argument("--eta",          type=float, default=0.1,  help="Agent loss weight")
    parser.add_argument("--lambda_beta",  type=float, default=0.0,  help="Unused; kept for CLI backward compat")
    parser.add_argument("--lambda_D",     type=float, default=0.01)
    parser.add_argument("--k_pool",       type=int,   default=50,   help="Dense candidate pool size")
    parser.add_argument("--val_size",     type=int,   default=2000,
                        help="Stratified validation subset size (target)")
    parser.add_argument("--refresh_hard_neg", action="store_true",
                        help="Refresh hard negatives after each epoch")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Fraction of steps for LR warmup")
    parser.add_argument("--patience",     type=int,   default=3,    help="Early-stop patience (epochs)")
    parser.add_argument("--val_freq",     type=int,   default=1,    help="Validate every N epochs")
    parser.add_argument("--encoder_type", default="bert", choices=["bert", "qwen"],
                        help="Query encoder backbone: bert (default) or qwen")
    parser.add_argument("--toolgen_init", action="store_true",
                        help="Initialise U from ToolGen virtual-token embeddings before training")
    parser.add_argument("--toolgen_path", default=os.path.join(ROOT, "models", "ToolGen-Qwen2.5-1.5B-Tool-Retriever"),
                        help="ToolGen model path (used with --toolgen_init)")
    parser.add_argument("--n_base_vocab", type=int, default=151_665,
                        help="ToolGen base vocab size (virtual tokens start here)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Loading corpus …")
    with open(args.corpus) as f:
        corpus = json.load(f)
    func_names = list(corpus.keys())
    fn_to_idx  = {fn: i for i, fn in enumerate(func_names)}
    api_texts  = [corpus[fn]["text"] for fn in func_names]
    print(f"  {len(func_names)} API endpoints.")

    reward_cache: dict = {}
    if os.path.exists(args.reward_cache):
        with open(args.reward_cache) as f:
            reward_cache = json.load(f)
        print(f"Reward cache: {len(reward_cache)} entries.")
    else:
        print("[WARN] No reward cache found – L_agent will be zero (eta effectively 0).")

    hard_neg_cache: dict = {}
    if os.path.exists(args.hard_neg):
        with open(args.hard_neg) as f:
            hard_neg_cache = json.load(f)
        print(f"Hard neg cache: {len(hard_neg_cache)} entries.")
    else:
        print("[WARN] No hard neg cache – hard negatives will fall back to random.")

    test_ids = load_test_ids(os.path.join(ROOT, "data", "test_query_ids"))
    inst_files = [
        ("G1", os.path.join(ROOT, "data", "instruction", "G1_query.json")),
        ("G2", os.path.join(ROOT, "data", "instruction", "G2_query.json")),
        ("G3", os.path.join(ROOT, "data", "instruction", "G3_query.json")),
    ]
    dataset = ToolSetDataset(inst_files, test_ids, fn_to_idx)

    rng = random.Random(args.seed)
    val_items, train_items = stratified_val_split(
        dataset.items, target_val=args.val_size, rng=rng, m_max=args.m_max,
    )
    # Shuffle the residual train set deterministically.
    rng.shuffle(train_items)
    print(f"Train: {len(train_items)}  Val: {len(val_items)}")

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume} …")
        model = HYSETModel.load(args.resume, args.encoder, device=args.device)
        payload = torch.load(args.resume, map_location="cpu")
        start_epoch = payload.get("epoch", 0)
    else:
        print("Initialising HYSET model …")
        model = HYSETModel(args.encoder, func_names, d=args.d, m_max=args.m_max,
                          encoder_type=args.encoder_type)
        model.to(args.device)
        if args.toolgen_init:
            print(f"Initialising U from ToolGen: {args.toolgen_path}")
            n_hit = model.init_api_embeddings_from_toolgen(
                args.toolgen_path, corpus,
                n_base_vocab=args.n_base_vocab,
                api_texts_fallback=api_texts,   # frozen g_psi fallback for unmatched
            )
            if n_hit == 0:
                print("[WARN] ToolGen init failed; falling back to encoder init.")
                model.init_api_embeddings(api_texts)
        else:
            model.init_api_embeddings(api_texts)

    if args.lr_encoder == 0:
        model.freeze_encoder()
        print(f"[HYSET] Encoder frozen (lr_encoder=0): no_grad on forward, ~2x memory saving")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1 and "cuda" in args.device:
        model.encoder_model = nn.DataParallel(model.encoder_model)
        print(f"[HYSET] DataParallel: using {n_gpus} GPUs for BERT encoder")
    else:
        print(f"[HYSET] Single device: {args.device}")
    enc_module = (model.encoder_model.module
                  if isinstance(model.encoder_model, nn.DataParallel)
                  else model.encoder_model)
    encoder_params = list(enc_module.parameters())
    hyset_params    = (
        list(model.U.parameters())
        + list(model.W.parameters())
        + list(model.D.parameters())
        + [model.log_tau]
    )

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.lr_encoder},
        {"params": hyset_params,    "lr": args.lr},
    ])

    steps_per_epoch = max(1, len(train_items) // args.batch_size)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))
    scheduler = make_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)
    print(f"[HYSET] Scheduler: {warmup_steps} warmup steps / {total_steps} total steps")

    sampler = NegativeSampler(func_names, fn_to_idx, hard_neg_cache, B=args.B, m_max=args.m_max)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_recall     = -1.0
    patience_count  = 0
    log_path        = os.path.join(args.ckpt_dir, "train_log.jsonl")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        random.shuffle(train_items)

        epoch_loss = 0.0
        epoch_l_tool = epoch_l_agent = 0.0
        n_steps = 0

        for step_start in range(0, len(train_items), args.batch_size):
            batch = train_items[step_start: step_start + args.batch_size]
            if not batch:
                continue

            candidates = sampler.build_candidates(batch)

            optimizer.zero_grad()
            loss, metrics = train_step(
                model, batch, candidates, reward_cache,
                args.eta, args.lambda_D, args.device
            )

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            epoch_loss    += metrics.get("loss", 0.0)
            epoch_l_tool  += metrics.get("l_tool", 0.0)
            epoch_l_agent += metrics.get("l_agent", 0.0)
            n_steps       += 1

            if n_steps % 200 == 0:
                cur_lr = scheduler.get_last_lr()[-1]
                print(
                    f"  Epoch {epoch+1} step {n_steps} | "
                    f"loss={metrics.get('loss',0):.4f}  "
                    f"l_tool={metrics.get('l_tool',0):.4f}  "
                    f"l_agent={metrics.get('l_agent',0):.4f}  "
                    f"lr={cur_lr:.2e}",
                    flush=True,
                )

        avg = lambda x: x / max(n_steps, 1)
        cur_lr = scheduler.get_last_lr()[0]

        do_val = ((epoch + 1) % args.val_freq == 0) or (epoch + 1 == args.epochs)
        recall_val = validate(model, val_items, k=5, device=args.device) if do_val else float("nan")

        log_entry = {
            "epoch":     epoch + 1,
            "loss":      avg(epoch_loss),
            "l_tool":    avg(epoch_l_tool),
            "l_agent":   avg(epoch_l_agent),
            "recall@5":  recall_val,
            "lr":        cur_lr,
        }
        print(f"\n[Epoch {epoch+1}] " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in log_entry.items()
        ))

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1}.pt")
        model.save(ckpt_path, extra={"epoch": epoch + 1, "recall@5": recall_val,
                                     "train_args": vars(args)})
        latest = os.path.join(args.ckpt_dir, "latest.pt")
        if os.path.islink(latest):
            os.remove(latest)
        os.symlink(os.path.abspath(ckpt_path), latest)
        if do_val:
            if recall_val > best_recall:
                best_recall    = recall_val
                patience_count = 0
                best_path = os.path.join(args.ckpt_dir, "best.pt")
                model.save(best_path, extra={"epoch": epoch + 1, "recall@5": recall_val,
                                             "train_args": vars(args)})
                print(f"  ✓ New best Recall@5 = {recall_val:.4f} → saved to {best_path}")
            else:
                patience_count += 1
                print(f"  No improvement ({patience_count}/{args.patience}). "
                      f"Best Recall@5 still {best_recall:.4f}")
                if patience_count >= args.patience:
                    print(f"  Early stopping triggered after epoch {epoch+1}.")
                    break
        if args.refresh_hard_neg:
            print("  Refreshing hard negatives …")
            subprocess.run([
                sys.executable, os.path.join(ROOT, "src", "precompute_hard_negatives.py"),
                "--model_ckpt", ckpt_path,
                "--out_path", args.hard_neg,
            ], check=False)
            if os.path.exists(args.hard_neg):
                with open(args.hard_neg) as f:
                    hard_neg_cache = json.load(f)
                sampler.hard_neg_cache = hard_neg_cache
                print(f"  Hard neg cache refreshed: {len(hard_neg_cache)} entries.")

    print(f"\nTraining complete. Best Recall@5 = {best_recall:.4f}")
    print(f"Checkpoints in: {args.ckpt_dir}")


if __name__ == "__main__":
    main()
