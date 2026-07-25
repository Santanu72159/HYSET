import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from hyset_model import HYSETModel, func_name_of

# ── stabletoolbench sys.path setup ────────────────────────────────────────────
_STBTOOLBENCH = os.path.join(ROOT, "stabletoolbench")


def _ensure_stbtoolbench_path() -> None:
    for p in (_STBTOOLBENCH,):
        if p not in sys.path:
            sys.path.insert(0, p)


SPLITS = {
    "I1-Inst": ("G1_query.json", "G1_instruction_test_query_ids.json"),
    "I1-Tool": ("G1_query.json", "G1_tool_test_query_ids.json"),
    "I1-Cate": ("G1_query.json", "G1_category_test_query_ids.json"),
    "I2-Inst": ("G2_query.json", "G2_instruction_test_query_ids.json"),
    "I2-Cate": ("G2_query.json", "G2_category_test_query_ids.json"),
    "I3-Inst": ("G3_query.json", "G3_instruction_test_query_ids.json"),
}

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


def compute_offline_metrics(
    retrieved_list: List[List[str]],
    gt_list: List[List[str]],
) -> dict:
    results = {}
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



def build_toolenv_from_corpus(corpus: dict, tool_root_dir: str) -> None:
    def _std(s: str) -> str:
        s = re.sub(r"[^一-龥a-zA-Z0-9_]", "_", s)
        s = re.sub(r"_+", "_", s).lower().strip("_")
        return ("get_" + s) if s and s[0].isdigit() else s

    groups: dict = {}
    for entry in corpus.values():
        aj   = entry["full_api_json"]
        cate = aj["category_name"]
        key  = (cate, aj["tool_name"])
        if key not in groups:
            groups[key] = {
                "tool_name":        aj["tool_name"],
                "tool_description": "",
                "api_list":         [],
            }
        groups[key]["api_list"].append({
            "name":                aj["api_name"],
            "description":         aj.get("api_description", ""),
            "required_parameters": aj.get("required_parameters", []),
            "optional_parameters": aj.get("optional_parameters", []),
        })

    written = 0
    for (cate, tool_orig), tool_data in groups.items():
        cate_dir = os.path.join(tool_root_dir, cate)
        os.makedirs(cate_dir, exist_ok=True)
        fpath = os.path.join(cate_dir, _std(tool_orig) + ".json")
        if not os.path.exists(fpath):
            with open(fpath, "w") as fw:
                json.dump(tool_data, fw)
            written += 1

    if written:
        print(f"  [toolenv] wrote {written} new tool JSON files → {tool_root_dir}")
    else:
        print(f"  [toolenv] tool_root_dir up-to-date ({tool_root_dir})")


# ── HYSET fn_keys → official api_list ─────────────────────────────────────────

def hyset_to_api_list(fn_keys: List[str], corpus: dict) -> List[dict]:
    """
    Convert HYSET-retrieved fn_keys to the official ToolBench api_list format:
      [{category_name, tool_name, api_name}, ...]
    Keys not found in the corpus are silently dropped.
    """
    result = []
    for fn in fn_keys:
        if fn not in corpus:
            continue
        aj = corpus[fn]["full_api_json"]
        result.append({
            "category_name": aj["category_name"],
            "tool_name":     aj["tool_name"],
            "api_name":      aj["api_name"],
        })
    return result


def run_toolbench_pipeline(
    split_queries:       List[dict],
    retrieved_per_query: List[List[str]],
    corpus:              dict,
    label:               str,
    split_name:          str,
    toolllama_url:       str,
    toolllama_model:     str,
    mirror_url:          str,
    tool_root_dir:       str,
    out_dir:             str,
) -> Tuple[int, int]:
    _ensure_stbtoolbench_path()

    try:
        from toolbench.inference.Downstream_tasks.rapidapi import (
            pipeline_runner,
            get_white_list,
            contain,
        )
        from toolbench.inference.LLM.tool_llama_vllm_model import ToolLLaMA_vllm
        from toolbench.utils import standardize
    except ImportError as exc:
        print(f"[WARN] stabletoolbench import failed ({exc}); skipping agent inference.")
        return 0, 0

    os.makedirs(out_dir, exist_ok=True)

    # Route rapidapi_wrapper._step() to MirrorAPI via SERVICE_URL env var.
    # stabletoolbench reads os.getenv("SERVICE_URL") in rapidapi_wrapper.__init__.
    os.environ["SERVICE_URL"] = mirror_url.rstrip("/") + "/virtual"

    # Build white_list once for this split (needed by generate_task_list-equivalent logic).
    print(f"  [toolenv] building white_list from {tool_root_dir} …")
    white_list = get_white_list(tool_root_dir)
    print(f"  [toolenv] {len(white_list)} tools indexed.")

    llm = ToolLLaMA_vllm(
        model=toolllama_model,
        template="tool-llama-single-round",
        openai_key="EMPTY",
        base_url=toolllama_url.rstrip("/") + "/v1",
    )

    class _PipelineArgs:
        backbone_model             = llm
        method                     = "DFS_woFilter_w2"
        toolbench_key              = ""
        rapidapi_key               = ""
        use_rapidapi_key           = False
        api_customization          = False
        max_observation_length     = 1024
        observ_compress_method     = "truncate"
        max_source_sequence_length = 2048
        max_sequence_length        = 8192
        openai_key                 = ""
        model_path                 = ""
        lora                       = False
        lora_path                  = ""

    pipe_args = _PipelineArgs()
    pipe_args.tool_root_dir = tool_root_dir

    runner = pipeline_runner(pipe_args, add_retrieval=False, process_id=0, server=True)

    n_done = n_answered = 0

    for i, (q, fn_keys) in enumerate(zip(split_queries, retrieved_per_query)):
        qid     = str(q.get("query_id", i))
        raw_q   = q.get("query", "")
        q_str   = " ".join(raw_q) if isinstance(raw_q, list) else raw_q

        out_file = os.path.join(out_dir, f"{qid}_DFS_woFilter_w2.json")

        if os.path.exists(out_file):
            n_done += 1
            try:
                with open(out_file) as fh:
                    d = json.load(fh)
                if d.get("answer_generation", {}).get("valid_data", False):
                    n_answered += 1
            except Exception:
                pass
            print(f"  [{i+1}/{len(split_queries)}] skip (resume): qid={qid}")
            continue

        api_list = hyset_to_api_list(fn_keys, corpus)
        if not api_list:
            print(f"  [{i+1}/{len(split_queries)}] qid={qid}: no valid APIs retrieved, skip")
            continue

        data_dict = {"query": q_str, "query_id": qid, "api_list": api_list}

        origin_tool_names = [standardize(e["tool_name"]) for e in api_list]
        tool_des = contain(origin_tool_names, white_list)
        if tool_des is False:
            tool_des = []
        else:
            tool_des = [[c["standard_tool_name"], c["description"]] for c in tool_des]

        try:
            runner.run_single_task(
                method          = "DFS_woFilter_w2",
                backbone_model  = llm,
                query_id        = qid,
                data_dict       = data_dict,
                args            = pipe_args,
                output_dir_path = out_dir,
                tool_des        = tool_des,
                retriever       = None,
                process_id      = 0,
                callbacks       = [],
                server          = False,   # enable per-query resume check
            )
        except Exception as exc:
            print(f"  [pipeline] qid={qid} error: {exc}")
            n_done += 1
            continue

        n_done += 1
        try:
            with open(out_file) as fh:
                d = json.load(fh)
            valid = d.get("answer_generation", {}).get("valid_data", False)
            fa    = d.get("answer_generation", {}).get("final_answer", "")
            if n_done % 10 == 1:
                print(f"  [{i+1}/{len(split_queries)}] qid={qid}  valid={valid}  "
                      f"ans={str(fa)[:80]!r}")
            if valid:
                n_answered += 1
        except Exception:
            pass

    return n_done, n_answered

def main():
    parser = argparse.ArgumentParser(description="Evaluate HYSET on all 6 test splits")

    # model / retriever
    parser.add_argument("--ckpt",    required=True, help="HYSET checkpoint (best.pt)")
    parser.add_argument("--encoder", default=os.path.join(ROOT, "models", "ToolBench_IR_bert_based_uncased"))
    parser.add_argument("--corpus",  default=os.path.join(ROOT, "data", "hyset_corpus.json"))
    parser.add_argument("--label",   default="HYSET", help="Experiment label")
    parser.add_argument("--k_pool",        type=int, default=50)
    parser.add_argument("--use_greedy",    action="store_true",
                        help="Use greedy set expansion instead of exhaustive enumeration")
    parser.add_argument("--k_pool_greedy", type=int, default=200,
                        help="Dense pool size when --use_greedy (allows larger pool cheaply)")
    parser.add_argument("--device",  default="cuda:0" if torch.cuda.is_available() else "cpu")

    # agent inference
    parser.add_argument("--run_agent", action="store_true",
                        help="Run the official StableToolBench pipeline "
                             "(rapidapi_wrapper + DFS_woFilter_w2 + ToolLLaMA_vllm)")
    parser.add_argument("--agent_k_return", type=int, default=5,
                        help="Number of tools given to the agent (default 5)")
    parser.add_argument("--toolllama_url",
                        default=os.getenv("TOOLLLAMA_URL", "http://localhost:8000"))
    parser.add_argument("--toolllama_model",
                        default=os.getenv("TOOLLLAMA_MODEL", "ToolLLaMA-2-7b-v2"))
    parser.add_argument("--mirror_url",
                        default=os.getenv("MIRROR_URL", "http://localhost:8080"))

    # toolenv path (auto-built from corpus when absent)
    parser.add_argument("--tool_root_dir",
                        default=os.path.join(ROOT, "data", "toolenv", "tools"),
                        help="ToolBench tool JSON directory (built from corpus if incomplete)")

    # output paths
    parser.add_argument("--result_dir",    default=os.path.join(ROOT, "result"))
    parser.add_argument("--agent_out_dir", default=os.path.join(ROOT, "outputs", "agent_outputs"),
                        help="Root directory for per-split agent outputs")

    args = parser.parse_args()

    # ── load corpus ──────────────────────────────────────────────────────────
    print("Loading corpus …")
    with open(args.corpus) as f:
        corpus = json.load(f)
    print(f"  {len(corpus)} APIs.")

    # ── load HYSET model ──────────────────────────────────────────────────────
    print(f"Loading HYSET model from {args.ckpt} …")
    model = HYSETModel.load(args.ckpt, args.encoder, device=args.device)
    model.eval()
    print(f"  Model loaded (N_T={model.N_T}, d={model.d}, m_max={model.m_max})")

    # ── build toolenv (one-time) ─────────────────────────────────────────────
    if args.run_agent:
        print(f"Building/verifying tool_root_dir …")
        build_toolenv_from_corpus(corpus, args.tool_root_dir)

    label_dir = os.path.join(args.result_dir, args.label)
    os.makedirs(label_dir, exist_ok=True)

    # ── save config ──────────────────────────────────────────────────────────
    ckpt_payload = torch.load(args.ckpt, map_location="cpu")
    config = {
        "label":        args.label,
        "checkpoint":   args.ckpt,
        "corpus":       args.corpus,
        "k_pool":       args.k_pool,
        "model_d":      model.d,
        "model_m_max":  model.m_max,
        "model_N_T":    model.N_T,
        "train_args":   ckpt_payload.get("train_args", {}),
        "best_recall5": ckpt_payload.get("recall@5", None),
    }
    with open(os.path.join(label_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── per-split evaluation ─────────────────────────────────────────────────
    all_split_metrics: Dict[str, dict] = {}
    INST_DIR = os.path.join(ROOT, "data", "instruction")
    ID_DIR   = os.path.join(ROOT, "data", "test_query_ids")

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

        retrieved_list:       List[List[str]] = []
        agent_retrieved_list: List[List[str]] = []
        gt_list:              List[List[str]] = []

        for q in queries:
            gt_fns = [
                func_name_of(e[0], e[1])
                for e in q.get("relevant APIs", [])
                if len(e) >= 2 and e[0] and e[1]
            ]
            gt_fns = [fn for fn in gt_fns if fn in model.fn_to_idx]
            gt_list.append(gt_fns)

            raw_q    = q["query"]
            q_str    = " ".join(raw_q) if isinstance(raw_q, list) else raw_q

            eff_k_pool = args.k_pool_greedy if args.use_greedy else args.k_pool
            ret = model.retrieve(
                q_str,
                target_size=None,
                k_pool=eff_k_pool,
                k_return=5,
                use_greedy=args.use_greedy,
            )
            retrieved_list.append(ret)

            if args.run_agent:
                agent_ret = model.retrieve(
                    q_str,
                    target_size=None,
                    k_pool=args.k_pool,
                    k_return=args.agent_k_return,
                    return_predicted_set=True,
                    normalize_across_orders=True,
                )
                agent_retrieved_list.append(agent_ret)

        # ── offline metrics ───────────────────────────────────────────────
        metrics = compute_offline_metrics(retrieved_list, gt_list)
        print("  Offline → " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        if args.run_agent:
            from collections import Counter
            mstar_dist = Counter(len(r) for r in agent_retrieved_list)
            gt_dist    = Counter(len(g) for g in gt_list)
            print("  m* distribution (HYSET predicted): " +
                  "  ".join(f"m={k}:{v}" for k, v in sorted(mstar_dist.items())))
            print("  GT  distribution (ground truth):  " +
                  "  ".join(f"m={k}:{v}" for k, v in sorted(gt_dist.items())))

        with open(os.path.join(split_dir, "offline_metrics.json"), "w") as f:
            json.dump({"split": split_name, "n_queries": len(queries), **metrics}, f, indent=2)

        split_summary = {"split": split_name, "n_queries": len(queries), **metrics}

        # ── agent inference (official ToolBench pipeline) ─────────────────
        if args.run_agent:
            agent_split_dir = os.path.join(args.agent_out_dir, args.label, split_name)
            print(f"  Running DFS_woFilter_w2 via StableToolBench pipeline → {agent_split_dir}")

            n_done, n_answered = run_toolbench_pipeline(
                split_queries       = queries,
                retrieved_per_query = agent_retrieved_list,
                corpus              = corpus,
                label               = args.label,
                split_name          = split_name,
                toolllama_url       = args.toolllama_url,
                toolllama_model     = args.toolllama_model,
                mirror_url          = args.mirror_url,
                tool_root_dir       = args.tool_root_dir,
                out_dir             = agent_split_dir,
            )
            split_summary["agent_n_queries"]  = n_done
            split_summary["agent_n_answered"] = n_answered
            print(f"  Agent done: {n_answered}/{n_done} queries answered.")

        with open(os.path.join(split_dir, "summary.json"), "w") as f:
            json.dump(split_summary, f, indent=2)

        all_split_metrics[split_name] = split_summary

    # ── overall average (offline metrics only) ───────────────────────────────
    overall_dir = os.path.join(label_dir, "overall")
    os.makedirs(overall_dir, exist_ok=True)

    offline_keys = [k for k in list(all_split_metrics.values())[0]
                    if k not in ("split", "n_queries", "agent_n_queries", "agent_n_answered")]
    overall = {"label": args.label, "splits": list(SPLITS.keys())}
    for k in offline_keys:
        vals = [v[k] for v in all_split_metrics.values() if k in v]
        overall[k] = sum(vals) / len(vals) if vals else 0.0

    with open(os.path.join(overall_dir, "summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"  OVERALL ({args.label})")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"    {k:15s} = {v:.4f}")

    if args.run_agent:
        print(f"\n  Agent outputs → {args.agent_out_dir}/{args.label}/")

    print(f"\nAll results saved under: {label_dir}")


if __name__ == "__main__":
    main()
