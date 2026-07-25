import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",          required=True)
    parser.add_argument("--judge_key",   default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--judge_base",  default=os.getenv("JUDGE_API_BASE",
                                                               ""))
    parser.add_argument("--judge_model", default=os.getenv("JUDGE_MODEL", ""))
    parser.add_argument("--num_queries",    type=int, default=None,
                        help="Evaluate only the first N queries")
    parser.add_argument("--out_dir",        default=os.path.join(ROOT, "results", "end2end"))
    parser.add_argument("--queries",        default=os.path.join(
        ROOT, "stabletoolbench", "solvable_queries", "test_instruction", "G1_instruction.json"))
    args = parser.parse_args()

    label_dir    = os.path.join(args.out_dir, args.label)
    answers_dir  = os.path.join(label_dir, "answers")
    offline_path = os.path.join(label_dir, "offline_metrics.json")

    if not os.path.isdir(answers_dir):
        print(f"[ERROR] answers directory not found: {answers_dir}")
        sys.exit(1)

    with open(args.queries) as f:
        queries = json.load(f)
    with open(offline_path) as f:
        offline = json.load(f)

    # ── Run judge ────────────────────────────────────────────────────────────
    from judge import run_judge
    results = run_judge(
        label=args.label,
        answers_dir=answers_dir,
        offline_metrics=offline,
        queries=queries,
        api_key=args.judge_key,
        api_base=args.judge_base,
        model=args.judge_model,
        out_dir=args.out_dir,
        num_queries=args.num_queries,
    )

    # Save per-query detailed records
    judge_path = os.path.join(label_dir, "judge_results.json")
    with open(judge_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("[WARN] No results to aggregate.")
        return

    # Binary pass rate (score >= 0.5) for backward compatibility
    pass_rate  = sum(1 for r in results if r["evaluation"].get("pass", False)) / n
    # Continuous mean score (main reward signal for L_agent)
    mean_score = sum(r["evaluation"].get("score", float(r["evaluation"].get("pass", False)))
                     for r in results) / n
    tca_mean   = sum(r.get("tool_call_accuracy", 0.0) for r in results) / n
    recall     = offline.get("recall_at_k_mean",    0.0)
    precision  = offline.get("precision_at_k_mean", 0.0)

    summary = {
        "label":                     args.label,
        "n_queries_evaluated":       n,
        "pass_rate":                 pass_rate,
        "mean_score":                mean_score,
        "tool_call_accuracy":        tca_mean,
        "required_tool_recall_at_k": recall,
        "tool_precision_at_k":       precision,
        "top_k":                     offline.get("top_k", 5),
        "judge_model":               args.judge_model if args.judge_key else "heuristic",
    }

    summary_path = os.path.join(label_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Results for {args.label}")
    print(f"{'='*55}")
    print(f"  1. Pass Rate               : {pass_rate:.4f}")
    print(f"  2. Tool Call Accuracy      : {tca_mean:.4f}")
    print(f"  3. Required Tool Recall@K  : {recall:.4f}")
    print(f"  4. Tool Precision@K        : {precision:.4f}")
    print(f"{'='*55}")
    print(f"Saved judge_results → {judge_path}")
    print(f"Saved summary       → {summary_path}")


if __name__ == "__main__":
    main()
