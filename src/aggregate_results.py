import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "results", "end2end")

EXPERIMENTS = [
    ("A0", "A0_random",  "ToolLLaMA-v2 + Random + DFSDT"),
    ("A1", "A1_bm25",    "ToolLLaMA-v2 + BM25 + DFSDT"),
    ("A2", "A2_dense",   "ToolLLaMA-v2 + Dense + DFSDT"),
    ("A3", "A3_ied",     "ToolLLaMA-v2 + IED + DFSDT"),
]

METRICS = [
    ("pass_rate",                 "Pass Rate"),
    ("tool_call_accuracy",        "Tool Call Acc."),
    ("required_tool_recall_at_k", "Req. Tool Recall@K"),
    ("tool_precision_at_k",       "Tool Prec.@K"),
]


def load_summary(label: str) -> dict:
    path = os.path.join(OUT, label, "summary.json")
    if not os.path.exists(path):
        # Try offline only
        path2 = os.path.join(OUT, label, "offline_metrics.json")
        if os.path.exists(path2):
            with open(path2) as f:
                d = json.load(f)
            return {
                "pass_rate":                 float("nan"),
                "tool_call_accuracy":        float("nan"),
                "required_tool_recall_at_k": d.get("recall_at_k_mean", float("nan")),
                "tool_precision_at_k":       d.get("precision_at_k_mean", float("nan")),
            }
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(v) -> str:
    try:
        if v != v:  # NaN
            return "  —  "
        return f"{v:.4f}"
    except Exception:
        return str(v)


def main():
    rows = []
    for exp_id, label, name in EXPERIMENTS:
        s = load_summary(label)
        if not s:
            print(f"  [WARN] No summary for {label} – skipping")
            continue
        row = {"id": exp_id, "name": name}
        for key, _ in METRICS:
            row[key] = s.get(key, float("nan"))
        rows.append(row)

    if not rows:
        print("No results found. Run End2end.sh first.")
        return

    col_w = [4, 35, 11, 16, 20, 14]
    header_cols = ["ID", "Method", "Pass Rate",
                   "Tool Call Acc.", "Req. Recall@K", "Prec.@K"]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"

    def row_str(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_w)) + " |"

    print()
    print("=" * 100)
    print("  END-TO-END EXPERIMENT RESULTS  (G1_instruction, 163 queries, Top-K=5)")
    print("=" * 100)
    print(sep)
    print(row_str(header_cols))
    print(sep)
    for r in rows:
        cells = [
            r["id"], r["name"],
            fmt(r["pass_rate"]),
            fmt(r["tool_call_accuracy"]),
            fmt(r["required_tool_recall_at_k"]),
            fmt(r["tool_precision_at_k"]),
        ]
        print(row_str(cells))
    print(sep)
    print()

    table = {"experiments": rows}
    out_path = os.path.join(OUT, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
