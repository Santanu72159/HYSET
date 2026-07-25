import argparse
import json
import os
import re
import sys
import time
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Judge prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an objective evaluator assessing how well an AI agent completed a user's task "
    "using external tools. Evaluate solely based on how well the final answer satisfies "
    "the user's request. Respond in JSON only."
)

USER_TEMPLATE = """\
## User Query
{query}

## Available Tools
{tools}

## Agent's Final Answer
{final_answer}

## Evaluation Task
Rate how well the agent's final answer satisfies the user's query with a continuous score
from 0.0 to 1.0 using the following rubric:

  1.0 — Fully correct and complete; directly answers every part of the query.
  0.8 — Mostly correct; minor details missing or slightly imprecise.
  0.6 — Partially correct; answers the main request but misses important sub-parts.
  0.4 — Weak attempt; provides some relevant information but largely incomplete or off-topic.
  0.2 — Minimal relevance; tried but failed to answer the query meaningfully.
  0.0 — Complete failure: agent gave up, gave no answer, or is entirely irrelevant.

Respond with ONLY valid JSON (no markdown fences), exactly this schema:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "one or two sentence explanation"
}}"""

PASS_THRESHOLD = 0.5   # score >= threshold counts as a binary "pass"

def call_judge(prompt: str, api_key: str, api_base: str, model: str,
                  max_retries: int = 3) -> Optional[dict]:
    """
    Call an OpenAI-compatible chat completions API.
    Returns parsed JSON dict or None on failure.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[judge] openai package not found – pip install openai")
        return None

    client = OpenAI(api_key=api_key, base_url=api_base or None)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            return json.loads(text)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [judge] attempt {attempt+1} failed: {e} – retrying in {wait}s")
            time.sleep(wait)
    return None


def _collect_actions(node: dict, actions: list):
    if node.get("node_type") == "Action":
        name = node.get("description", "").strip()
        if name:
            actions.append(name)
    for child in node.get("children", []):
        _collect_actions(child, actions)


def parse_dfs_file(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    win = data.get("win", False)
    all_actions: list = []
    _collect_actions(data.get("tree", {}).get("tree", {}), all_actions)
    tool_calls = [a for a in all_actions if a.strip().lower() != "finish"]

    final_answer = None
    for cand in data.get("compare_candidates", []):
        ans = cand.get("final_answer") or cand.get("answer")
        if ans:
            final_answer = str(ans)
            break

    # Also look inside answer_generation
    if not final_answer:
        ag = data.get("answer_generation", {})
        for step in ag.get("answer", []):
            if isinstance(step, dict):
                fc = step.get("message", {}).get("tool_calls", [])
                for call in (fc or []):
                    fn = call.get("function", {})
                    if fn.get("name") == "Finish":
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                            if args.get("return_type") == "give_answer":
                                final_answer = args.get("final_answer", "")
                        except Exception:
                            pass

    return {
        "win":         win,
        "tool_calls":  tool_calls,
        "final_answer": final_answer,
    }

def standardize(s: str) -> str:
    s = re.sub(r"[^一-龥a-zA-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).lower().strip("_")
    if s and s[0].isdigit():
        s = "get_" + s
    return s


def change_name(name: str) -> str:
    return ("is_" + name) if name in {"from","class","return","false","true","id","and"} else name


def relevant_api_to_func(tool_name: str, api_name: str) -> str:
    return f"{change_name(standardize(api_name))}_for_{standardize(tool_name)}"

def run_judge(
    label: str,
    answers_dir: str,
    offline_metrics: dict,
    queries: list,
    api_key: str,
    api_base: str,
    model: str,
    out_dir: str,
    num_queries: Optional[int] = None,
) -> list:
    qid_to_query = {str(q["query_id"]): q for q in queries}
    per_query_offline = offline_metrics.get("per_query", {})

    answer_files = sorted([f for f in os.listdir(answers_dir) if f.endswith(".json")])
    if num_queries:
        answer_files = answer_files[:num_queries]

    results = []
    method_label = label.split("_", 1)[1] if "_" in label else label

    for idx, fname in enumerate(answer_files, 1):
        qid  = fname.split("_")[0]
        path = os.path.join(answers_dir, fname)

        print(f"  [{idx}/{len(answer_files)}] query_id={qid} …", end=" ", flush=True)

        try:
            parsed = parse_dfs_file(path)
        except Exception as e:
            print(f"parse error: {e}")
            continue

        q = qid_to_query.get(qid, {})
        user_query   = q.get("query", "")
        relevant_apis = q.get("relevant APIs", [])

        # Available tools summary (for judge context)
        avail_tools = [entry.get("tool_name", "")
                       for entry in q.get("api_list", []) if "tool_name" in entry]
        avail_tools_str = ", ".join(sorted(set(avail_tools))) if avail_tools else "(retrieved)"

        final_answer = parsed["final_answer"] or (
            "[Agent gave up or produced no answer]" if not parsed["win"] else "[No answer extracted]"
        )

        # Tool call accuracy (offline)
        relevant_funcs = {relevant_api_to_func(tn, an) for tn, an in relevant_apis}
        tc = parsed["tool_calls"]
        tca = (sum(1 for c in tc if c in relevant_funcs) / len(tc)) if tc else 0.0

        # Retrieve offline metrics
        off = per_query_offline.get(qid, {})

        # ── Call judge ───────────────────────────────────────────────────────
        evaluation: dict = {}
        if api_key:
            prompt = USER_TEMPLATE.format(
                query=user_query,
                tools=avail_tools_str,
                final_answer=final_answer,
            )
            verdict = call_judge(prompt, api_key=api_key, api_base=api_base, model=model)
            if verdict:
                raw_score = verdict.get("score", verdict.get("pass", 0))
                # Tolerate boolean responses from older prompts
                if isinstance(raw_score, bool):
                    raw_score = 1.0 if raw_score else 0.0
                score = float(max(0.0, min(1.0, raw_score)))
                evaluation = {
                    "judge_model": model,
                    "score":       score,
                    "pass":        score >= PASS_THRESHOLD,
                    "reason":      verdict.get("reason", ""),
                }
                print(f"score={score:.2f}  pass={evaluation['pass']}")
            else:
                print("judge failed → heuristic")
                heuristic_score = 1.0 if parsed["win"] else 0.0
                evaluation = {
                    "judge_model": "heuristic",
                    "score":       heuristic_score,
                    "pass":        bool(parsed["win"]),
                    "reason":      "Judge call failed; using agent win flag.",
                }
        else:
            # No API key: heuristic
            heuristic_score = 1.0 if parsed["win"] else 0.0
            evaluation = {
                "judge_model": "heuristic",
                "score":       heuristic_score,
                "pass":        bool(parsed["win"]),
                "reason":      "No judge API key provided; using agent win flag.",
            }
            print(f"heuristic score={heuristic_score:.1f}")

        record = {
            "query_id":         qid,
            "user_query":       user_query,
            "retriever_method": method_label.upper(),
            "retrieved_apis":   off.get("retrieved", []),
            "final_answer":     final_answer,
            "tool_calls":       tc,
            "tool_call_accuracy": tca,
            "recall_at_k":      off.get("recall_at_k", None),
            "precision_at_k":   off.get("precision_at_k", None),
            "evaluation":       evaluation,
        }
        results.append(record)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",          required=True)
    parser.add_argument("--judge_key",   default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--judge_base",  default=os.getenv("JUDGE_API_BASE",
                                                               ""))
    parser.add_argument("--judge_model", default=os.getenv("JUDGE_MODEL", ""))
    parser.add_argument("--num_queries",    type=int, default=None,
                        help="Evaluate only the first N queries (default: all)")
    parser.add_argument("--out_dir",        default=os.path.join(ROOT, "results", "end2end"))
    parser.add_argument("--queries",        default=os.path.join(
        ROOT, "stabletoolbench", "solvable_queries", "test_instruction", "G1_instruction.json"))
    args = parser.parse_args()

    label_dir    = os.path.join(args.out_dir, args.label)
    answers_dir  = os.path.join(label_dir, "answers")
    offline_path = os.path.join(label_dir, "offline_metrics.json")

    with open(args.queries) as f:
        queries = json.load(f)
    with open(offline_path) as f:
        offline = json.load(f)

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

    # Save per-query JSON
    out_path = os.path.join(label_dir, "judge_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_pass = sum(1 for r in results if r["evaluation"].get("pass", False))
    print(f"\n[{args.label}] Judge done: {n_pass}/{len(results)} passed")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
