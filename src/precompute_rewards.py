import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFS_PATH   = os.path.join(ROOT, "data", "toolllama_G123_dfs_train.json")
CACHE_PATH = os.path.join(ROOT, "data", "reward_cache.json")

PASS_THRESHOLD = 0.5

SYSTEM_PROMPT = (
    "You are an objective evaluator. Given a user query and an agent's final answer, "
    "rate how well the answer satisfies the query. Respond in JSON only."
)

USER_TEMPLATE = """\
## User Query
{query}

## Agent's Final Answer
{final_answer}

## Task
Rate how well the final answer satisfies the query (0.0-1.0):
  1.0 — Fully correct and complete.
  0.8 — Mostly correct, minor gaps.
  0.6 — Partially correct.
  0.4 — Some relevant content but largely incomplete.
  0.2 — Minimal relevance.
  0.0 — Complete failure or gave up.

Respond with ONLY valid JSON:
{{"score": <float 0.0-1.0>, "reason": "one sentence"}}"""


def query_hash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


def standardize(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).lower().strip("_")
    if s and s[0].isdigit():
        s = "get_" + s
    return s


def change_name(n: str) -> str:
    return ("is_" + n) if n in {"from", "class", "return", "false", "true", "id", "and"} else n


def parse_trajectory(item: dict) -> dict:
    """Extract query, final_answer, and APIs called from a DFS conversation."""
    convs = item.get("conversations", [])
    query = ""
    final_answer = ""
    apis: list = []

    for c in convs:
        role = c.get("from", "")
        val  = c.get("value", "")

        if role == "user" and not query:
            # Strip trailing "Begin!" instruction
            query = val.replace("Begin!", "").strip()

        if role == "assistant":
            m = re.search(r"Action:\s*(\S+)", val)
            if m:
                fn = m.group(1).strip()
                if fn == "Finish":
                    # Extract final answer from Action Input JSON
                    mi = re.search(r"Action Input:\s*(\{.*\})", val, re.DOTALL)
                    if mi:
                        try:
                            inp = json.loads(mi.group(1))
                            if inp.get("return_type") == "give_answer":
                                final_answer = inp.get("final_answer", "")
                        except Exception:
                            pass
                else:
                    if fn and fn != "invalid_hallucination_function_name":
                        apis.append(fn)

    return {"query": query, "final_answer": final_answer, "apis": list(dict.fromkeys(apis))}


def call_judge(query: str, final_answer: str,
                  api_key: str, api_base: str, model: str,
                  max_retries: int = 3) -> Optional[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key, base_url=api_base or None)
    answer_str = final_answer or "[Agent gave up or produced no answer]"
    prompt = USER_TEMPLATE.format(query=query, final_answer=answer_str)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [judge] attempt {attempt+1} failed: {e} – retry in {wait}s")
            time.sleep(wait)
    return None


def process_one(item: dict, api_key: str, api_base: str, model: str) -> tuple:
    parsed = parse_trajectory(item)
    query  = parsed["query"]
    if not query:
        return None, None

    key = query_hash(query)

    # Heuristic fallback: give_answer vs give_up
    heuristic = 1.0 if parsed["final_answer"] else 0.0

    if api_key:
        verdict = call_judge(query, parsed["final_answer"], api_key, api_base, model)
        if verdict:
            raw = verdict.get("score", heuristic)
            score = float(max(0.0, min(1.0, raw)))
        else:
            score = heuristic
    else:
        score = heuristic

    record = {
        "score": score,
        "pass":  score >= PASS_THRESHOLD,
        "apis":  parsed["apis"],
    }
    return key, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfs_path",  default=DFS_PATH)
    parser.add_argument("--out_path",  default=CACHE_PATH)
    parser.add_argument("--workers",   type=int, default=10,
                        help="Concurrent judge API calls")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Only process first N trajectories (debugging)")
    parser.add_argument("--api_key",   default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--api_base",  default=os.getenv("JUDGE_API_BASE",
                                                          ""))
    parser.add_argument("--model",     default=os.getenv("JUDGE_MODEL", ""))
    args = parser.parse_args()

    print(f"Loading DFS trajectories from {args.dfs_path} …")
    with open(args.dfs_path) as f:
        data = json.load(f)

    if args.limit:
        data = data[: args.limit]
    print(f"  {len(data)} trajectories loaded.")

    # Load existing cache (resume support)
    if os.path.exists(args.out_path):
        with open(args.out_path) as f:
            cache: dict = json.load(f)
        print(f"  Resuming: {len(cache)} already cached.")
    else:
        cache = {}

    # Filter out already-done items
    pending = []
    for item in data:
        convs = item.get("conversations", [])
        query = next((c["value"].replace("Begin!", "").strip()
                      for c in convs if c.get("from") == "user"), "")
        if query and query_hash(query) not in cache:
            pending.append(item)

    print(f"  Pending: {len(pending)} trajectories to score.")

    if not pending:
        print("All done.")
        return

    done = 0
    save_every = 200

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, item, args.api_key, args.api_base, args.model): item
            for item in pending
        }
        for fut in as_completed(futures):
            key, record = fut.result()
            if key and record:
                cache[key] = record
            done += 1
            if done % save_every == 0:
                with open(args.out_path, "w") as f:
                    json.dump(cache, f)
                print(f"  [{done}/{len(pending)}] saved checkpoint …")

    with open(args.out_path, "w") as f:
        json.dump(cache, f, indent=2)

    n_pass = sum(1 for v in cache.values() if v.get("pass"))
    print(f"\nDone. {len(cache)} entries, pass rate = {n_pass/len(cache):.3f}")
    print(f"Saved → {args.out_path}")


if __name__ == "__main__":
    main()
