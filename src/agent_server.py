import argparse
import json
import os
import sys
import time
from typing import Optional

from flask import Flask, jsonify, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

app = Flask(__name__)

CFG: dict = {}

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Given a user query and a list of available tools, "
    "answer the query as accurately and completely as possible. "
    "If you need to call a tool, describe what result it would return and use that "
    "to construct your final answer. Respond with the final answer only."
)


def call_toolllama(query: str, tools: list) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key="EMPTY",
            base_url=CFG["vllm_url"].rstrip("/") + "/v1",
        )

        tool_block = "\n".join(
            f"- {t.get('api_name', '')}: {t.get('api_description', '')[:300]}"
            for t in tools
        ) or "(no tools)"

        # ToolLLaMA-2 is LLaMA-2-based; format manually as [INST]…[/INST].
        prompt = (
            f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"Query: {query}\n\nAvailable tools:\n{tool_block} [/INST]"
        )

        resp = client.completions.create(
            model=CFG["vllm_model"],
            prompt=prompt,
            max_tokens=512,
            temperature=0.1,
            timeout=60,
        )
        return resp.choices[0].text.strip()
    except Exception as e:
        app.logger.warning(f"vllm call failed: {e}")
        return ""

JUDGE_SYSTEM = (
    "You are an objective evaluator assessing how well an AI agent answered a user query. "
    "Respond in JSON only."
)

JUDGE_USER = """\
## User Query
{query}

## Agent Answer
{agent_answer}

## Ground Truth (may be empty)
{ground_truth}


Rate the agent answer with a score 0.0–1.0:
  1.0 = fully correct and complete
  0.7 = mostly correct, minor gaps
  0.4 = partial / incomplete
  0.0 = wrong, irrelevant, or no answer

Respond ONLY with valid JSON:
{{"score": <float 0.0-1.0>, "reason": "one sentence"}}"""

## Specific Evaluation.

def judge_answer(query: str, agent_answer: str, ground_truth: str) -> float:
    if not CFG.get("judge_key"):
        return 0.5 if agent_answer.strip() else 0.0

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=CFG["judge_key"],
            base_url=CFG["judge_base"] or None,
        )
        prompt = JUDGE_USER.format(
            query=query,
            agent_answer=agent_answer or "(no answer)",
            ground_truth=ground_truth or "(not provided)",
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=CFG["judge_model"],
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=128,
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
                raw = data.get("score", data.get("pass", 0))
                if isinstance(raw, bool):
                    raw = 1.0 if raw else 0.0
                return float(max(0.0, min(1.0, raw)))
            except Exception as e:
                time.sleep(2 ** attempt)
    except Exception as e:
        app.logger.warning(f"Judge call failed: {e}")

    return 0.5 if agent_answer.strip() else 0.0

@app.route("/run_agent", methods=["POST"])
def run_agent():
    data = request.get_json(force=True)
    query        = data.get("query", "")
    tools        = data.get("tools", [])
    ground_truth = data.get("answer", "")

    if not query:
        return jsonify({"score": 0.0, "agent_answer": ""}), 400

    agent_answer = call_toolllama(query, tools)
    score        = judge_answer(query, agent_answer, ground_truth)

    return jsonify({"score": score, "agent_answer": agent_answer})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_url",      default=os.getenv("VLLM_URL",   "http://localhost:12345"))
    p.add_argument("--vllm_model",    default=os.getenv("VLLM_MODEL", "simulation-250123-qwen25-mixed"))
    p.add_argument("--mirror_url",    default=os.getenv("MIRROR_URL", "http://localhost:12001/virtual"))
    p.add_argument("--judge_key",  default=os.getenv("JUDGE_API_KEY", ""))
    p.add_argument("--judge_base", default=os.getenv("JUDGE_API_BASE", ""))
    p.add_argument("--judge_model",default=os.getenv("JUDGE_MODEL", ""))
    p.add_argument("--port",          type=int, default=int(os.getenv("AGENT_PORT", "8080")))
    p.add_argument("--host",          default="0.0.0.0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CFG.update(vars(args))

    print(f"Agent server config:")
    print(f"  vllm        : {CFG['vllm_url']}  model={CFG['vllm_model']}")
    print(f"  judge       : {CFG['judge_base']}  key={'set' if CFG['judge_key'] else 'NOT SET'}")
    print(f"  listening   : {CFG['host']}:{CFG['port']}")

    app.run(host=args.host, port=args.port, threaded=True)
