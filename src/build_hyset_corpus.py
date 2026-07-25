import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUCTION_DIR = os.path.join(ROOT, "data", "instruction")
OUT_PATH = os.path.join(ROOT, "data", "hyset_corpus.json")


def standardize(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).lower().strip("_")
    if s and s[0].isdigit():
        s = "get_" + s
    return s


def change_name(n: str) -> str:
    return ("is_" + n) if n in {"from", "class", "return", "false", "true", "id", "and"} else n


def func_name_of(tool_name: str, api_name: str) -> str:
    return f"{change_name(standardize(api_name))}_for_{standardize(tool_name)}"


def api_text(entry: dict) -> str:
    parts = [
        entry.get("category_name", ""),
        entry.get("tool_name", ""),
        entry.get("api_name", ""),
        entry.get("api_description", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def main():
    corpus: dict = {}

    for fname in ["G1_query.json", "G2_query.json", "G3_query.json"]:
        path = os.path.join(INSTRUCTION_DIR, fname)
        print(f"Scanning {fname} …", flush=True)
        with open(path) as f:
            queries = json.load(f)
        for item in queries:
            for api in item.get("api_list", []):
                tool = api.get("tool_name", "")
                name = api.get("api_name", "")
                if not tool or not name:
                    continue
                fn = func_name_of(tool, name)
                if fn not in corpus:
                    corpus[fn] = {
                        "idx":          len(corpus),
                        "category":     api.get("category_name", ""),
                        "tool_name":    tool,
                        "api_name":     name,
                        "text":         api_text(api),
                        "full_api_json": api,
                    }

    print(f"Total unique API endpoints: {len(corpus)}")

    with open(OUT_PATH, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
