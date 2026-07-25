import argparse
import glob
import json
import os
import re
import sys

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


def func_name_of(api_name: str, tool_name: str) -> str:
    """Produce the standardized function name used throughout ToolBench."""
    api_std  = change_name(standardize(api_name))
    tool_std = standardize(tool_name)
    return f"{api_std}_for_{tool_std}"

def build_corpus(toolenv_dir: str) -> dict:
    corpus = {}
    tool_jsons = glob.glob(os.path.join(toolenv_dir, "**", "*.json"), recursive=True)
    print(f"Scanning {len(tool_jsons)} tool JSON files …")

    for path in tool_jsons:
        category = os.path.basename(os.path.dirname(path))
        try:
            with open(path) as f:
                tool_json = json.load(f)
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue

        tool_name     = tool_json.get("tool_name") or ""
        tool_name_std = standardize(tool_name)
        tool_desc     = (tool_json.get("tool_description") or "").strip()

        for api in tool_json.get("api_list", []):
            api_name     = api.get("name") or ""
            api_name_std = change_name(standardize(api_name))
            fname        = f"{api_name_std}_for_{tool_name_std}"
            api_desc     = (api.get("description") or "").strip()

            # Rich text used for BM25 / dense retrieval
            text = " ".join(filter(None, [
                tool_name, tool_desc, api_name, api_desc,
                category.replace("_", " "),
            ]))

            corpus[fname] = {
                "category":      category,
                "tool_name":     tool_name,
                "tool_name_std": tool_name_std,
                "api_name":      api_name,
                "api_name_std":  api_name_std,
                "func_name":     fname,
                "text":          text,
                "full_api_json": {
                    "category_name":        category,
                    "tool_name":            tool_name,
                    "api_name":             api_name,
                    "api_description":      api_desc,
                    "required_parameters":  api.get("required_parameters", []),
                    "optional_parameters":  api.get("optional_parameters", []),
                    "method":               api.get("method", "GET"),
                },
            }

    return corpus

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolenv", default=os.path.join(
        ROOT, "stabletoolbench", "toolenv2404_filtered"))
    parser.add_argument("--output", default=os.path.join(
        ROOT, "results", "end2end", "corpus.json"))
    args = parser.parse_args()

    corpus = build_corpus(args.toolenv)
    print(f"Built corpus with {len(corpus)} API functions.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(corpus, f)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
