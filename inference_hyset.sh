set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"
PYTHON="${PYTHON:-python3}"

[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a


CKPT="$ROOT/checkpoints/hyset/best.pt"
ENCODER="${HYSET_ENCODER_PATH:-$ROOT/models/ToolGen-Qwen2.5-1.5B-Tool-Retriever}"
CORPUS="$ROOT/data/hyset_corpus.json"
RESULT_DIR="$ROOT/result"
LABEL="HYSET"
K_POOL=100
DEVICE="cuda:0"

USE_GREEDY=true
K_POOL_GREEDY=200

RUN_AGENT=false
AGENT_K_RETURN=5
TOOLLLAMA_URL="${TOOLLLAMA_URL:-http://localhost:8000}"
TOOLLLAMA_MODEL="${TOOLLLAMA_MODEL:-ToolLLaMA-2-7b-v2}"
MIRROR_URL="${MIRROR_URL:-http://localhost:8080}"
AGENT_OUT_DIR="$ROOT/outputs/agent_outputs"
TOOL_ROOT_DIR="$ROOT/data/toolenv/tools"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)             CKPT="$2";              shift 2 ;;
    --encoder)          ENCODER="$2";           shift 2 ;;
    --corpus)           CORPUS="$2";            shift 2 ;;
    --result_dir)       RESULT_DIR="$2";        shift 2 ;;
    --label)            LABEL="$2";             shift 2 ;;
    --k_pool)           K_POOL="$2";            shift 2 ;;
    --device)           DEVICE="$2";            shift 2 ;;
    --use_greedy)       USE_GREEDY=true;        shift ;;
    --k_pool_greedy)    K_POOL_GREEDY="$2";     shift 2 ;;
    --run_agent)        RUN_AGENT=true;         shift ;;
    --agent_k_return)   AGENT_K_RETURN="$2";    shift 2 ;;
    --toolllama_url)    TOOLLLAMA_URL="$2";      shift 2 ;;
    --toolllama_model)  TOOLLLAMA_MODEL="$2";   shift 2 ;;
    --mirror_url)       MIRROR_URL="$2";         shift 2 ;;
    --agent_out_dir)    AGENT_OUT_DIR="$2";      shift 2 ;;
    --tool_root_dir)    TOOL_ROOT_DIR="$2";      shift 2 ;;
    *) echo "[inference_hyset.sh] Unknown arg: $1"; exit 1 ;;
  esac
done


log() { echo -e "\n\033[1;36m[inference_hyset] $*\033[0m"; }
die() { echo -e "\033[1;31m[inference_hyset ERROR] $*\033[0m" >&2; exit 1; }
check_file() { [[ -f "$1" ]] || die "Required file not found: $1"; }


check_file "$CKPT"
check_file "$CORPUS"

log "Checkpoint  : $CKPT"
log "Label       : $LABEL"
log "Device      : $DEVICE"
log "k_pool      : $K_POOL  (greedy: $USE_GREEDY, k_pool_greedy: $K_POOL_GREEDY)"

$PYTHON - <<PYEOF
import torch
p = torch.load("$CKPT", map_location="cpu")
ep   = p.get("epoch", "?")
rec  = p.get("recall\@5", float("nan"))
d    = p.get("d", "?")
mmax = p.get("m_max", "?")
print(f"  Checkpoint info: epoch={ep}  train_recall\@5={rec:.4f}  d={d}  m_max={mmax}")
PYEOF


log "Running offline evaluation on all 6 splits …"

EVAL_ARGS=(
  --ckpt            "$CKPT"
  --encoder         "$ENCODER"
  --corpus          "$CORPUS"
  --label           "$LABEL"
  --k_pool          "$K_POOL"
  --k_pool_greedy   "$K_POOL_GREEDY"
  --result_dir      "$RESULT_DIR"
  --agent_out_dir   "$AGENT_OUT_DIR"
  --tool_root_dir   "$TOOL_ROOT_DIR"
  --device          "$DEVICE"
)
[[ "$USE_GREEDY" == true ]] && EVAL_ARGS+=(--use_greedy)

if [[ "$RUN_AGENT" == true ]]; then
  log "Agent inference: ON"
  log "  ToolLLaMA URL   : $TOOLLLAMA_URL"
  log "  ToolLLaMA model : $TOOLLLAMA_MODEL"
  log "  MirrorAPI URL   : $MIRROR_URL"
  EVAL_ARGS+=(
    --run_agent
    --agent_k_return   "$AGENT_K_RETURN"
    --toolllama_url    "$TOOLLLAMA_URL"
    --toolllama_model  "$TOOLLLAMA_MODEL"
    --mirror_url       "$MIRROR_URL"
  )
fi

$PYTHON "$SRC/evaluate_hyset.py" "${EVAL_ARGS[@]}"


OVERALL="$RESULT_DIR/$LABEL/overall/summary.json"
if [[ -f "$OVERALL" ]]; then
  log "=== RESULTS: $LABEL ==="
  LABEL="$LABEL" ROOT="$ROOT" $PYTHON - <<'PYEOF'
import json, os
label   = os.environ["LABEL"]
root    = os.environ["ROOT"]
splits  = ["I1-Inst","I1-Tool","I1-Cate","I2-Inst","I2-Cate","I3-Inst"]
metrics = ["Recall@3","Recall@5","NDCG@3","NDCG@5","COMP@3","COMP@5"]

print(f"\n{'Split':<12}" + "".join(f"{m:>12}" for m in metrics))
print("-" * (12 + 12 * len(metrics)))

for sp in splits:
    path = f"{root}/result/{label}/{sp}/summary.json"
    if not os.path.exists(path):
        print(f"{sp:<12}" + f"{'—':>12}" * len(metrics))
        continue
    with open(path) as f:
        d = json.load(f)
    row = f"{sp:<12}"
    for m in metrics:
        v = d.get(m, None)
        row += f"{v:>12.4f}" if isinstance(v, float) else f"{'—':>12}"
    print(row)

ov_path = f"{root}/result/{label}/overall/summary.json"
if os.path.exists(ov_path):
    with open(ov_path) as f:
        ov = json.load(f)
    print("-" * (12 + 12 * len(metrics)))
    row = f"{'Overall':<12}"
    for m in metrics:
        v = ov.get(m, None)
        row += f"{v:>12.4f}" if isinstance(v, float) else f"{'—':>12}"
    print(row)
PYEOF
fi

log "Done. Results: $RESULT_DIR/$LABEL/"
