set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"
PYTHON="${PYTHON:-python3}"

[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a

CORPUS="$ROOT/data/hyset_corpus.json"
RESULT_DIR="$ROOT/result"
CACHE_DIR="$ROOT/data/cache"
DEVICE="cuda"
BATCH_SIZE=256
K_POOL=50
SEED=42

BM25_K1=1.5
BM25_B=0.75

ENCODER_TOOLLLAMA="${TOOLLLAMA_IR_MODEL:-$ROOT/models/ToolBench_IR_bert_based_uncased}"
ENCODER_CONTRIEVER="${CONTRIEVER_MODEL:-$ROOT/models/contriever}"
TOOLGEN_MODEL="${TOOLGEN_MODEL:-$ROOT/models/ToolGen-Qwen2.5-1.5B-Tool-Retriever}"

LABEL_RANDOM="Random"
LABEL_BM25="BM25"
LABEL_DENSE="ToolLLaMA_IR"
LABEL_CONTRIEVER="Contriever"
LABEL_TOOLGEN="ToolGen"

METHOD=""

# =============================================================================
# CLI argument parsing
# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --corpus)            CORPUS="$2";              shift 2 ;;
    --result_dir)        RESULT_DIR="$2";          shift 2 ;;
    --cache_dir)         CACHE_DIR="$2";           shift 2 ;;
    --device)            DEVICE="$2";              shift 2 ;;
    --batch_size)        BATCH_SIZE="$2";          shift 2 ;;
    --k_pool)            K_POOL="$2";              shift 2 ;;
    --seed)              SEED="$2";                shift 2 ;;
    --bm25_k1)           BM25_K1="$2";             shift 2 ;;
    --bm25_b)            BM25_B="$2";              shift 2 ;;
    --encoder_toolllama) ENCODER_TOOLLLAMA="$2";   shift 2 ;;
    --encoder_contriever) ENCODER_CONTRIEVER="$2"; shift 2 ;;
    --toolgen_model)     TOOLGEN_MODEL="$2";       shift 2 ;;
    --label_random)      LABEL_RANDOM="$2";        shift 2 ;;
    --label_bm25)        LABEL_BM25="$2";          shift 2 ;;
    --label_dense)       LABEL_DENSE="$2";         shift 2 ;;
    --label_contriever)  LABEL_CONTRIEVER="$2";    shift 2 ;;
    --label_toolgen)     LABEL_TOOLGEN="$2";       shift 2 ;;
    --method)            METHOD="$2";              shift 2 ;;
    *) echo "[baselines_retrieval.sh] Unknown arg: $1"; exit 1 ;;
  esac
done

log() { echo -e "\n\033[1;36m[Baselines] $*\033[0m"; }
die() { echo -e "\033[1;31m[Baselines ERROR] $*\033[0m" >&2; exit 1; }

check_file() { [[ -f "$1" ]] || die "Required file not found: $1"; }

COMMON_ARGS=(
  --corpus     "$CORPUS"
  --result_dir "$RESULT_DIR"
  --cache_dir  "$CACHE_DIR"
  --k_pool     "$K_POOL"
  --device     "$DEVICE"
)

check_file "$CORPUS"
mkdir -p "$RESULT_DIR" "$CACHE_DIR"

should_run() { [[ -z "$METHOD" || "$METHOD" == "$1" ]]; }

if should_run random; then
  log "Running Random baseline …"
  $PYTHON "$SRC/evaluate_baselines.py" \
    --method  random \
    --label   "$LABEL_RANDOM" \
    --seed    "$SEED" \
    "${COMMON_ARGS[@]}"
fi

if should_run bm25; then
  log "Running BM25 baseline (k1=$BM25_K1, b=$BM25_B) …"
  $PYTHON "$SRC/evaluate_baselines.py" \
    --method  bm25 \
    --label   "$LABEL_BM25" \
    --bm25_k1 "$BM25_K1" \
    --bm25_b  "$BM25_B" \
    "${COMMON_ARGS[@]}"
fi

if should_run dense; then
  if [[ ! -d "$ENCODER_TOOLLLAMA" ]]; then
    log "WARNING: ToolLLaMA-IR encoder not found at $ENCODER_TOOLLLAMA — skipping."
  else
    log "Running ToolLLaMA-IR dense baseline …"
    log "  Encoder : $ENCODER_TOOLLLAMA"
    log "  Device  : $DEVICE"
    $PYTHON "$SRC/evaluate_baselines.py" \
      --method     dense \
      --label      "$LABEL_DENSE" \
      --encoder    "$ENCODER_TOOLLLAMA" \
      --batch_size "$BATCH_SIZE" \
      "${COMMON_ARGS[@]}"
  fi
fi
if should_run contriever; then
  if [[ ! -d "$ENCODER_CONTRIEVER" ]]; then
    log "Contriever model not found at $ENCODER_CONTRIEVER — skipping."
    log "  Set CONTRIEVER_MODEL env var or pass --encoder_contriever to enable."
  else
    log "Running Contriever dense baseline …"
    log "  Encoder : $ENCODER_CONTRIEVER"
    log "  Device  : $DEVICE"
    $PYTHON "$SRC/evaluate_baselines.py" \
      --method     dense \
      --label      "$LABEL_CONTRIEVER" \
      --encoder    "$ENCODER_CONTRIEVER" \
      --batch_size "$BATCH_SIZE" \
      "${COMMON_ARGS[@]}"
  fi
fi

if should_run toolgen; then
  if [[ ! -d "$TOOLGEN_MODEL" ]]; then
    log "ToolGen model not found at $TOOLGEN_MODEL — skipping."
    log "  Pass --toolgen_model to specify the path."
  else
    log "Running ToolGen baseline …"
    log "  Model  : $TOOLGEN_MODEL"
    log "  Device : $DEVICE"
    $PYTHON "$SRC/evaluate_baselines.py" \
      --method        toolgen \
      --label         "$LABEL_TOOLGEN" \
      --toolgen_model "$TOOLGEN_MODEL" \
      "${COMMON_ARGS[@]}"
  fi
fi

SUMMARY_OUT="$RESULT_DIR/baselines_summary.txt"
log "=== BASELINE RETRIEVAL SUMMARY ==="
RESULT_DIR="$RESULT_DIR" LABEL_RANDOM="$LABEL_RANDOM" \
LABEL_BM25="$LABEL_BM25" LABEL_DENSE="$LABEL_DENSE" \
LABEL_CONTRIEVER="$LABEL_CONTRIEVER" LABEL_TOOLGEN="$LABEL_TOOLGEN" \
$PYTHON - <<'PYEOF' | tee "$SUMMARY_OUT"
import json, os

result_dir      = os.environ.get("RESULT_DIR",       "result")
label_random    = os.environ.get("LABEL_RANDOM",     "Random")
label_bm25      = os.environ.get("LABEL_BM25",       "BM25")
label_dense     = os.environ.get("LABEL_DENSE",      "ToolLLaMA_IR")
label_contriever= os.environ.get("LABEL_CONTRIEVER", "Contriever")
label_toolgen   = os.environ.get("LABEL_TOOLGEN",    "ToolGen")

splits  = ["I1-Inst","I1-Tool","I1-Cate","I2-Inst","I2-Cate","I3-Inst","Overall"]
metrics = ["Recall@3","Recall@5","NDCG@3","NDCG@5","COMP@3","COMP@5"]
labels  = [label_random, label_bm25, label_dense, label_contriever, label_toolgen]

def load_split(label, split):
    if split == "Overall":
        p = os.path.join(result_dir, label, "overall", "summary.json")
    else:
        p = os.path.join(result_dir, label, split, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

for label in labels:
    overall_path = os.path.join(result_dir, label, "overall", "summary.json")
    if not os.path.exists(overall_path):
        continue
    print(f"\n{'─'*80}")
    print(f"  {label}")
    print(f"{'─'*80}")
    print(f"  {'Split':<12}" + "".join(f"{m:>12}" for m in metrics))
    print("  " + "-"*(12 + 12*len(metrics)))
    for sp in splits:
        d = load_split(label, sp)
        row = f"  {sp:<12}"
        for m in metrics:
            v = (d or {}).get(m)
            row += f"{v:>12.4f}" if isinstance(v, float) else f"{'—':>12}"
        print(row)

PYEOF

log "Summary saved → $SUMMARY_OUT"
log "Done. All results under: $RESULT_DIR/"
