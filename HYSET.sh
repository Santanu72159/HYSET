set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"
PYTHON="${PYTHON:-python3}"

# ── load env ──────────────────────────────────────────────────────────────────
[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a


D=1536                       # API embedding dimension
M_MAX=5                      # maximum set order 

# Training
EPOCHS=3
BATCH_SIZE=32
LR=1e-4                      # HYSET-specific params learning rate
LR_ENCODER=0                 # freeze g_psi 
ETA=0.3                      # L_agent weight (η)
LAMBDA_BETA=0.01             # β regularisation
LAMBDA_D=0.01                # D regularisation
B=10                         # candidate set size per training query
SEED=42
REFRESH_HARD_NEG=false       

# Inference
K_POOL=100                  
K_RETURN=5                   

# Paths
ENCODER="${HYSET_ENCODER_PATH:-$ROOT/models/ToolGen-Qwen2.5-1.5B-Tool-Retriever}"
CKPT_DIR="$ROOT/checkpoints/hyset"
CORPUS="$ROOT/data/hyset_corpus.json"
REWARD_CACHE="$ROOT/data/reward_cache.json"
HARD_NEG="$ROOT/data/hard_neg_cache.json"
RESULT_DIR="$ROOT/result"
LABEL="HYSET"

# Precompute rewards
REWARD_WORKERS=10           

TOOLLLAMA_URL="${TOOLLLAMA_URL:-http://localhost:8000}"
TOOLLLAMA_MODEL="${TOOLLLAMA_MODEL:-ToolLLaMA-2-7b-v2}"
MIRROR_URL="${MIRROR_URL:-http://localhost:8080}"
AGENT_OUT_DIR="$ROOT/outputs/agent_outputs"

SKIP_CORPUS=false
SKIP_REWARDS=false
SKIP_HARD_NEG=false
SKIP_TRAIN=false
SKIP_EVAL=false
RUN_AGENT=false

# Checkpoint to use for eval 
EVAL_CKPT="$CKPT_DIR/best.pt"
RESUME=""                    


while [[ $# -gt 0 ]]; do
  case "$1" in
    --d)            D="$2";           shift 2 ;;
    --m_max)        M_MAX="$2";       shift 2 ;;
    --epochs)       EPOCHS="$2";      shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";  shift 2 ;;
    --lr)           LR="$2";          shift 2 ;;
    --lr_encoder)   LR_ENCODER="$2";  shift 2 ;;
    --eta)          ETA="$2";         shift 2 ;;
    --lambda_beta)  LAMBDA_BETA="$2"; shift 2 ;;
    --lambda_d)     LAMBDA_D="$2";    shift 2 ;;
    --B)            B="$2";           shift 2 ;;
    --seed)         SEED="$2";        shift 2 ;;
    --k_pool)       K_POOL="$2";      shift 2 ;;
    --k_return)     K_RETURN="$2";    shift 2 ;;
    --label)        LABEL="$2";       shift 2 ;;
    --encoder)      ENCODER="$2";     shift 2 ;;
    --ckpt_dir)     CKPT_DIR="$2";    shift 2 ;;
    --eval_ckpt)    EVAL_CKPT="$2";   shift 2 ;;
    --resume)       RESUME="$2";      shift 2 ;;
    --reward_workers) REWARD_WORKERS="$2"; shift 2 ;;
    --refresh_hard_neg) REFRESH_HARD_NEG=true; shift ;;
    --skip_corpus)  SKIP_CORPUS=true;  shift ;;
    --skip_rewards) SKIP_REWARDS=true; shift ;;
    --skip_hard_neg) SKIP_HARD_NEG=true; shift ;;
    --skip_train)   SKIP_TRAIN=true;  shift ;;
    --skip_eval)    SKIP_EVAL=true;   shift ;;
    --run_agent)    RUN_AGENT=true;   shift ;;
    --toolllama_url)   TOOLLLAMA_URL="$2";    shift 2 ;;
    --toolllama_model) TOOLLLAMA_MODEL="$2";  shift 2 ;;
    --mirror_url)      MIRROR_URL="$2";       shift 2 ;;
    --agent_out_dir)   AGENT_OUT_DIR="$2";    shift 2 ;;
    *) echo "[HYSET.sh] Unknown arg: $1"; exit 1 ;;
  esac
done

log() { echo -e "\n\033[1;36m[HYSET] $*\033[0m"; }
die() { echo -e "\033[1;31m[HYSET ERROR] $*\033[0m" >&2; exit 1; }

check_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

if [[ "$SKIP_CORPUS" == false ]]; then
  if [[ -f "$CORPUS" ]]; then
    N=$($PYTHON -c "import json; d=json.load(open('$CORPUS')); print(len(d))" 2>/dev/null || echo 0)
    log "Corpus already exists ($N APIs). Skipping build. (use --skip_corpus to suppress this check)"
  else
    log "Stage 0: Building API corpus …"
    $PYTHON "$SRC/build_hyset_corpus.py"
  fi
  check_file "$CORPUS"
fi

if [[ "$SKIP_REWARDS" == false ]]; then
  log "Stage 1: Precomputing agent rewards …"
  log "  Judge model : $JUDGE_MODEL"
  log "  Workers     : $REWARD_WORKERS"
  $PYTHON "$SRC/precompute_rewards.py" \
    --workers  "$REWARD_WORKERS" \
    --api_key  "$JUDGE_API_KEY" \
    --api_base "" \
    --model    "$JUDGE_MODEL"
  check_file "$REWARD_CACHE"
fi

if [[ "$SKIP_HARD_NEG" == false ]]; then
  log "Stage 2: Precomputing hard negatives (ToolBench_IR) …"
  $PYTHON "$SRC/precompute_hard_negatives.py" \
    --corpus   "$CORPUS" \
    --encoder  "$ENCODER" \
    --out_path "$HARD_NEG" \
    --top_k    "$K_POOL"
  check_file "$HARD_NEG"
fi

if [[ "$SKIP_TRAIN" == false ]]; then
  log "Stage 3: Training HYSET …"
  log "  d=$D  m_max=$M_MAX  epochs=$EPOCHS  bs=$BATCH_SIZE"
  log "  lr=$LR  lr_encoder=$LR_ENCODER  eta=$ETA"
  log "  lambda_beta=$LAMBDA_BETA  lambda_D=$LAMBDA_D  B=$B"

  TRAIN_ARGS=(
    --corpus        "$CORPUS"
    --reward_cache  "$REWARD_CACHE"
    --hard_neg      "$HARD_NEG"
    --ckpt_dir      "$CKPT_DIR"
    --encoder       "$ENCODER"
    --d             "$D"
    --m_max         "$M_MAX"
    --B             "$B"
    --epochs        "$EPOCHS"
    --batch_size    "$BATCH_SIZE"
    --lr            "$LR"
    --lr_encoder    "$LR_ENCODER"
    --eta           "$ETA"
    --lambda_beta   "$LAMBDA_BETA"
    --lambda_D      "$LAMBDA_D"
    --k_pool        "$K_POOL"
    --seed          "$SEED"
  )

  [[ "$REFRESH_HARD_NEG" == true ]] && TRAIN_ARGS+=(--refresh_hard_neg)
  [[ -n "$RESUME" ]] && TRAIN_ARGS+=(--resume "$RESUME")

  $PYTHON "$SRC/train_hyset.py" "${TRAIN_ARGS[@]}"

  # Verify checkpoint
  check_file "$EVAL_CKPT"
  log "Training done. Using $EVAL_CKPT for evaluation."
fi

if [[ "$SKIP_EVAL" == false ]]; then
  check_file "$EVAL_CKPT"

  log "Stage 4: Evaluating offline metrics on all 6 splits …"

  EVAL_ARGS=(
    --ckpt           "$EVAL_CKPT"
    --encoder        "$ENCODER"
    --corpus         "$CORPUS"
    --label          "$LABEL"
    --k_pool         "$K_POOL"
    --result_dir     "$RESULT_DIR"
    --agent_out_dir  "$AGENT_OUT_DIR"
    --tool_root_dir  "$ROOT/data/toolenv/tools"
    --device         "cpu"
  )

  if [[ "$RUN_AGENT" == true ]]; then
    log "Stage 5: Agent inference via official StableToolBench pipeline"
    log "  ToolLLaMA URL   : $TOOLLLAMA_URL"
    log "  ToolLLaMA model : $TOOLLLAMA_MODEL"
    log "  MirrorAPI URL   : $MIRROR_URL"
    log "  Tool root dir   : $ROOT/data/toolenv/tools"
    log "  Agent outputs   : $AGENT_OUT_DIR/$LABEL/"
    log "  Pipeline        : rapidapi_wrapper + DFS_woFilter_w2 (unchanged)"
    EVAL_ARGS+=(
      --run_agent
      --toolllama_url   "$TOOLLLAMA_URL"
      --toolllama_model "$TOOLLLAMA_MODEL"
      --mirror_url      "$MIRROR_URL"
    )
  fi

  $PYTHON "$SRC/evaluate_hyset.py" "${EVAL_ARGS[@]}"
fi

OVERALL="$RESULT_DIR/$LABEL/overall/summary.json"
if [[ -f "$OVERALL" ]]; then
  log "=== RESULTS: $LABEL ==="
  LABEL="$LABEL" ROOT="$ROOT" $PYTHON - <<'PYEOF'
import json, os
label   = os.environ.get("LABEL", "HYSET")
root    = os.environ.get("ROOT", ".")
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

# Overall
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

# Agent inference summary (ToolEval-format per-query JSON files)
agent_dir = f"{root}/outputs/agent_outputs/{label}"
if os.path.isdir(agent_dir):
    print(f"\n  Agent outputs ({label})  [DFS_woFilter_w2]:")
    total_q = total_ans = 0
    for sp in splits:
        sp_dir = f"{agent_dir}/{sp}"
        if not os.path.isdir(sp_dir):
            continue
        n_q = n_ans = 0
        for fname in os.listdir(sp_dir):
            if not fname.endswith("_DFS_woFilter_w2.json"):
                continue
            n_q += 1
            try:
                with open(os.path.join(sp_dir, fname)) as fh:
                    rec = json.load(fh)
                if rec.get("answer_generation", {}).get("valid_data", False):
                    n_ans += 1
            except Exception:
                pass
        total_q += n_q
        total_ans += n_ans
        print(f"    {sp:<12}  {n_ans}/{n_q} answered")
    if total_q:
        print(f"    {'Total':<12}  {total_ans}/{total_q} answered")
PYEOF
fi

log "Done. Results: $RESULT_DIR/$LABEL/"
if [[ "$RUN_AGENT" == true ]]; then
  log "Agent outputs: $AGENT_OUT_DIR/$LABEL/"
fi
