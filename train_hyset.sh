set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"
PYTHON="${PYTHON:-python3}"

[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a


# Model architecture
D=1536                       # API embedding dimension 
M_MAX=5                      # maximum set order 

# Training
EPOCHS=10
BATCH_SIZE=32
LR=1e-4                      # HYSET-specific params learning rate
LR_ENCODER=0                 # freeze g_psi 
ETA=0.3                      
LAMBDA_BETA=0.0              
LAMBDA_D=0.01               
B=64                         
WARMUP_RATIO=0.06            
PATIENCE=3                   
VAL_FREQ=1                   
SEED=42
REFRESH_HARD_NEG=false

# Encoder / ToolGen init
ENCODER_TYPE="qwen"         
TOOLGEN_INIT=true           
TOOLGEN_PATH="${TOOLGEN_MODEL:-$ROOT/models/ToolGen-Qwen2.5-1.5B-Tool-Retriever}"
N_BASE_VOCAB=151665

# Inference
K_POOL=100                   

# Paths
ENCODER="${HYSET_ENCODER_PATH:-$ROOT/models/ToolGen-Qwen2.5-1.5B-Tool-Retriever}"
CKPT_DIR="$ROOT/checkpoints/hyset"
CORPUS="$ROOT/data/hyset_corpus.json"
REWARD_CACHE="$ROOT/data/reward_cache.json"
HARD_NEG="$ROOT/data/hard_neg_cache.json"

# Stage skip flags
SKIP_CORPUS=false
SKIP_HARD_NEG=false

RESUME=""


while [[ $# -gt 0 ]]; do
  case "$1" in
    --d)              D="$2";              shift 2 ;;
    --m_max)          M_MAX="$2";          shift 2 ;;
    --epochs)         EPOCHS="$2";         shift 2 ;;
    --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
    --lr)             LR="$2";             shift 2 ;;
    --lr_encoder)     LR_ENCODER="$2";     shift 2 ;;
    --eta)            ETA="$2";            shift 2 ;;
    --lambda_beta)    LAMBDA_BETA="$2";    shift 2 ;;
    --lambda_d)       LAMBDA_D="$2";       shift 2 ;;
    --B)              B="$2";              shift 2 ;;
    --warmup_ratio)   WARMUP_RATIO="$2";   shift 2 ;;
    --patience)       PATIENCE="$2";       shift 2 ;;
    --val_freq)       VAL_FREQ="$2";       shift 2 ;;
    --seed)           SEED="$2";           shift 2 ;;
    --k_pool)         K_POOL="$2";         shift 2 ;;
    --encoder)        ENCODER="$2";        shift 2 ;;
    --encoder_type)   ENCODER_TYPE="$2";   shift 2 ;;
    --toolgen_init)   TOOLGEN_INIT=true;   shift ;;
    --toolgen_path)   TOOLGEN_PATH="$2";   shift 2 ;;
    --n_base_vocab)   N_BASE_VOCAB="$2";   shift 2 ;;
    --ckpt_dir)       CKPT_DIR="$2";       shift 2 ;;
    --label)          CKPT_DIR="$ROOT/checkpoints/$2"; shift 2 ;;
    --resume)         RESUME="$2";         shift 2 ;;
    --refresh_hard_neg) REFRESH_HARD_NEG=true; shift ;;
    --skip_corpus)    SKIP_CORPUS=true;    shift ;;
    --skip_hard_neg)  SKIP_HARD_NEG=true;  shift ;;
    *) echo "[train_hyset.sh] Unknown arg: $1"; exit 1 ;;
  esac
done

log() { echo -e "\n\033[1;36m[train_hyset] $*\033[0m"; }
die() { echo -e "\033[1;31m[train_hyset ERROR] $*\033[0m" >&2; exit 1; }
check_file() { [[ -f "$1" ]] || die "Required file not found: $1"; }

if [[ "$SKIP_CORPUS" == false ]]; then
  if [[ -f "$CORPUS" ]]; then
    N=$($PYTHON -c "import json; d=json.load(open('$CORPUS')); print(len(d))" 2>/dev/null || echo 0)
    log "Corpus already exists ($N APIs) — skipping build."
  else
    log "Stage 0: Building API corpus …"
    $PYTHON "$SRC/build_hyset_corpus.py"
  fi
  check_file "$CORPUS"
fi

if [[ "$SKIP_HARD_NEG" == false ]]; then
  if [[ -f "$HARD_NEG" ]]; then
    N=$($PYTHON -c "import json; d=json.load(open('$HARD_NEG')); print(len(d))" 2>/dev/null || echo 0)
    log "Hard neg cache already exists ($N entries) — skipping precompute."
  else
    log "Stage 1: Precomputing hard negatives …"
    $PYTHON "$SRC/precompute_hard_negatives.py" \
      --corpus   "$CORPUS" \
      --encoder  "$ENCODER" \
      --out_path "$HARD_NEG" \
      --top_k    "$K_POOL"
  fi
  check_file "$HARD_NEG"
fi

if [[ ! -f "$REWARD_CACHE" ]]; then
  log "[WARN] Reward cache not found at $REWARD_CACHE — L_agent will be zero."
  log "       Run: python src/precompute_rewards.py to build it."
fi

log "Stage 2: Training HYSET"
log "  Architecture : d=$D  m_max=$M_MAX  encoder=$ENCODER_TYPE"
log "  Training     : epochs=$EPOCHS  bs=$BATCH_SIZE  B=$B"
log "  LR           : lr=$LR  lr_encoder=$LR_ENCODER  warmup_ratio=$WARMUP_RATIO"
log "  Loss weights : eta=$ETA  lambda_D=$LAMBDA_D"
log "  Early stop   : patience=$PATIENCE  val_freq=$VAL_FREQ"
log "  Checkpoint   : $CKPT_DIR"
[[ "$TOOLGEN_INIT" == true ]] && log "  ToolGen init : $TOOLGEN_PATH"

N_GPU=$($PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
log "  GPUs available: $N_GPU"

TRAIN_ARGS=(
  --corpus        "$CORPUS"
  --hard_neg      "$HARD_NEG"
  --ckpt_dir      "$CKPT_DIR"
  --encoder       "$ENCODER"
  --encoder_type  "$ENCODER_TYPE"
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
  --warmup_ratio  "$WARMUP_RATIO"
  --patience      "$PATIENCE"
  --val_freq      "$VAL_FREQ"
  --seed          "$SEED"
  --n_base_vocab  "$N_BASE_VOCAB"
)

[[ -f "$REWARD_CACHE" ]] && TRAIN_ARGS+=(--reward_cache "$REWARD_CACHE")
[[ "$REFRESH_HARD_NEG" == true ]] && TRAIN_ARGS+=(--refresh_hard_neg)
[[ "$TOOLGEN_INIT"     == true ]] && TRAIN_ARGS+=(--toolgen_init --toolgen_path "$TOOLGEN_PATH")
[[ -n "$RESUME" ]] && TRAIN_ARGS+=(--resume "$RESUME")

$PYTHON "$SRC/train_hyset.py" "${TRAIN_ARGS[@]}"

log "Training complete."
BEST="$CKPT_DIR/best.pt"
if [[ -f "$BEST" ]]; then
  RECALL=$($PYTHON -c "
import torch
p = torch.load('$BEST', map_location='cpu')
print(f\"best Recall@5 = {p.get('recall@5', float('nan')):.4f}  (epoch {p.get('epoch','?')})\")
" 2>/dev/null || echo "?")
  log "Best checkpoint: $RECALL → $BEST"
fi
log "Run inference_hyset.sh to evaluate on all 6 test splits."
