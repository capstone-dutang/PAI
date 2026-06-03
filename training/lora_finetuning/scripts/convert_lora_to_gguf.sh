#!/usr/bin/env bash
# =============================================================================
# convert_lora_to_gguf.sh
# -----------------------------------------------------------------------------
# train_gemma_qlora.py 가 저장한 LoRA 디렉터리 (HuggingFace safetensors) 를
# llama.cpp 의 GGUF 어댑터 (.gguf) 로 변환한다. 결과 GGUF 는 Jetson 의
# llama-server 가 /lora-adapter HTTP 로 hot-swap 적재한다.
#
# 사용 예 (외부 학습 PC):
#   # 1회 환경 준비 — llama.cpp clone + 변환에 필요한 파이썬 의존성 설치
#   git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
#   pip install -r ~/llama.cpp/requirements/requirements-convert_lora_to_gguf.txt
#   export LLAMA_CPP_DIR=~/llama.cpp
#
#   # 도메인별 변환
#   bash training/lora_finetuning/scripts/convert_lora_to_gguf.sh \
#     training/lora_finetuning/outputs/lora_recycling \
#     models/gemma/adapters/adapter_recycling.gguf
#   bash training/lora_finetuning/scripts/convert_lora_to_gguf.sh \
#     training/lora_finetuning/outputs/lora_tableware \
#     models/gemma/adapters/adapter_tableware.gguf
#
# 인자:
#   $1 : train_gemma_qlora.py 가 저장한 LoRA 디렉터리 (adapter_config.json +
#        adapter_model.safetensors 포함)
#   $2 : 출력 GGUF 파일 경로 (없는 디렉터리는 자동 생성)
#
# 환경 변수:
#   LLAMA_CPP_DIR  : llama.cpp clone 경로 (기본 ~/llama.cpp)
#   BASE_MODEL     : base 모델 위치 — 로컬 디렉터리 path 또는 HF repo id
#                    (기본 'unsloth/gemma-4-E2B-it'). HF repo id 입력 시
#                    자동으로 snapshot_download → 로컬 path 로 변환.
#                    llama.cpp 변환기 본체는 로컬 디렉터리만 허용함.
#   HF_BASE_CACHE  : 자동 다운로드 받은 base 모델을 저장할 부모 디렉터리
#                    (기본 ~/models/hf). 실제 path = $HF_BASE_CACHE/<repo_basename>
#
# 학습 env 보호 (중요):
#   본 스크립트는 학습용 env (torch/transformers) 와 충돌할 수 있는 의존성을
#   설치하지 않는다. 그러나 README 의 '1회 환경 준비' 단계 (
#   pip install -r llama.cpp/requirements/...) 를 학습 venv 안에서 그대로
#   실행하면 torch/torchvision/transformers 가 강제 업그레이드되어 학습 env
#   가 깨질 수 있다. 변환 전용 venv 사용을 권장 — 자세한 절차는
#   training/README.md "2-8. GGUF 변환" 참조.
#
# llama.cpp 변환 스크립트 정책:
#   llama.cpp 는 LoRA 변환용 스크립트 이름이 빌드 버전마다 다르다 (
#   convert_lora_to_gguf.py / convert-lora-to-gguf.py). 본 스크립트는 양쪽
#   파일명을 모두 탐색해 존재하는 것을 사용. 둘 다 없으면 에러.
#
#   --base 인자: HuggingFace base 모델 경로/repo id 를 받는다. LoRA adapter
#   의 dim/quant 정보를 base 와 맞추기 위해 필요. 기본값은 train 시 사용한
#   `unsloth/gemma-4-E2B-it` 으로 가정 (CLI 인자 BASE_MODEL 로 override).
#
# Jetson 배치:
#   생성된 GGUF 를 Jetson 의 ~/models/gemma/adapters/ 로 복사. 파일명은
#   adapter_<domain>.gguf 규약 (deploy/config/domain_map.yaml 의 domain 키와
#   일치) — llama_server_manager 가 이 경로를 build 해서 hot-swap 호출.
# =============================================================================

set -euo pipefail

LORA_DIR="${1:?usage: $0 <lora_dir> <output_gguf>}"
OUTPUT_GGUF="${2:?usage: $0 <lora_dir> <output_gguf>}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
BASE_MODEL="${BASE_MODEL:-unsloth/gemma-4-E2B-it}"
HF_BASE_CACHE="${HF_BASE_CACHE:-$HOME/models/hf}"

# ── 인자/환경 검증 ────────────────────────────────────────────────
if [[ ! -d "$LORA_DIR" ]]; then
  echo "[convert_lora_to_gguf] ERROR: LoRA 디렉터리 없음: $LORA_DIR" >&2
  exit 1
fi
for required in adapter_config.json adapter_model.safetensors; do
  if [[ ! -f "$LORA_DIR/$required" ]]; then
    echo "[convert_lora_to_gguf] ERROR: $LORA_DIR/$required 없음 " \
         "(train_gemma_qlora.py 출력이 맞는지 확인)" >&2
    exit 1
  fi
done

if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
  echo "[convert_lora_to_gguf] ERROR: llama.cpp 디렉터리 없음: $LLAMA_CPP_DIR" >&2
  echo "  git clone https://github.com/ggerganov/llama.cpp.git $LLAMA_CPP_DIR" >&2
  echo "  또는 LLAMA_CPP_DIR=<path> 환경 변수로 지정" >&2
  exit 1
fi

# 변환 스크립트 탐색 (underscore / hyphen 양쪽)
CONVERT_SCRIPT=""
for cand in convert_lora_to_gguf.py convert-lora-to-gguf.py; do
  if [[ -f "$LLAMA_CPP_DIR/$cand" ]]; then
    CONVERT_SCRIPT="$LLAMA_CPP_DIR/$cand"
    break
  fi
done
if [[ -z "$CONVERT_SCRIPT" ]]; then
  echo "[convert_lora_to_gguf] ERROR: llama.cpp 에 변환 스크립트 없음" >&2
  echo "  탐색 위치: $LLAMA_CPP_DIR/{convert_lora_to_gguf.py,convert-lora-to-gguf.py}" >&2
  echo "  llama.cpp 최신 버전으로 pull / 빌드 후 재시도." >&2
  exit 1
fi

# BASE_MODEL 처리 — llama.cpp 변환기는 로컬 디렉터리 path 만 허용.
# HF repo id (= 디렉터리 아님) 입력 시 자동으로 snapshot_download → 로컬 path 변환.
if [[ ! -d "$BASE_MODEL" ]]; then
  LOCAL_BASE="$HF_BASE_CACHE/$(basename "$BASE_MODEL")"
  if [[ ! -f "$LOCAL_BASE/config.json" ]]; then
    echo "[convert_lora_to_gguf] base 모델 로컬에 없음 — HF 에서 다운로드"
    echo "  repo_id     = $BASE_MODEL"
    echo "  local_dir   = $LOCAL_BASE"
    mkdir -p "$LOCAL_BASE"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$BASE_MODEL', local_dir='$LOCAL_BASE',
                  allow_patterns=['config.json','tokenizer*','*.json',
                                  'special_tokens_map.json','generation_config.json'])
"
  else
    echo "[convert_lora_to_gguf] base 모델 로컬 캐시 사용: $LOCAL_BASE"
  fi
  BASE_MODEL="$LOCAL_BASE"
fi

# 출력 디렉터리 자동 생성
OUTPUT_DIR=$(dirname "$OUTPUT_GGUF")
mkdir -p "$OUTPUT_DIR"

echo "[convert_lora_to_gguf] start"
echo "  LORA_DIR     = $LORA_DIR"
echo "  OUTPUT_GGUF  = $OUTPUT_GGUF"
echo "  LLAMA_CPP    = $LLAMA_CPP_DIR"
echo "  BASE_MODEL   = $BASE_MODEL"
echo "  CONVERT      = $CONVERT_SCRIPT"

python "$CONVERT_SCRIPT" \
  --base    "$BASE_MODEL" \
  --outfile "$OUTPUT_GGUF" \
  "$LORA_DIR"

echo "[convert_lora_to_gguf] done → $OUTPUT_GGUF"
ls -lh "$OUTPUT_GGUF"
