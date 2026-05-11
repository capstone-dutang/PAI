#!/usr/bin/env bash
# =============================================================================
# convert_lora_to_gguf.sh
# -----------------------------------------------------------------------------
# 목적: Part C-3 — HuggingFace LoRA(safetensors)를 llama.cpp의 GGUF 어댑터로
#       변환한다. 결과 GGUF는 Jetson의 llama-server가 /lora-adapter HTTP로
#       hot-swap 적재한다.
#
# 사용 예 (README L657-661):
#   bash convert_lora_to_gguf.sh \
#     training/outputs/lora_recycling \
#     models/gemma/adapters/adapter_recycling.gguf
#
# 인자:
#   $1 : Part C-2 train_gemma_qlora.py가 저장한 LoRA 디렉터리
#   $2 : 출력 GGUF 파일 경로
#
# 선행 조건:
#   - llama.cpp 빌드 완료
#   - llama.cpp/convert_lora_to_gguf.py 가용
#   - base 모델의 GGUF가 이미 준비되어 있음 (어댑터-only 변환에 필요할 수 있음)
#
# 미구현 / 결정 필요 항목:
#   - llama.cpp 빌드 버전에 맞는 변환 스크립트 경로 확인
#       (convert_lora_to_gguf.py 또는 convert-lora-to-gguf.py 등 명칭이 빌드 버전마다 다름)
#   - base 모델 GGUF 경로 인자 필요 여부
#   - 양자화 옵션 (변환 단계가 아닌 base 모델의 양자화에 종속)
# =============================================================================

set -euo pipefail

LORA_DIR="${1:?usage: $0 <lora_dir> <output_gguf>}"
OUTPUT_GGUF="${2:?usage: $0 <lora_dir> <output_gguf>}"

# TODO: llama.cpp 변환 스크립트 경로 확정 후 활성화
#
# LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
# python "$LLAMA_CPP_DIR/convert_lora_to_gguf.py" \
#     --base   <base_model_path_or_gguf> \
#     --outfile "$OUTPUT_GGUF" \
#     "$LORA_DIR"

echo "[convert_lora_to_gguf] 미구현 — llama.cpp 변환 스크립트 경로/인자 확정 후 활성화하세요."
echo "  LORA_DIR     = $LORA_DIR"
echo "  OUTPUT_GGUF  = $OUTPUT_GGUF"
exit 1
