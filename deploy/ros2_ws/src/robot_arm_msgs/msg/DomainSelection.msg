# DomainSelection
#
# agent_node → llama_server_manager
# 도메인 분류 결과를 알려 LoRA hot-swap을 트리거한다.
#
# domain : 활성화할 도메인 ("recycling", "tableware" 등)
#          adapter_<domain>.gguf 파일명 규약과 매칭됨
# scale  : LoRA 스케일 (llama-server /lora-adapter API 인자, 0.0~1.0)

string domain
float32 scale
