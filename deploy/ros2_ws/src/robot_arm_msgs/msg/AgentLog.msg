# AgentLog
#
# agent_node → 디버그 로그 토픽
# Gemma 함수 호출 루프의 각 step 디버그 로그.
# 운영 중 rqt나 콘솔에서 함수 호출 흐름을 확인하기 위한 용도.
#
# timestamp    : ISO 8601 timestamp 또는 ros2 시간 문자열
# thought      : Gemma의 reasoning 텍스트(있을 경우)
# tool_called  : "look_scene" | "execute_subtask" | "verify_result" | ""
# tool_args    : tool 입력 인자 (JSON 직렬화 문자열)
# tool_result  : tool 출력 결과 (JSON 직렬화 문자열)

string timestamp
string thought
string tool_called
string tool_args
string tool_result
