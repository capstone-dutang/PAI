# ExecutionStatus
#
# act_node → agent_node
# subtask 실행 종료 시 1회 발행. agent_node는 이를 받아 verify_result 호출 또는
# 다음 subtask 발행 여부를 결정한다.
#
# subtask_id      : 대응되는 SubtaskCommand의 subtask_id
# status          : "SUCCESS" | "FAIL" | "TIMEOUT"
#                   robot_arm_pkg.prompts.ExecStatus 와 동일 값 (Gemma tool_result
#                   문자열과 통일 — agent_node 가 case 변환 없이 그대로 전달)
# failure_reason  : 실패/타임아웃 시 사유 (성공 시 빈 문자열)
# confidence      : ACT 자체 신뢰도(있을 경우). 미지원 시 0.0

string subtask_id
string status
string failure_reason
float32 confidence
