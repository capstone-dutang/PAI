# SubtaskCommand
#
# agent_node → act_node
# 한 subtask 단위 ACT 실행 명령. 이전 설계의 target_class / target_position 필드는
# 폐기되었고, 자연어 묘사 + 사전 임베딩 벡터로 대체됨.
#
# subtask_id          : agent_node가 발급하는 subtask 식별자 (UUID 또는 단조증가 카운터)
# domain              : 현재 활성 도메인 ("library", "recycling" 등)
# target_description  : Gemma look_scene이 산출한 자연어 타겟 묘사
#                       (예: "blue can on the right side of the table")
# language_embedding  : embedding_node가 target_description을 임베딩한 벡터.
#                       길이 D는 embedding_node가 사용 중인 모델에 따라 결정됨.

string subtask_id
string domain
string target_description
float32[] language_embedding
