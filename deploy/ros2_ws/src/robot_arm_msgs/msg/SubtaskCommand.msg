# SubtaskCommand
#
# agent_node → act_node
# 한 subtask 단위 ACT 실행 명령. 이전 설계의 target_class / target_position 필드는
# 폐기되었고, 자연어 묘사 + 사전 임베딩 벡터로 대체됨.
#
# subtask_id          : agent_node가 발급하는 subtask 식별자 (UUID 또는 단조증가 카운터)
# domain              : 현재 활성 도메인 ("recycling", "tableware" 등)
# target_description  : Gemma look_scene이 산출한 자연어 타겟 묘사.
#                       형식: 영어 반구조화 템플릿
#                         "the {attribute} {object} {spatial_qualifier}"
#                       예: "the blue can on the right side of the table"
#                       길이: 5~15 단어
#                       어휘: deploy/config/label_vocab.yaml과 일치
# language_embedding  : embedding_node가 target_description을 임베딩한 벡터.
#                       모델: paraphrase-multilingual-MiniLM-L12-v2 (D=384)
#                       데이터 수집 / ACT 학습 / 배포 전 구간에서
#                       동일 모델·동일 D 강제.

string subtask_id
string domain
string target_description
float32[] language_embedding
