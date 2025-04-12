from typing import TypedDict, List, Optional
from langchain.schema import Document

class AgentState(TypedDict):
    question: str                        # 사용자의 질문
    classification_score: Optional[str]        # 분류 결과 ("일반", "문서", "웹검색")
    improved_prompt: Optional[str]         # 개선된 질문 (retrieval grader 프롬프트에서 나온 결과)
    score: Optional[str]                 # 관련성 평가 점수 ('1' 또는 '0')
    web_results: Optional[List[str]]     # 웹 검색 결과 (검색된 문서 리스트)
    context: Optional[str]               # 문서 내용을 합친 것
    answer: Optional[str]                # 최종 답변
    relevant_docs: Optional[List[Document]] # 문서 비교를 위해 사용 
    feedback : Optional[str]
    grade_score : Optional[str]
