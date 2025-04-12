from langchain.prompts import ChatPromptTemplate, PromptTemplate

class PromptChain:
    def __init__(self):
        
        # 일반 질의 구간 
        self.llama_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 질문에 맞는 답변을 생성하는 AI이다. 모든 응답은 반드시 한국어로 작성하며 자세하고 정확하게 답변해라."),
            ("user", "{question}")
        ])
        self.general_grader_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 이전 답변자가 만든 답변의 품질을 평가하는 AI 전문가이다."),
            ("user", """다음 질문에 따른 답변을 읽고 평가해주세요:

            질문: {question}
            
            답변 : {answer}

            - 답변이 구체적이고 명확하면 1점을 주세요.
            - 답변이 모호하거나 개선이 필요하면 0점을 주세요.

            JSON 형식으로 반환해주세요.
            {{
            "grade_score": "1" 또는 "0",
            "feedback": "모호한 답변이 발생할 시 개선된 답변을 작성해주세요."
            }}""")
        ])
        

        
        # 응답 구분 프롬프트 
        self.classfication_question_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 사용자 질문에 대해 적절한 답변 방향을 결정하는 전문가입니다."),
            ("user", """아래 질문을 분석하여 가장 적합한 답변 유형을 결정해주세요. 각 유형의 정의는 다음과 같습니다.

        1. 일반 답변: 질문에 대해 상식이나 단순 일반 상식으로만 답변할 수 있는 경우.
        2. 문서 검색: 최신 정보나 구체적인 사실 확인을 위해 문서 검색이 필요한 경우.
        3. 웹 검색 : 최신 정보 등 문서의 내용과 비교하기 위해 웹 검색이 필요한 경우. 
        
        주어진 질문을 분석한 후, 아래 JSON 형식에 따라 하나의 유형에 해당하는 코드 값을 반환해주세요.

        예시 형식:
        {{
            "classification_score": "0"  // "0" for 일반, "1" for 문서, "2" for 웹검색
        }}
        
        예시: 
        - "적금이 뭐야?" -> "일반"
        - "한화오션 주가는 얼마야?" -> "웹검색" 
        - "반도체 동향 분석해줘" -> "문서"

        질문: {question}
        """)
        ])
        
        self.document_prompt = PromptTemplate(
            template="""
            당신은 정확하고 신뢰할 수 있는 답변을 제공할 수 있는 금융전문가 입니다.
            아래 제공된 문맥을 기반으로 질문에 대한 답변을 작성하세요. 
            
            다음 지침을 반드시 따라주세요:
            1. 답변은 반드시 한국어로 작성해야 합니다.  
            2. 문서의 주요 내용을 항목별로 정리하여 답변하세요. 
            3. documents에 있는 정보만을 사용해야 하며, 추측하거나 새로운 정보를 생성하지 마세요.
            4. 가능한 경우 원문의 문장을 유지하여 정보를 전달하세요.  
            5. 불확실한 내용이 포함된 경우, `추가 정보가 필요합니다.`라고 답변하세요.  
            6. 질문에 대한 명확한 답변을 제공할 수 없으면, '주어진 정보로는 답변하기 어렵습니다.'라고만 말하세요.  
            7. 답변은 간결하고 명확하게 작성해야 합니다. 

            질문: {question}  
            참고 문서: {documents}  
            답변:  
            """,
            input_variables=["question", "documents"],
        )
        
        self.retrieval_grader_prompt = PromptTemplate(
            template="""
            당신은 주어진 문서가 질문과 얼마나 관련이 있는지 평가하는 전문가입니다.
            
            - 문서가 질문과 관련이 있으면 1점을 부여합니다.
            - 문서가 질문과 관련이 없으면 0점을 부여합니다.
            
            또한, 만약 문서가 질문과 관련성이 낮다고 판단되면, 더 관련성 높은 결과를 얻기 위해 개선할 수 있는 질문(프롬프트)의 예시도 함께 작성해주세요.
            
            질문: {question}  
            문서 내용: {documents}  
            
            JSON 형식으로 결과를 반환해주세요.  
            예시 형식:
            {{
            "score": "1" 또는 "0",
            "improved_prompt": "개선된 질문 내용 (관련성이 낮은 경우에만 작성, 관련성이 높으면 빈 문자열로)"
            }}
            """,
            input_variables=["question", "documents"],
        )
        
        self.web_report_prompt = PromptTemplate(
            template="""
            당신은 최신 정보를 바탕으로 정확하고 포괄적인 보고서를 작성하는 전문가입니다.
            
            아래의 지침에 따라 보고서를 작성해주세요:
            1. 보고서는 반드시 한국어로 작성되어야 합니다.
            2. 서론에서는 질문의 배경과 목적을 간략히 설명합니다.
            3. 본론에서는 웹 검색 결과를 토대로 주요 정보, 데이터, 분석 결과를 체계적으로 정리합니다.
            4. 결론에서는 주요 발견 사항과 제언을 명확하게 제시합니다.
            5. 각 정보의 출처를 명시하고, 최신 정보를 우선시하여 반영합니다.
            
            질문: {question}
            웹 검색 결과: {documents}
            
            보고서:
            """,
            input_variables=["question", "documents"],
        )


        
        



        
        
