from state.agent_state import AgentState
from tools.document_retrieve import Retriever
from tools.search_retrieve import WebSearch
from langchain.schema import Document

class Node:
    
    def classification_node(self, state: AgentState, chain):
        print("\n==== 질문 분류 ====\n")
        question = state["question"]
        
        # 체인을 호출하여 분류 결과를 생성합니다.
        generate_classification = chain.invoke({"question": question})
        result = generate_classification.get("classification_score")
        
        # AgentState의 "classification" 필드를 업데이트합니다.
        state["classification_score"] = result
        
        # 전체 state를 반환합니다.
        return state
        
    def general_node(self, state: AgentState, chain):
        question = state["question"]
        classification = state["classification_score"]

        if classification == "0":
            print("\n==== 일반 답변 ====\n")
            response = chain.invoke({"question": question})
            state["answer"] = response  
        else:
            print("일반 답변 아님:", classification)

        return state  

    def document_retriever(self, state: AgentState):
        print("\n==== 문서 검색 ====\n")
        classification_score = state.get("classification_score")
        
        if classification_score == "1":
            question = state["question"]
            
            # Retriever 인스턴스 생성 (제공된 클래스를 사용)
            retriever_instance = Retriever()
            
            # 질문에 대해 관련 문서 검색 (get_relevant_documents 메서드 사용)
            relevant_docs = retriever_instance.retriever.invoke(question)
            
            # 각 문서의 텍스트(page_content)를 추출하여 하나의 문자열로 결합
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # AgentState의 "context" 필드에 업데이트
            state["context"] = context
            
            # RAG 평가용
            state["relevant_docs"] = relevant_docs
            
            print("검색된 문서 (context):")
            print(state["context"])
        else:
            print("문서 검색 노드 조건 미충족. classification_score:", classification_score)
        
        return state
    
    def generate(self, state: AgentState, chain):
        print("\n==== DOCUMENT WITH GENERATE ====\n")
        question = state["question"]
        documents = state["context"]

        streamed_answer = ""

        for chunk in chain.stream({"question": question, "documents": documents}):
            print(chunk, end="", flush=True)  # ✅ 실시간 출력
            streamed_answer += chunk

        state["answer"] = streamed_answer
        return state
    
    def grade_documents(self, state: AgentState, chain):
        print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
        question = state["question"]
        documents = state["relevant_docs"]

        relevant_doc_count = 0
        filtered_docs = []
        improved_prompts = []

        for d in documents:
            result = chain.invoke({
                "question": question,
                "documents": d.page_content
            })

            score = result["score"]
            improved = result.get("improved_prompt", "").strip()

            if score == "1":
                print("==== [GRADE: DOCUMENT RELEVANT] ====")
                filtered_docs.append(d)
                relevant_doc_count += 1
            else:
                print("==== [GRADE: DOCUMENT NOT RELEVANT] ====")
                if improved:
                    print("💡 개선된 질문 예시:", improved)
                    improved_prompts.append(improved)

        # ✅ 루프 끝난 후 판단해야 정확
        web_search = "Yes" if relevant_doc_count == 0 else "No"

        # 필요 시 improved_prompts도 상태에 저장
        state["relevant_docs"] = filtered_docs
        state["web_search"] = web_search
        if improved_prompts:
            state["improved_prompts"] = improved_prompts

        return state

    
    def web_search(self, state: AgentState, chain):
        print("\n==== [WEB SEARCH] ====\n")
        question = state["question"]
        documents = state.get("context", [])

        searcher = WebSearch()
        docs = searcher.search.invoke(question)

        web_results = "\n".join([d["snippet"] for d in docs])

        # ✅ 여기에 LLM 호출 (예: 보고서 생성)
        streamed_report = ""
        for chunk in chain.stream({"question": question, "documents": web_results}):
            print(chunk, end="", flush=True)
            streamed_report += chunk

        # 문서 형태로 감싸서 저장
        doc_obj = Document(page_content=streamed_report)
        documents.append(doc_obj)
        state["context"] = documents

        # 답변도 저장해주자
        state["answer"] = streamed_report
        return state


        
        
    
