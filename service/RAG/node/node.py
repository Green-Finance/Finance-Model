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
            relevant_docs = retriever_instance.retriever.get_relevant_documents(question)
            
            # 각 문서의 텍스트(page_content)를 추출하여 하나의 문자열로 결합
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # AgentState의 "context" 필드에 업데이트
            state["context"] = context
            
            print("검색된 문서 (context):")
            print(state["context"])
        else:
            print("문서 검색 노드 조건 미충족. classification_score:", classification_score)
        
        return state
    
    def generate(self, state: AgentState, chain):
        print("\n==== DOCUMENT WITH GENERATE ====\n")
        question = state["question"]
        documents = state["context"]  # context에 검색된 문서들이 저장되어 있음
        
        # 프롬프트 체인 호출 시, 올바른 키 이름("documents") 사용
        generation = chain.invoke({"question": question, "documents": documents})
        
        # AgentState에 생성된 답변 업데이트
        state["answer"] = generation
        return state
    
    def grade_documents(self, state: AgentState, chain):
        print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
        question = state["question"]
        documents = state["documents"]

        # 필터링된 문서를 저장할 리스트와 관련 문서 개수 초기화
        filtered_docs = []
        relevant_doc_count = 0

        for d in documents:
            # 각 문서의 관련성을 평가 (retrieval_grader는 미리 초기화되어 있어야 함)
            score = chain.invoke({
                "question": question,
                "document": d.page_content
            })
            # retrieval_grader가 반환한 결과에서 binary_score 값을 가져옴
            grade = score.binary_score

            if grade == "yes":
                print("==== [GRADE: DOCUMENT RELEVANT] ====")
                filtered_docs.append(d)
                relevant_doc_count += 1
            else:
                print("==== [GRADE: DOCUMENT NOT RELEVANT] ====")
                continue

        # 관련 문서가 하나도 없으면 웹 검색이 필요함을 표시
        web_search = "Yes" if relevant_doc_count == 0 else "No"
        return {"documents": filtered_docs, "web_search": web_search}
    
    def web_search(self, state: AgentState):
        print("\n==== [WEB SEARCH] ====\n")
        question = state["question"]
        documents = state["documents"]
        
        docs = WebSearch.wrapper.search.invoke(question)
        
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        
        return {"documents": documents}
        
        
        
    
