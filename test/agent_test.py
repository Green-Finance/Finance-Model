import json
from langgraph.graph import StateGraph, START, END
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List, TypedDict

# vector db
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# PDF 
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_ollama import ChatOllama

# search 
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

# model define 
kanana = ChatOllama(
    model="huihui_ai/kanana-nano-abliterated:2.1b"
    )

seek = ChatOllama(
    model="jinbora/deepseek-r1-Bllossom:8b",
)

# ✅ PDF 로드 및 변환
file_path = "./20250311_industry_897229000.pdf"
loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
docs = loader.load() 
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")])
splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
texts = [d.page_content for d in splits]

# ✅ 임베딩 모델 및 벡터 DB 설정
model_name = "jhgan/ko-sroberta-multitask"
embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False})
vdb = FAISS.from_texts(texts, embedding=embedding_model)
retriever = vdb.as_retriever(search_kwargs={"k": 3})

# ✅ DuckDuckGo 검색 설정
wrapper = DuckDuckGoSearchAPIWrapper(region="ko-kr", time="d", max_results=5)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news", output_format="list")

# ✅ 상태 정의
class GraphState(TypedDict):
    question: str  # 질의
    generation: str  # 생성
    search: str  # 검색
    documents: List[Document]  # 문서
    steps: List[str]  # 순서
    
# 문서 검색     
def retrieve(state):
    print("🔍 [retrieve] 문서 검색 시작...")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"📄 [retrieve] 검색된 문서 개수: {len(documents)}")
    return {"documents": documents, "question": question, "steps": state["steps"] + ["retrieve_documents"]}


# ✅ 문서 평가 프롬프트 (한글화)
retrieval_grader_prompt = PromptTemplate(
    template="""
    당신은 주어진 문서가 질문과 얼마나 관련이 있는지 평가하는 전문가입니다.
    
    - 문서가 질문과 관련이 있으면 1점을 부여합니다.
    - 문서가 질문과 관련이 없으면 0점을 부여합니다.

    질문: {question}  
    문서 내용: {documents}  

    JSON 형식으로 결과를 반환해주세요.  
    {{
      "score": "1" 또는 "0"
    }}
    """,
    input_variables=["question", "documents"],
)

retrieval_grader = retrieval_grader_prompt | kanana | JsonOutputParser()

# 문서 평가 노드 
def grade_documents(state):
    print("📝 [grade_documents] 문서 평가 시작...")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    search = "No"

    for d in documents:
        try:
            score = retrieval_grader.invoke({"question": question, "documents": d.page_content})
            print(f"📊 [grade_documents] 평가 결과: {score}")
            if score["score"] == "1":
                filtered_docs.append(d)
            else:
                search = "Yes"
        except Exception as e:
            print(f"❌ [grade_documents] 오류 발생: {e}")

    print(f"📑 [grade_documents] 최종 유효 문서 개수: {len(filtered_docs)}")
    return {"documents": filtered_docs, "question": question, "search": search, "steps": state["steps"] + ["grade_document_retrieval"]}
# ✅ 웹 검색 노드 (DuckDuckGo 검색 사용)

def web_search(state):
    print("🌐 [web_search] 웹 검색 시작...")
    question = state["question"]
    documents = state.get("documents", [])
    
    try:
        web_results = search.invoke(question)
        documents.extend([Document(page_content=res["content"], metadata={"url": res["url"]}) for res in web_results])
        print(f"🔎 [web_search] 웹 검색 결과 개수: {len(web_results)}")
    except Exception as e:
        print(f"❌ [web_search] 오류 발생: {e}")

    return {"documents": documents, "question": question, "steps": state["steps"] + ["web_search"]}



# ✅ 답변 생성 프롬프트
generate_prompt = PromptTemplate(
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

rag_chain = generate_prompt | seek | StrOutputParser()

# ✅ 답변 생성 노드
def generate(state):
    print("📝 [generate] 답변 생성 시작...")
    question = state["question"]
    documents = state["documents"]

    try:
        generation = rag_chain.invoke({"documents": documents, "question": question})
        print(f"✅ [generate] 생성된 답변: {generation}")
    except Exception as e:
        print(f"❌ [generate] 오류 발생: {e}")
        generation = "❌ 답변 생성 중 오류 발생"

    return {"documents": documents, "question": question, "generation": generation, "steps": state["steps"] + ["generate_answer"]}

# ✅ 검색 필요 여부 결정
def decide_to_generate(state):
    return "web_search" if state["search"] == "Yes" else "generate"

# ✅ LangGraph 워크플로우 구축
workflow = StateGraph(GraphState)

# 🔹 노드 추가
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)

# 🔹 노드 연결
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)


custom_graph = workflow.compile()

# ✅ 실행 테스트 (디버깅 포함)
if __name__ == "__main__":
    query = "반도체 시장 동향 분석"
    
    try:
        print("🚀 LangGraph 실행 시작...")
        result = custom_graph.invoke({"question": query, "steps": []})

        print(f"✅ LangGraph 실행 성공!")
        print(f"🔹 검색된 문서 개수: {len(result['documents'])}")
        print(f"📝 생성된 답변: {result['generation']}")
        print(f"🛠️ 수행된 단계: {result['steps']}")

    except Exception as e:
        print(f"❌ LangGraph 실행 중 오류 발생: {e}")

