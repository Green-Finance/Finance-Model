from langchain_ollama import ChatOllama 
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

model = ChatOllama(
    model="jinbora/deepseek-r1-Bllossom:8b",
)


file_path = "./20250310_industry_15280000.pdf"

loader = DoclingLoader(
    file_path=file_path,
    export_type=ExportType.MARKDOWN
)

docs = loader.load() 

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
)

splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
texts = [d.page_content for d in splits]

## 가져온 PDF 파일을 마크다운 형식으로 변형해주는 작업 


model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


vdb = FAISS.from_texts(
    texts, embedding=embedding_model
)


retriever = vdb.as_retriever(search_kwargs={"k": 3})

## 변형된 마크다운을 기준으로 리트리버 작성 


query = "코스피 기관이랑 외국인이 7주 연속 주식을 어떻게 처분했니?"

    
system_prompt = (
    "당신은 금융 전문가로서 금융 관련 질문을 대답해주는 입장입니다."
    "주어진 문맥을 참고하여 질문에 답하세요. "
    "답을 모를 경우, '모르겠습니다'라고만 답하고 스스로 답을 만들지 마세요. "
    "답변은 최대 3문장으로 간결하게 작성하세요. "
    "최종 답변은 무조건 한국어(korean)으로 작성해주세요"
    "문맥: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt.format(context="{context}")),  # context 포함
        ("human", "{input}"),
    ]
)
combine_docs_chain = create_stuff_documents_chain(model, prompt)

rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

response = rag_chain.invoke({"input": query})

# 체인 생성 후 프롬프트까지

answer = response.get("answer", "No answer found")
print(answer)