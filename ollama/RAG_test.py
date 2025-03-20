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
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool
from langchain import hub 
import json 

from langchain.vectorstores.base import VectorStoreRetriever
from typing import List, Dict
from langchain.tools import Tool
from langchain.schema import Document
from langchain_core.output_parsers import JsonOutputParser

kanana = ChatOllama(model="huihui_ai/kanana-nano-abliterated:2.1b")

model = ChatOllama(
    model="jinbora/deepseek-r1-Bllossom:8b",
)

file_path = "./20250311_industry_897229000.pdf"
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
    
sector_industry_prompt = (
    "당신은 정확하고 신뢰할 수 있는 답변을 제공할 수 있는 금융전문가 입니다."
    "아래의 문맥을 사용해서 질문에 대한 답변을 작성해야 합니다."
    
    "다음 지침을 반드시 따라주세요:"
    "1. 반드시 한국어로 생각하고 한국어로 답변해야 합니다."
    "2. context에 있는 정보만을 사용해서 답변해야 합니다."
    "3. 확실한 답변을 줄 수 없다면 '주어진 정보로는 답변하기 어렵습니다'라고만 말해야 합니다."
    "4. 답변시 없는 내용을 추측하면 안되고 주어진 정보에 따라 개인적인 의견을 추가해야 합니다."
    "5. 답변은 마크다운 형식으로 작성해야 하며 형식을 벗어나서는 안됩니다."
    "6. 가능한 간결하고 명확하게 답변하세요.."
    "문맥: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sector_industry_prompt.format(context="{context}")),  # context 포함
        ("human", "{input}"),
    ]
)
combine_docs_chain = create_stuff_documents_chain(model, prompt)

rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

query = "뉴진스 데뷔날짜?"

response = rag_chain.invoke({"input": query})

# 체인 생성 후 프롬프트까지

answer = response.get("answer", "No answer found")
print(answer)
