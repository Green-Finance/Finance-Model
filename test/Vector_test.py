# main retriever 
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

# reranker 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

vectorstore = PGVector(
    embeddings=embeddings,
    connection="postgresql+psycopg://myuser:mypassword@localhost:5432/mydatabase",
    collection_name="industry_reports",
    use_jsonb=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})



# reranker models 
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=5)

# 문서 압축 검색기 초기화
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("반도체 현황")

# 문서 출력 도우미 함수
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


pretty_print_docs(compressed_docs)