from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

vectorstore = PGVector(
    collection_name="industry_reports_langchain",  # 실제 테이블과 연결
    connection_string="postgresql+psycopg2://myuser:mypassword@localhost:5432/mydatabase",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 쿼리
docs = retriever.invoke("반도체 산업 동향")


print(f"문서 개수: {len(docs)}")
for i, doc in enumerate(docs, 1):
    print(f"\n📄 문서 {i}")
    print("제목:", doc.metadata.get("title"))
    print("내용:", doc.page_content[:200], "...\n")