import psycopg2
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# 1. 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# 2. LangChain vectorstore 연결
vectorstore = PGVector(
    collection_name="industry_reports_langchain",
    connection_string="postgresql+psycopg2://myuser:mypassword@localhost:5432/mydatabase",
    embedding_function=embeddings
)

# 3. DB에서 기존 데이터 꺼내기
conn = psycopg2.connect(
    dbname="mydatabase",
    user="myuser",
    password="mypassword",
    host="localhost",
    port="5432"
)

cursor = conn.cursor()
cursor.execute("SELECT title, link, date, chunk FROM industry_reports WHERE chunk IS NOT NULL")

# 4. Document 형태로 변환
docs = []
for title, link, date, chunk in cursor.fetchall():
    docs.append(Document(
        page_content=chunk,
        metadata={"title": title, "link": link, "date": date}
    ))

# 5. LangChain vectorstore에 저장
vectorstore.add_documents(docs)
print(f"{len(docs)}개 문서를 LangChain 포맷으로 저장 완료 ✅")

cursor.close()
conn.close()

# 결론, Vector Searching을 위해서 전처리 과정을 2번 해야하는 번거로움으로 
# 태초부터 langchain 데이터에 걸맞게 변형해서 재주입할 예정 
