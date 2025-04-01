from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class Retriever:
    def __init__(
        self,
        collection_name: str = "industry_reports",
        db_url: str = "postgresql+psycopg://myuser:mypassword@localhost:5432/mydatabase",
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
    ):
        # 1. 임베딩 모델
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )

        # 2. 벡터스토어
        vectorstore = PGVector(
            embeddings=embeddings,
            connection=db_url,
            collection_name=collection_name,
            use_jsonb=True
        )

        # 3. Retriever
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 3, "lambda_mult": 0.5}
        )

        # 4. Re-ranker
        cross_encoder = HuggingFaceCrossEncoder(model_name=rerank_model)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

        # 5. 압축기 기반 Retriever
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

