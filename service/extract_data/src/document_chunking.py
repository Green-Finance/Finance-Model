from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from .filter import filter_irrelevant_chunks


def chunking_documents(file_path: str, embedding_model: HuggingFaceEmbeddings) -> List[Tuple[str, List[float]]]:
    loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
    docs = loader.load()

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2")
        ]
    )

    # 1. 분리
    splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]

    # 2. 텍스트만 추출
    raw_texts = [chunk.page_content.strip() for chunk in splits]

    # 3. 전처리 필터링
    cleaned_texts = filter_irrelevant_chunks(raw_texts)
    
    # 4. 임베딩 생성
    return [(text, embedding_model.embed_query(text)) for text in cleaned_texts]