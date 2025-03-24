from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings


def chunking_documents(file_path: str, embedding_model: HuggingFaceEmbeddings) -> List[Tuple[str, List[float]]]:
    loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
    docs = loader.load()

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
    )
    splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
    texts = [d.page_content for d in splits]

    return [(text, embedding_model.embed_query(text)) for text in texts if text.strip()]