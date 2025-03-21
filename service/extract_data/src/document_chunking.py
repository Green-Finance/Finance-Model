from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

def chunking_documents(url: str, model_name: str = "jhgan/ko-sroberta-multitask") -> List:
    
    loader = DoclingLoader(file_path=url, 
                           export_type=ExportType.MARKDOWN
                           )
    
    docs = loader.load() 
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header_1"), 
                             ("##", "Header_2"), 
                             ("###", "Header_3")])
    
    splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
    
    texts = [d.page_content for d in splits]
    

    embedding_model = HuggingFaceEmbeddings(model_name=model_name, 
                                            model_kwargs={"device": "cpu"}, 
                                            encode_kwargs={"normalize_embeddings": False}
                                            )
    
    return  [(text, embedding_model.embed_query(text)) for text in texts if text.strip()]



