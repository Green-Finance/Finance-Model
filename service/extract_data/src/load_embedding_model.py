from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model(model_name: str = "jhgan/ko-sroberta-multitask"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
