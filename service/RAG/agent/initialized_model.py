from langchain_ollama import ChatOllama

class AgentInitialized:
    def __init__(self, model_name: str, **kwargs):
        self.model = ChatOllama(model=model_name, **kwargs)

    def __call__(self, *args, **kwargs):  # ✅ 인자 받아주기
        return self.model
