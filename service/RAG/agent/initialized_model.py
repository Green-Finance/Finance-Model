from langchain_ollama import ChatOllama

class AgentInitialized:
    def __init__(self, model_name: str, **kwargs):
        self.model = ChatOllama(model=model_name, **kwargs)
    def __call__(self):
        return self.model