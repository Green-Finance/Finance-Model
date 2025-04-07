# chat modules 
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from functools import partial
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.graph import MermaidDrawMethod

# chaining modules 

from state.agent_state import AgentState
from chaining.chain import create_chaining
from prompts.prompt import PromptChain
from agent.initialized_model import AgentInitialized
from node.node import Node

# Tracing 

import os 
from dotenv import load_dotenv

load_dotenv("C:/Users/user/Desktop/Finance-Model/.env")

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_API_KEY"] =  os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] =  os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

def main():

    general_chain = create_chaining(
    prompt=PromptChain().llama_prompt,
    model=AgentInitialized(model_name="GreenFinance-Llama3.2:0.0.1v"),
    parser=StrOutputParser() 
    )
    
    classification_chain = create_chaining(
        prompt=PromptChain().classfication_question_prompt,
        model=AgentInitialized(model_name="huihui_ai/kanana-nano-abliterated:2.1b"),
        parser=JsonOutputParser()
    )
    
    document_chain = create_chaining(
        prompt=PromptChain().document_prompt,
        model=AgentInitialized(model_name="GreenFinance-Deepseek-Llama3.1-8B:0.0.1v"),
        parser=StrOutputParser()
    )
    
    grader_chain = create_chaining(
        prompt=PromptChain().retrieval_grader_prompt,
        model=AgentInitialized(model_name="huihui_ai/kanana-nano-abliterated:2.1b"),
        parser=JsonOutputParser()
    )
    
    web_search_chain = create_chaining(
        prompt=PromptChain().web_report_prompt,
        model=AgentInitialized("GreenFinance-gemma2:0.0.1v"),
        parser=StrOutputParser()
    )
    
    # 노드 결합 완료
    classification_node = partial(Node().classification_node, chain=classification_chain)
    general_node = partial(Node().general_node, chain=general_chain)
    generate = partial(Node().generate, chain=document_chain)
    grade_documents = partial(Node().grade_documents, chain=grader_chain)

    # Workflow 
    workflow = StateGraph(AgentState)

    workflow.add_node("classification", classification_node)
    workflow.add_node("general", general_node)
    workflow.add_node("document_search", Node().document_retriever)
    workflow.add_node("generate", generate)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("web_search", Node().web_search)
    
    
    workflow.add_edge(START, "classification") 
    
    workflow.add_conditional_edges(
        "classification",
        lambda state: (
            "general" if state["classification_score"] == "0" 
            else "document_search" if state["classification_score"] == "1"
            else "web_search"
        )
    )
    
    workflow.add_edge("general", END)
    
    workflow.add_edge("document_search", "grade")
    workflow.add_conditional_edges(
    "grade",
    lambda state: "web_search" if state.get("web_search") == "Yes" else "generate"
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    result = workflow.compile()
    
    return result 
    

if __name__ == "__main__":
    app = main() 
    
    question = "반도체 동향 분석해줄래?"
    
    state = AgentState(question=question)
    
    result = app.invoke(state)
    
    print("/n🧠 질문:", question)
    print("🤖 답변:", result.get("answer"))
