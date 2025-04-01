# chat modules 
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from functools import partial
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# chaining modules 

from state.agent_state import AgentState
from chaining.chain import create_chaining
from prompts.prompt import PromptChain
from agent.initialized_model import AgentInitialized
from node.node import Node


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
        parser=StrOutputParser
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
    
    return general_chain
    
    
    

if __name__ == "__main__":
    print(main())