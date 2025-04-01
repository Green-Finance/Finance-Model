from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

def create_chaining(prompt, model, parser):
    result = prompt | model | parser
    
    return result
    