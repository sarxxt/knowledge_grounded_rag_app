from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import logging
import traceback

load_dotenv()

openai_api_key = os.getenv("openai_api_key")
print(openai_api_key)
try:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    template = """You are an expert pdf reader and data analyst,
    Objectives:
    provide me an efficient answer and Please be as detailed as you can.
    Instrunctions:
    If you don't know the answer, just say that you don't know, don't try to make an answer.
    {context}
    Question: {question}
    Helpful Answer:
    """
except Exception as e:
    error_msg = "Error while Initializing ChatOpenAi: "
    logging.error(f"{error_msg}{e} trace_back:{traceback.format_exc()}")
    raise Exception (f"{error_msg} {e}")

def gpt_chain(vector_store,question):
    """
    Retrieves context and generates an answer to a question using GPT.

    Parameters:
    - vector_store: The vectorized data store for context retrieval.
    - question: The question string to be processed.

    Returns:
    - dict: Contains the generated answer and source documents.
    """

    try:
        logging.info("Generating Q/A Chain")
        retriever_from_llm = vector_store.as_retriever()
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question","context"],template=template,)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_from_llm, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, })
        result = qa_chain({"query":question})
        logging.info("Q/A Chain Processed Successfully")
        return result["result"]

    except Exception as e:
        error_msg =" Error while Generating Chain:"
        logging.error(f"{error_msg}{e} trace_back:{traceback.format_exc()}")
        raise Exception (f"{error_msg} {e}")
