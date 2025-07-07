from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

code_analysis_llm = ChatGroq(
    temperature=0.7,
    model_name="llama3-70b-8192"
)

# This powerful LLM is specifically for the orchestrator agent, as it needs to reason about tool usage
orchestrator_llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192"
)