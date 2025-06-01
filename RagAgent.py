from dotenv import load_dotenv
import os
from typing import TypedDict,Annotated,Sequence
from langgraph.graph import StateGraph,END,START
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,ToolMessage,SystemMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPdfLoader



from langchain_core.tools import tool
load_dotenv()

document_content = ''