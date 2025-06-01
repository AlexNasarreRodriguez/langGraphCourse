from dotenv import load_dotenv
import os
from typing import TypedDict,Annotated,Sequence
from langgraph.graph import StateGraph,END,START
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,ToolMessage,SystemMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model='gpt-4o',
                 temperature=0,)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
pdf_path = os.getenv('PDF_PATH')

if not os.path.exists(pdf_path):
    raise Exception('PDF_PATH not found')

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f'Loaded {len(pages)} pages') 
except Exception as e:
    print(f'Error loading PDF: {e}')
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


pages_split = text_splitter.split_documents(pages)

persist_directory = os.getenv('PERSIST_DIRECTORY')
collection_name = 'ArquitecturaLimpia'

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
                documents=pages_split,
                persist_directory=persist_directory, 
                collection_name=collection_name, 
                embedding=embeddings)
    print('Created vectorstore')
except Exception as e:
    print(f'Error adding documents to Chroma: {e}')
    raise

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={
        'k': 5
    }
)

@tool
def retriever_tool(query: str) -> str:
    
    """
    Searches the vectorstore with the given query and returns the relevant documents

    Args:
        query (str): The query to search for

    Returns:
        str: The relevant documents, or "No relevant documents found" if none
    """
    docs = retriever.invoke(query)
    if not docs:
        return 'No relevant documents found'
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
        
    return '\n\n'.join(results)


tools = [
    retriever_tool
]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.content) > 0

system_prompt = SystemMessage(content=f''' 
                              You are an intelligent AI assistant who answers questions about the book Arquitectura Limpia based on the document loaded into your knowledge base.
                              Use the retriever tool to find relevant information about the book. You can make multiple calls if needed.
                              If you need to look up some information before asking a follow up question, you are allowed to do that.
                              Please always cite the specific parts of the documents you use in your answers.
                              ''')

tools_dict = {our_tool.name: our_tool for our_tool in tools}
def call_llm(state: AgentState) -> AgentState:
    """
    Calls the llm with the given state messages and returns a new state with the llm response.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: A new state with the llm response.
    """
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm(messages)
    return {'messages' : [message]}

def take_action(state: AgentState) -> AgentState:
    '''Execute tool calls from the LLM's response'''
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f'Calling tool: {t['name']} with query: {t['args'].get("query", "No query provided")}')
        
        if not t['name'] in tools_dict:
            print(f'Unknown tool: {t["name"]}')
            result = 'incorrect tool name'
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f'Result lenght: {len(str(result))}')
            
        results.append(ToolMessage(tool_call_id=t['id'],name=t['name'],content=str(result)))
    
    print('Tools Execution completed, BACK TO MODEL')
    return{'messages':results}



graph = StateGraph(AgentState)

graph.add_node('llm', call_llm)
graph.add_node('retriever_agent', take_action)


graph.set_entry_point('llm')
graph.add_conditional_edges('llm', should_continue, {True: 'retriever_agent', False: END})
graph.add_edge('retriever_agent', 'llm')

app = graph.compile()

def running_agent():
    print('\n------------RAG AGENT------------')
    while True:
        user_input = input("\nWHAT IS YOUR QUESTION: ")
        if user_input == 'exit':
            break
        messages = [HumanMessage(content=user_input)]
        
        result = app.invoke({'messages': messages})
        print('************ANSWER************')
        print(result['messages'][-1].content)
    
    
    
    
    
    
    
    
    
    
    
    