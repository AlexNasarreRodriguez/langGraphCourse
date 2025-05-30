from typing import TypedDict,List
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]
    
llm = ChatOpenAI(model='gpt-4o') 

def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node('process', process_node)
graph.add_edge(START, 'process')
graph.add_edge('process', END)

agent = graph.compile()
print(agent)

user_input = input("User: ")
while user_input != 'exit':
    agent.invoke({'messages': [HumanMessage(content=user_input)]}) 
    user_input = input("User: ")