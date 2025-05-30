from typing import TypedDict,List,Union
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[AIMessage,HumanMessage]]
    
llm = ChatOpenAI(model='gpt-4o') 

def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node('process', process_node)
graph.add_edge(START, 'process')
graph.add_edge('process', END)
agent = graph.compile()

conversation_history = []
user_input = input("User: ")

while user_input != 'exit':
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({'messages': conversation_history}) 
    conversation_history = result['messages']
    user_input = input("User: ")

with open('chatbot.txt', 'w') as file:
    file.write('Your conversation log:\n')
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write('End of conversation\n')
print("Conversation saved to chatbot.txt") 