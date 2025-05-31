from typing import TypedDict,Annotated,Sequence
from langgraph.graph import StateGraph,END,START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,ToolMessage,SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict): #state schema
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

@tool
def add(a: int, b: int):
    '''this is an addition tool a + b'''
    return a + b

@tool
def substract(a: int, b: int):
    '''this is a subtraction tool a - b'''
    return a - b

@tool
def multiply(a: int, b: int):
    '''this is a multiplication tool a * b'''
    return a * b


tools = [
    add,
    substract,
    multiply
]

model = ChatOpenAI(model='gpt-4o').bind_tools(tools)

def node_model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content='You are mi AI assistant, answer my query at the best of your ability.') 
    response = model.invoke([system_prompt]+state['messages']) 
    return {'messages': [response]}

def router_should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return 'end'
    else:
        return 'continue'

graph = StateGraph(AgentState)

graph.add_node('model', node_model_call)

tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'model')

graph.add_conditional_edges('model', router_should_continue, {'continue': 'tools', 'end': END})

graph.add_edge('tools', 'model')

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {'messages': [('user', 'por favor puedes decirme cual es la suma de 4 y 9??'+
                        'y el resultado de la suma anterior restale 5 y luego multiplicalo por 3,quiero que tengas'+
                        ' en cuenta que debes hacer los pasos 1 a uno para no equivocarte,graaaciaas')]}
print_stream(agent.stream(inputs,stream_mode='values'))