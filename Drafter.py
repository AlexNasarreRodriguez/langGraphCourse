from typing import TypedDict,Annotated,Sequence
from langgraph.graph import StateGraph,END,START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,ToolMessage,SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

document_content = ''

class AgentState(TypedDict): #state schema
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    
@tool
def update(content: str) -> str:
    """
    Updates the content of the document.

    Args:
        content (str): The new content of the document.

    Returns:
        str: A message indicating that the document has been updated.
    """
    global document_content
    document_content = content
    return f'Document updated: {document_content}'

@tool
def save(filename: str) -> str:
    """
    Saves the content of the document to a file.

    Args:
        filename (str): The name of the file to save the document to.

    Returns:
        str: A message indicating that the document has been saved.
    """
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    with open(filename, 'w') as f:
        f.write(document_content)
    return f'Document saved to {filename}'

tools = [
    update,
    save
]

model = ChatOpenAI(model='gpt-4o').bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    """
    The agent function that will be used to generate a response.

    Given the current state of the document and the user's input, this function will:

    1. Update or modify the document content if the user wants to.
    2. Save and finish if the user wants to.
    3. Always show the current document state after modifications.

    The current document content is given by the global document_content variable.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The new state of the agent after processing the user's input.
    """

    system_prompt = SystemMessage(content=f'''
You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

- If the user wants to update or modify content, use the 'update' tool with the complete updated content.
- If the user wants to save and finish, you need to use the 'save' tool.
- Make sure to always show the current document state after modifications.

The current document content is: {document_content}''')
    
    if not state['messages']:
        user_input = "I'm ready to help you update and modify the document. What do you want to do?"
        user_message = HumanMessage(content=user_input)
        
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUSER: {user_input}")
        user_message = HumanMessage(content=user_input)
        
    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    response = model.invoke(all_messages)
    
    print(f"\nAI: {response.content}")
    if hasattr(response,'tool_calls') and response.tool_calls:
        print(f'USING TOOLS: {[tc['name'] for tc in response.tool_calls]}')
     
    return {'messages': list(state['messages']) + [user_message, response]}


def should_continue(state: AgentState)-> str:
    """
    Determines whether the agent should continue processing or end.

    This function iterates through the messages in the given agent state
    to decide if the processing should be terminated. If there are no
    messages, it defaults to continuing. If a message indicating that the
    document has been saved is found, it returns 'end' to signal termination.

    Args:
        state (AgentState): The current state of the agent containing messages.

    Returns:
        str: 'end' if a document save confirmation is found, otherwise 'continue'.
    """

    messages = state['messages']
    if not messages:
        return 'continue'
    
    for message in messages:
        if (isinstance(message, ToolMessage) and
            'saved' in message.content.lower() and
            'document' in message.content.lower()):
            return 'end'

    return 'continue'

def print_messages(messages):
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f'\nTOOL RESULT: {message.content}')
            
graph = StateGraph(AgentState)

graph.add_node('agent', our_agent)
graph.add_node('tools', ToolNode(tools=tools))

graph.set_entry_point('agent')
graph.add_edge('agent', 'tools')

graph.add_conditional_edges('tools', should_continue, {'continue': 'agent', 'end': END})    
app = graph.compile()

def run_document_agent():
    print('------------DRAFTER------------')
    state = {'messages': []}
    for step in app.stream(state,stream_mode='values'):
        if 'messages' in step:
            print_messages(step['messages'])
    
    print('------------DRAFTER FINISHED ------------')
    
if __name__ == '__main__':
    run_document_agent()