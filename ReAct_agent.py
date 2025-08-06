from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def add(a: int, b:int)->int:
    """
    Add two integers and return the result.

    Args:
        a (int): The first number to add.
        b (int): The second number to add.

    Returns:
        int: The sum of a and b.

    Example:
        add(2, 3) -> 5
    """

    return a+b

def sub(a: int, b:int)->int:
    """
    Subtract the second integer from the first and return the result.

    Args:
        a (int): The number from which to subtract.
        b (int): The number to subtract.

    Returns:
        int: The result of a - b.

    Example:
        sub(5, 2) -> 3
    """

    return a - b

tools= [add,sub]

# llma = ChatOllama(model='qwen2.5:0.5B').bind_tools(tools)
llma = ChatOllama(model='qwen3:1.7b',reasoning=True).bind_tools(tools)

def model_call(state: AgentState)->AgentState:
    system_prompt = SystemMessage(
    content=(
        # "You are a helpful assistant. For each user query, break it into clear, logical steps, "
        # "use available tools for calculations, and always return the final answer in JSON format. "
        # "If a step is ambiguous, ask for clarification."
        "You are here to work on some assignments that I will provide you. Try to break problem statement into small chunks and provide me output in JSON format."
            )
        )
    response = llma.invoke([system_prompt] + state['messages'])
    return {'messages':[response]}

def should_continue(state: AgentState)->str:
    messaages= state['messages']
    last_message = messaages[-1]
    if not last_message.tool_calls:
        return "end"
    
    else:
        return "Continue"
    

#----------------Graph--------------------------

graph = StateGraph(AgentState)
graph.add_node('our_agent',model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    'our_agent',
    should_continue,
    {
        "Continue":"tools",
        'end': END
    }
)

graph.add_edge('tools',"our_agent")

app = graph.compile()

#-----------------------------------------------

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {'messages': [('user', "add 4 and 5 and sub 9 and 2  then add both results and give output in json format only DONOT add your comments Only JSON where key:value is {'final': result} ")]}

print_stream(app.stream(inputs, stream_mode='values'))