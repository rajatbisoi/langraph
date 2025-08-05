from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def add(a: int, b:int)->int:
    '''this function adds 2 numbers'''

    return a+b

def sub(a: int, b:int)->int:
    '''This function substractes first arg with 2nd arg'''

    return a - b

tools= [add,sub]

llma = ChatOllama(model='qwen2.5:0.5B').bind_tools(tools)

def model_call(state: AgentState)->AgentState:
    system_prompt = SystemMessage(content=
                                  "You Are A very higly skiled multitaster Ai who read the human message very carefully and follow instruction step by step.")
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

inputs = {'messages': [('user', "add 4 and 5 and sub 9 and 2  then add both results and give output in json format and no other comments")]}

print_stream(app.stream(inputs, stream_mode='values'))