from typing import TypedDict, Union, cast, Protocol
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START,END
from langgraph.pregel import Pregel, protocol, StateT
from dotenv import load_dotenv

_ = load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]

llm = ChatOpenAI(model='gpt-4o')
llma = ChatOllama(model='gemma:2b')

def process(state: AgentState)->AgentState:
    '''
    This Node will invke the mml for human message input`
    '''
    response = llma.invoke(state['messages'])
    print('AI:', response.content)
    return state

graph= StateGraph(AgentState)
graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)
agent = graph.compile()
conversation_history: list[ Union[HumanMessage , AIMessage]] = []
user_input = input('Enter: ')
while True:
    conversation_history.append(HumanMessage(content=user_input))
    state =  AgentState({"messages": conversation_history})
    result = agent.invoke(state)
    conversation_history = result['messages']
    user_input = input('Enter: ')