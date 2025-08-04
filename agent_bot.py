from typing import TypedDict, cast, Protocol
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START,END
from langgraph.pregel import Pregel, protocol, StateT
from dotenv import load_dotenv

_ = load_dotenv()

class AgentState(TypedDict):
    messages: list[HumanMessage | AIMessage]

# llm = ChatOpenAI(model='gpt-4o')
# llma = ChatOllama(model='gemma:2b')
llma = ChatOllama(model='qwen2.5:0.5B')

def process(state: AgentState)->AgentState:
    '''̦
    This Node will invoke the llm for human message input`
    '''
    response = llma.invoke(state['messages'])
    print('AI:', response.content)
    conversation_history.append(AIMessage(content=response.content))
    return state

graph= StateGraph(AgentState)
graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)
agent = graph.compile()
conversation_history: list[HumanMessage | AIMessage] = []
try:
    with open('conversation_history.txt', 'r') as f:
        for line in f:
            if str(line).startswith('User:'):
                conversation_history.append(f"User: {HumanMessage(content=line)}")
            if str(line).startswith('AI:'):
                conversation_history.append(f"User: {AIMessage(content=line)}")
    print("Conversation history loaded from 'conversation_history.txt'.")
except FileNotFoundError:
    print('file not found')


user_input = input('Enter: ')
while user_input != 'exit':
    conversation_history.append(HumanMessage(content=user_input))
    state =  AgentState({"messages": conversation_history})
    result = agent.invoke(state)
    conversation_history = result['messages']
    user_input = input('Enter: ')
with open('conversation_history.txt', 'w') as f:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
print("Conversation history saved to 'conversation_history.txt'.")

