from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content =""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str)->str:
    """
    updates the doc with provided content.
    """

    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str)->str:
    """Save the current document to a text file and finish the process
        args: 
        filename: Name for the file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
            print(f"\nDocument has been saved to : {filename}")
            return f"\nDocument has been saved to : {filename}"
    except Exception as e:
        return f"Error saving dovument: {str(e)}"
    
tools = [update, save]

model = ChatOllama(model="qwen2.5:0.5B").bind_tools(tools)
# model = ChatOllama(model="qwen3:1.7b").bind_tools(tools)

def our_agent(state: AgentState)->AgentState:
    system_prompt = SystemMessage(content=f"""
    you are a Drafter, a helpgul writimg assistant. you are going to help the user update, modify or create documents.
                                  
    - If the user update, modify, or create content use "update" tool with complete updated content.
    - If the user save or finish, you need to use "save" tool.
    - Make sure to always show the current document state after modification.
    - read input carefully 2 times to make sure not to make mistakes or marks will be deducted.
                                  
    The current document content is: {document_content}
                                  """)
    
    if not state["messages"]:
        user_input = "I am ready to help you with a document . what would you like to  create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    print(all_messages)
    response = model.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState)->str:
    """Determine if we should comntine or end the convrsation."""

    messages = state["messages"]

    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and "saved" in message.content.lower() \
            and "document" in message.content.lower()):
            return "end"
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""

    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent","tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():

    print("\n =========== DRAFTER ===========")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ============ DRAFTER FINISHED ==========")

if __name__ == "__main__":
    run_document_agent()