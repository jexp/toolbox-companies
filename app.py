import json
import os
import asyncio

import inspect
from typing import Annotated, TypedDict, Literal
from typing import Callable, TypeVar

from dotenv import load_dotenv

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool, StructuredTool
from langchain_google_vertexai import ChatVertexAI

from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from toolbox_langchain import ToolboxClient


# Utility function to get a Streamlit callback handler with context
# Define a function to wrap and add context to Streamlit's integration with LangGraph
def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that integrates fully with any LangChain ChatLLM integration,
    updating the provided Streamlit container with outputs such as tokens, model responses,
    and intermediate steps. This function ensures that all callback methods run within
    the Streamlit execution context, fixing the NoSessionContext() error commonly encountered
    in Streamlit callbacks.

    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered
                                           during the LLM interaction.
    Returns:
        BaseCallbackHandler: An instance of StreamlitCallbackHandler configured for full integration
                             with ChatLLM, enabling dynamic updates in the Streamlit app.
    """

    # Define a type variable for generic type hinting in the decorator, ensuring the original
    # function and wrapped function maintain the same return type.
    fn_return_type = TypeVar('fn_return_type')

    # Decorator function to add Streamlit's execution context to a function
    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        """
        Decorator to ensure that the decorated function runs within the Streamlit execution context.
        This is necessary for interacting with Streamlit components from within callback functions
        and prevents the NoSessionContext() error by adding the correct session context.

        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated, typically a callback method.
        Returns:
            Callable[..., fn_return_type]: The decorated function that includes the Streamlit context setup.
        """
        # Retrieve the current Streamlit script execution context.
        # This context holds session information necessary for Streamlit operations.
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            """
            Wrapper function that adds the Streamlit context and then calls the original function.
            If the Streamlit context is not set, it can lead to NoSessionContext() errors, which this
            wrapper resolves by ensuring that the correct context is used when the function runs.

            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.
            Returns:
                fn_return_type: The result from the original function.
            """
            # Add the previously captured Streamlit context to the current execution.
            # This step fixes NoSessionContext() errors by ensuring that Streamlit knows which session
            # is executing the code, allowing it to properly manage session state and updates.
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)  # Call the original function with its arguments

        return wrapper

    # Create an instance of Streamlit's StreamlitCallbackHandler with the provided Streamlit container
    st_cb = StreamlitCallbackHandler(parent_container)

    # Iterate over all methods of the StreamlitCallbackHandler instance
    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):  # Identify callback methods that respond to LLM events
            # Wrap each callback method with the Streamlit context setup to prevent session errors
            setattr(st_cb, method_name,
                    add_streamlit_context(method_func))  # Replace the method with the wrapped version

    # Return the fully configured StreamlitCallbackHandler instance, now context-aware and integrated with any ChatLLM
    return st_cb

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Function to decide whether to continue tool usage or end the process
def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return "__end__"  # End the conversation if no tool is needed

# Load the tools from the Toolbox server
client = ToolboxClient("http://127.0.0.1:5000")
tools = client.load_toolset()
tool_node = ToolNode(tools)

prompt = """
  You're a helpful investment research assistant. 
  You can use the provided tools to search for companies, 
  people at companies, industries, and news articles from 2023.
  Make sure to use prior tool outputs from the conversation to filter, e.g. by location, sentiment, etc.
  or as inputs for subsequent operations. If needed use the tools that provide more detailed information 
  on articles or companies to get the information you need for filtering or sorting.
  Don't ask for confirmations from the user.
  User: 
"""

queries = [
    "What industries deal with neurological implants?",
    "List 5 companies in from those industries with their description and filter afterwards by California.",
    "Who is working at these companies?",
    "What were the news in January 2023 with positive sentiment? List top 5 articles.",
    "Summarize these articles.",
    "Which 3 companies were mentioned by these articles?"
    "Who is working there as board members?",
]

llm = ChatVertexAI(model_name="gemini-2.0-flash-001", streaming=True)

# Core invocation of the model
def _call_model(state: GraphsState):
    messages = state["messages"]
    # llm = ChatOpenAI( temperature=0.7, streaming=True, ).bind_tools(tools)

#    llm = ChatVertexAI(model_name="gemini-2.0-flash-001", streaming=True)

#    memory = MemorySaver()
    # agent = create_react_agent(llm, tools, checkpointer=memory)

    config = {"configurable": {"thread_id": "thread-1"}}

#    inputs = {"messages": [("user", prompt + query)]}
    # response = agent.invoke(messages, stream_mode="values", config=config)

    response = llm.invoke(messages)
    return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_edge(START, "modelNode")
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)

# Add conditional logic to determine the next step based on the state (to continue or to end)
graph.add_conditional_edges(
    "modelNode",
    should_continue,  # This function will decide the flow of execution
)
graph.add_edge("tools", "modelNode")
graph.add_edge("modelNode", "tools")

memory = MemorySaver()
# Compile the state graph into a runnable object
# graph_runnable = graph.compile(checkpointer=memory)
graph_runnable = create_react_agent(llm, tools, checkpointer=memory)

# Function to invoke the compiled graph externally
def invoke_our_graph(st_messages, callables):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    # stream_mode="values"
    return graph_runnable.invoke({"messages": st_messages}, config={"configurable": {"thread_id": "thread-1"}, "callbacks": callables})

load_dotenv()

st.title("Investment Research Agent - StreamLit 🤝 LangGraph 🤝 Gen AI Toolbox 🤝 Neo4j")
st.markdown("####  This is an investment research agent built with Google Gen AI Toolbox using a Neo4j Knowledge Graph")

# st write magic
# """
# In this example, we're going to be using the official [`StreamlitCallbackHandler`](https://api.python.langchain.com/en/latest/callbacks/langchain_community.callbacks.streamlit.streamlit_callback_handler.StreamlitCallbackHandler.html) 
# within [_LangGraph_](https://langchain-ai.github.io/langgraph/) by leveraging callbacks in our 
# graph's [`RunnableConfig`](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html).

# ---
# """

# Check if the API key is available as an environment variable
if not os.getenv('GOOGLE_API_KEY'):
    # If not, display a sidebar input for the user to provide the API key
    st.sidebar.header("GOOGLE_API_KEY Project Setup")
    api_key = st.sidebar.text_input(label="API Key", type="password", label_visibility="collapsed")
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
    # If no key is provided, show an info message and stop further execution and wait till key is entered
    if not api_key:
        st.info("Please enter your GOOGLE_API_KEY in the sidebar.")
        st.stop()

if "messages" not in st.session_state:
    # default initial message to render in message state
    st.session_state["messages"] = [
        SystemMessage(content=prompt),
        AIMessage(content="How can I help you?")
    ]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

# takes new input in chat box from user and invokes the graph
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Process the AI's response and handles graph events using the callback mechanism
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()  # Placeholder for visually updating AI's response after events end
        # create a new placeholder for streaming messages and other events, and give it context
        st_callback = get_streamlit_cb(st.empty())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        last_msg = response["messages"][-1].content
        st.session_state.messages.append(AIMessage(content=last_msg))  # Add that last message to the st_message_state
        msg_placeholder.write(last_msg) # visually refresh the complete response after the callback container





