from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from app.tools import all_tools
from app.llm_utils import orchestrator_llm
from rich.console import Console

console = Console()

# The state for our agentic graph. It tracks the conversation history.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# Bind the tools to the orchestrator LLM to create the agent
orchestrator_agent = orchestrator_llm.bind_tools(all_tools)

# --- Agent Node Definitions ---

def call_orchestrator(state: AgentState):
    """
    This is the primary node where the orchestrator agent thinks.
    It decides which tool to call next to continue the code review.
    """
    console.print("[bold magenta]Senior Developer is thinking...[/bold magenta]")
    response = orchestrator_agent.invoke(state['messages'])
    console.print("[bold green]Senior Developer made a decision.[/bold green]")
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """
    Conditional Edge: This function decides the next step in the graph.
    - If the agent called a tool, we execute it.
    - If the agent did not call a tool, it means the review is finished.
    """
    if state['messages'][-1].tool_calls:
        return "continue"
    else:
        return "end"

# --- Build the Agentic Graph ---

workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("orchestrator", call_orchestrator)
tool_node = ToolNode(all_tools)
workflow.add_node("action", tool_node)

# Define the agentic loop
workflow.set_entry_point("orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "orchestrator")

# Compile the graph
code_reviewer_agent_graph = workflow.compile()
console.print("[bold magenta]Dynamic code reviewer agent compiled successfully.[/bold magenta]")
