from typing import List, TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from app.tools import all_tools
from app.llm_utils import orchestrator_llm
from rich.console import Console

console = Console()

# --- State Definition ---
# The state now has dedicated fields for large data payloads,
# keeping the `messages` list lean to avoid context window errors.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    code_content: Optional[str]
    error_report: Optional[str]
    quality_assessment: Optional[str]
    improvement_suggestions: Optional[str]

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

def process_tool_results(state: AgentState) -> dict:
    """
    Processes the output of the last tool call(s), stores the data in the appropriate
    state field, and replaces the verbose tool output in the message history
    with a concise confirmation message. This version handles parallel tool calls.
    """
    console.print("[bold blue]Processing tool results...[/bold blue]")
    
    # Find the last AI message with tool calls to identify what was called.
    last_ai_message = None
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_ai_message = msg
            break
    
    if not last_ai_message:
        return {}

    # Identify the tool messages that correspond to the last AI call
    # These will be all messages after the last AI message.
    try:
        last_ai_message_index = state['messages'].index(last_ai_message)
        tool_messages = state['messages'][last_ai_message_index + 1:]
    except ValueError:
        return {}


    updates = {}
    
    for tool_message in tool_messages:
        if not isinstance(tool_message, ToolMessage):
            continue

        # Find the tool name from the original call
        tool_name = ""
        for call in last_ai_message.tool_calls:
            if call['id'] == tool_message.tool_call_id:
                tool_name = call['name']
                break
        
        if not tool_name:
            continue

        # Store the output in the correct state field
        if tool_name == "read_code_file":
            updates['code_content'] = tool_message.content
            tool_message.content = f"Successfully read the code file."
        elif tool_name == "check_for_common_errors":
            updates['error_report'] = tool_message.content
            tool_message.content = f"Successfully checked for errors."
        elif tool_name == "assess_code_quality":
            updates['quality_assessment'] = tool_message.content
            tool_message.content = f"Successfully assessed code quality."
        elif tool_name == "suggest_improvements":
            updates['improvement_suggestions'] = tool_message.content
            tool_message.content = f"Successfully generated improvement suggestions."

        console.print(f"[bold green]Stored output from '{tool_name}' in state.[/bold green]")
        
    return updates


def should_continue(state: AgentState) -> str:
    """
    Conditional Edge: This function decides the next step in the graph.
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
workflow.add_node("process_results", process_tool_results)

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
# After a tool is executed, process its results before looping back to the orchestrator
workflow.add_edge("action", "process_results")
workflow.add_edge("process_results", "orchestrator")


# Compile the graph
code_reviewer_agent_graph = workflow.compile()
console.print("[bold magenta]Dynamic code reviewer agent compiled successfully.[/bold magenta]")