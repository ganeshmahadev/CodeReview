import argparse
from app.graph import code_reviewer_agent_graph
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.markdown import Markdown

def main():
    """
    Main function to run the dynamic code reviewer agent.
    """
    parser = argparse.ArgumentParser(description="Review a code file using a dynamic agent.")
    parser.add_argument("file_path", type=str, help="The path to the code file to be reviewed.")
    
    args = parser.parse_args()
    
    console = Console()
    console.print(f"[bold yellow]Starting Code Review for:[/bold yellow] {args.file_path}")

    # The initial prompt to kick off the agent's workflow
    initial_prompt = (
        "You are a senior developer performing a code review. Follow these steps strictly and sequentially. Do not move to the next step until the previous one is complete.\n"
        "1. **Read the code:** Use the `read_code_file` tool to get the content of the file.\n"
        "2. **Analyze for Errors:** Take the code content from step 1 and use the `check_for_common_errors` tool to find any bugs.\n"
        "3. **Assess Quality:** Take the code content from step 1 and use the `assess_code_quality` tool.\n"
        "4. **Suggest Improvements:** Take the code content from step 1 and use the `suggest_improvements` tool.\n"
        "5. **Synthesize Final Report:** Once you have the results from steps 2, 3, and 4, and ONLY then, call the `generate_review_summary` tool. You MUST provide the outputs from the previous steps as arguments for `error_report`, `quality_assessment`, and `improvement_suggestions`.\n\n"
        f"Begin the review for the file located at: {args.file_path}"
    )

    initial_state = {
        "messages": [HumanMessage(content=initial_prompt)]
    }

    try:
        # Stream the events to see the agent's process in real-time
        for event in code_reviewer_agent_graph.stream(initial_state, stream_mode="values"):
            final_response_message = event["messages"][-1]
            # Only print the final output if the last message is from the AI and has no tool calls.
            if isinstance(final_response_message, AIMessage) and not final_response_message.tool_calls:
                console.print("\n" + "="*50)
                console.print("[bold green]Final Code Review Report:[/bold green]")
                console.print("="*50 + "\n")
                
                markdown_output = Markdown(final_response_message.content)
                console.print(markdown_output)

    except Exception as e:
        console.print(f"[bold red]An error occurred during the process: {e}[/bold red]")

if __name__ == "__main__":
    main()