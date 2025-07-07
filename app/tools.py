from langchain_core.tools import tool
from app.llm_utils import code_analysis_llm
from langchain_core.prompts import ChatPromptTemplate
import json

# --- Expert Code Analysis Tools ---

@tool
def read_code_file(file_path: str) -> str:
    """
    Reads the content of a code file from the given file path.
    This should be the first step in the code review process.
    """
    print(f"--- Reading code from {file_path} ---")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: The file at {file_path} was not found."
    except Exception as e:
        return f"An error occurred while reading the file: {e}"

@tool
def check_for_common_errors(code: str) -> str:
    """
    Analyzes the code to identify common errors, bugs, and style violations (e.g., PEP 8).
    Use this tool to perform a static analysis of the code.
    Returns a JSON string with a list of identified issues.
    """
    print("--- Checking for common errors and bugs ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert code analyst. Your task is to identify potential bugs, syntax errors, and style violations in the provided code. Focus on correctness and adherence to best practices like PEP 8. Provide your findings as a JSON object with a key 'issues' containing a list of strings, where each string is a specific issue found."),
        ("user", "Please analyze the following code:\n\n```python\n{code}\n```")
    ])
    chain = prompt | code_analysis_llm
    response = chain.invoke({"code": code})
    return response.content

@tool
def assess_code_quality(code: str) -> str:
    """
    Assesses the overall quality of the code, focusing on readability, maintainability, and efficiency.
    Use this tool to get a high-level assessment of the code's architecture and design.
    Returns a JSON object with keys 'readability', 'maintainability', and 'efficiency', each with a score (1-10) and a brief justification.
    """
    print("--- Assessing code quality ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a principal software engineer. Assess the provided code for overall quality. Evaluate its readability, maintainability, and efficiency. Provide a score from 1 (poor) to 10 (excellent) for each category, along with a brief justification. Return the result as a JSON object."),
        ("user", "Please assess this code:\n\n```python\n{code}\n```")
    ])
    chain = prompt | code_analysis_llm
    response = chain.invoke({"code": code})
    return response.content

@tool
def suggest_improvements(code: str) -> str:
    """
    Provides specific, actionable suggestions for improving and refactoring the code.
    Use this tool to get concrete examples of how the code could be made better.
    Returns a JSON object with a key 'suggestions' containing a list of improvement descriptions.
    """
    print("--- Suggesting improvements and refactoring ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coding assistant. Your goal is to provide specific, actionable suggestions to improve the given code. Focus on refactoring for clarity, performance, and simplicity. Provide your findings as a JSON object with a key 'suggestions' containing a list of strings."),
        ("user", "Please provide improvement suggestions for this code:\n\n```python\n{code}\n```")
    ])
    chain = prompt | code_analysis_llm
    response = chain.invoke({"code": code})
    return response.content

@tool
def generate_review_summary(error_report: str, quality_assessment: str, improvement_suggestions: str) -> str:
    """
    Synthesizes all analysis reports into a final, structured code review summary.
    This is the final step. Use this tool after all other analyses are complete.
    The inputs should be the JSON string outputs from the other tools.
    """
    print("--- Generating final review summary ---")
    
    # Safely parse JSON inputs
    try:
        errors = json.loads(error_report)
        quality = json.loads(quality_assessment)
        suggestions = json.loads(improvement_suggestions)
    except json.JSONDecodeError as e:
        return f"Error decoding JSON input: {e}. Please ensure inputs are valid JSON strings."

    # Build the final report in Markdown format
    summary = "# Code Review Summary\n\n"
    
    summary += "## 1. Code Quality Assessment\n"
    summary += f"- **Readability:** {quality.get('readability', {}).get('score', 'N/A')}/10 - *{quality.get('readability', {}).get('justification', '')}*\n"
    summary += f"- **Maintainability:** {quality.get('maintainability', {}).get('score', 'N/A')}/10 - *{quality.get('maintainability', {}).get('justification', '')}*\n"
    summary += f"- **Efficiency:** {quality.get('efficiency', {}).get('score', 'N/A')}/10 - *{quality.get('efficiency', {}).get('justification', '')}*\n\n"

    summary += "## 2. Issues and Bugs Found\n"
    if errors.get('issues'):
        for issue in errors['issues']:
            summary += f"- {issue}\n"
    else:
        summary += "- No major issues found.\n"
    summary += "\n"

    summary += "## 3. Suggested Improvements\n"
    if suggestions.get('suggestions'):
        for suggestion in suggestions['suggestions']:
            summary += f"- {suggestion}\n"
    else:
        summary += "- No specific improvements suggested.\n"
        
    return summary

# List of all available tools for the orchestrator
all_tools = [read_code_file, check_for_common_errors, assess_code_quality, suggest_improvements, generate_review_summary]
