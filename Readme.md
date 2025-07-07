This project uses a dynamic, agentic workflow to automate the peer review process for a given code file. It leverages LangGraph, open-source LLMs via Groq, and LangSmith for observability.

### How It Works

- An **Orchestrator Agent** acts as a "Senior Developer," planning and executing the code review. It decides which aspects of the code to analyze first.
- **Worker Tools** are specialized "expert" agents, each focused on a specific part of the review:
    - Checking for common errors and bugs.
    - Assessing overall code quality and readability.
    - Suggesting concrete improvements and refactoring.
- The **Graph** is a loop where the orchestrator uses its tools to analyze the code from multiple angles and then synthesizes the findings into a final, structured review summary.

### How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set up your `.env` file** with your `GROQ_API_KEY` and `LANGSMITH_KEY` keys.

3.  **Create a sample code file to review**, for example, `sample_code.py`:
    ```python
    # sample_code.py
    def fibonaci(n):
        a, b = 0, 1
        result = []
        while a < n:
            result.append(a)
            a, b = b, a+b
        return result

    print(fibonaci(100))
    ```

4.  **Run from the command line:**
    Provide the path to the code file you want to review.

    ```bash
    python -m app.main "sample_code.py"
    ```