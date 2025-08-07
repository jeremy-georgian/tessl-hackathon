from pydantic import BaseModel
from pydantic_ai import Agent
import logfire
import os
from pathlib import Path

logfire.configure()
logfire.instrument_pydantic_ai()


class Argument(BaseModel):
    arg_name: str
    description: str

    def __str__(self) -> str:
        return f"'{self.arg_name}', '{self.description}'"


class LLMTextRepresentation(BaseModel):
    class_name: str | None  # None for root functions, class name for methods
    function_name: str
    visibility: str  # "public", "private", "protected"
    input_args: list[Argument]
    output_args: list[str]
    description: str
    complexity: str  # "low", "medium", "high"
    dependencies: list[str]  # external libraries/modules used
    side_effects: list[str]  # file I/O, network calls, global state changes
    error_handling: list[str]  # types of exceptions handled
    decorators: list[str]  # @property, @staticmethod, etc.


def summarize_code(file_content: str) -> list[LLMTextRepresentation]:
    """Analyze code file and extract functions into LLMTextRepresentation objects."""
    agent = Agent(
        'openai:gpt-4o',
        output_type=list[LLMTextRepresentation],
        system_prompt='You are a code analyzer. Analyze the provided code and extract all functions, returning a list of LLMTextRepresentation objects with function_name, input_args, output_args, and description for each function found.',
        instrument=True,
    )

    prompt = f"""
    Analyze this code file and extract all functions:

    {file_content}

    For each function found, create an LLMTextRepresentation with:
    - class_name: the name of the class (if function is a method) or null for root functions
    - function_name: the name of the function
    - visibility: "public" (no underscore), "private" (starts with _), or "protected" (starts with __)
    - input_args: list of Argument objects with arg_name and description for each parameter
    - output_args: list of return values with types
    - description: what the function does
    - complexity: assess as "low", "medium", or "high" based on cyclomatic complexity
    - dependencies: list of external libraries/modules imported and used
    - side_effects: list any file I/O, network calls, database operations, global state changes
    - error_handling: list types of exceptions handled (try/catch blocks)
    - decorators: list any decorators used (@property, @staticmethod, @classmethod, etc.)

    Return a list of all functions found.
    """

    result = agent.run_sync(prompt)
    return result.output


def summarize_directory(directory_path: str) -> dict[str, list[LLMTextRepresentation]]:
    """Analyze all code files in a directory and return summarized functions for each file."""
    directory = Path(directory_path)

    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {directory_path} does not exist or is not a directory")

    results = {}

    # Common code file extensions
    code_extensions = {'.py'}

    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix in code_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                # Get relative path for cleaner output
                relative_path = str(file_path.relative_to(directory))
                results[relative_path] = summarize_code(file_content)

            except (UnicodeDecodeError, PermissionError) as e:
                logfire.info(f"Skipping {file_path}: {e}")
                continue

    return results

if __name__ == "__main__":
    # Summarize all Python files in the pydantic_ai models directory
    directory_path = "/Users/jeremychua/dev/pydantic-ai/pydantic_ai_slim/pydantic_ai/models"
    results = summarize_directory(directory_path)

    for file_path, functions in results.items():
        print(f"\n=== {file_path} ===")
        for func in functions:
            print(f"class name: {func.class_name}")
            print(f"function name: {func.function_name}")
            print(f"input args: {func.input_args}")
            print(f"output args: {func.output_args}")
            print(f"description: {func.description}")
            print("\n-----\n")

