from pydantic import BaseModel
from pydantic_ai import Agent
import logfire

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

if __name__ == "__main__":
    # Test the summarize_code function with a sample Python file
    with open("sample_file.py") as f:
        sample_code = f.read()

    result = summarize_code(sample_code)

    for r in result:
        print(f"class name: {r.class_name}")
        print(f"function name: {r.function_name}")
        print(f"input args: {r.input_args}")
        print(f"output args: {r.output_args}")
        print(f"description: {r.description}")
        print("\n\n-----\n\n")

