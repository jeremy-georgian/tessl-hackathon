from pydantic import BaseModel
from pydantic_ai import Agent
import logfire
import os
from pathlib import Path
import turbopuffer
import sqlite3
import json
import hashlib
import asyncio
from openai import AsyncOpenAI

tpuf = turbopuffer.Turbopuffer(
    # API tokens are created in the dashboard: https://turbopuffer.com/dashboard
    api_key=os.getenv("TURBOPUFFER_API_KEY"),
    # Pick the right region: https://turbopuffer.com/docs/regions
    region="gcp-us-central1",
)

ns = tpuf.namespace(f'tessl-hackathon-{os.getenv("USER")}')

# Initialize OpenAI client for embeddings
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize SQLite cache database
def init_cache_db():
    """Initialize the SQLite cache database with the required schema."""
    conn = sqlite3.connect('function_cache.db')
    cursor = conn.cursor()

    # Create table for cached function representations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS function_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            class_name TEXT,
            function_name TEXT NOT NULL,
            visibility TEXT NOT NULL,
            input_args TEXT NOT NULL,  -- JSON string
            output_args TEXT NOT NULL,  -- JSON string
            description TEXT NOT NULL,
            complexity TEXT NOT NULL,
            dependencies TEXT NOT NULL,  -- JSON string
            side_effects TEXT NOT NULL,  -- JSON string
            error_handling TEXT NOT NULL,  -- JSON string
            decorators TEXT NOT NULL,  -- JSON string
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(file_path, file_hash, function_name, class_name)
        )
    ''')

    conn.commit()
    conn.close()

# Initialize the database
init_cache_db()

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


def summarize_code(file_content: str, file_path: str = None) -> list[LLMTextRepresentation]:
    """Analyze code file and extract functions into LLMTextRepresentation objects."""
    # Check cache first if file_path is provided
    if file_path:
        file_hash = get_file_hash(file_content)
        cached_functions = get_cached_functions(file_path, file_hash)
        if cached_functions:
            logfire.info(f"Using cached results for {file_path}")
            return cached_functions

    logfire.info(f"Analyzing code for {file_path or 'unknown file'}")
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
    functions = result.output

    # Save to cache if file_path is provided
    if file_path:
        file_hash = get_file_hash(file_content)
        save_functions_to_cache(file_path, file_hash, functions)

    return functions


async def summarize_code_async(file_content: str, file_path: str = None) -> list[LLMTextRepresentation]:
    """Async version of summarize_code that analyzes code file and extracts functions."""
    # Check cache first if file_path is provided
    if file_path:
        file_hash = get_file_hash(file_content)
        cached_functions = get_cached_functions(file_path, file_hash)
        if cached_functions:
            logfire.info(f"Using cached results for {file_path}")
            return cached_functions

    logfire.info(f"Analyzing code for {file_path or 'unknown file'}")
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

    result = await agent.run(prompt)
    functions = result.output

    # Save to cache if file_path is provided
    if file_path:
        file_hash = get_file_hash(file_content)
        save_functions_to_cache(file_path, file_hash, functions)

    return functions


async def create_embedding_from_description(description: str) -> list[float]:
    """Generate embedding from function description using OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=description
        )
        return response.data[0].embedding
    except Exception as e:
        logfire.error(f"Error generating embedding: {e}")
        raise


async def save_function_to_turbopuffer(func: LLMTextRepresentation, file_path: str):
    """Save function representation and its description embedding to TurboPuffer."""
    # Create unique ID for this function using full file path
    func_full_id = f"{file_path}#{func.class_name or 'root'}#{func.function_name}"
    
    # Create a hash that fits TurboPuffer's 64-byte limitation
    func_id = hashlib.sha256(func_full_id.encode('utf-8')).hexdigest()[:63]  # 63 chars to be safe

    # Generate embedding from description
    embedding = await create_embedding_from_description(func.description)
    
    # Create the vector record with minimal metadata
    vector_data = {
        "id": func_id,
        "vector": embedding,
        "file_path": file_path,
        "class_name": func.class_name,
        "function_name": func.function_name,
    }

    try:
        # Use write() with upsert_rows as per TurboPuffer API (run in thread pool for async)
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ns.write(
                upsert_rows=[vector_data],
                distance_metric='cosine_distance',
                schema={
                    "file_path": {"type": "string"},
                    "class_name": {"type": "string"},
                    "function_name": {"type": "string"},
                }
            )
        )
        logfire.info(f"Saved embedding for {func_id} to TurboPuffer")
    except Exception as e:
        logfire.error(f"Error saving to TurboPuffer: {e}")
        raise


def get_file_hash(file_content: str) -> str:
    """Generate a hash of the file content for cache invalidation."""
    return hashlib.md5(file_content.encode('utf-8')).hexdigest()


def is_file_cached(file_path: str, file_hash: str) -> bool:
    """Check if a file with the given hash is already cached."""
    conn = sqlite3.connect('function_cache.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT COUNT(*) FROM function_cache
        WHERE file_path = ? AND file_hash = ?
    ''', (file_path, file_hash))

    count = cursor.fetchone()[0]
    conn.close()

    return count > 0


def get_cached_functions(file_path: str, file_hash: str) -> list[LLMTextRepresentation] | None:
    """Check if functions for this file are already cached."""
    conn = sqlite3.connect('function_cache.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT class_name, function_name, visibility, input_args, output_args,
               description, complexity, dependencies, side_effects, error_handling, decorators
        FROM function_cache
        WHERE file_path = ? AND file_hash = ?
    ''', (file_path, file_hash))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    functions = []
    for row in rows:
        class_name, function_name, visibility, input_args, output_args, description, complexity, dependencies, side_effects, error_handling, decorators = row

        # Parse JSON fields back to lists
        input_args_list = [Argument(**arg) for arg in json.loads(input_args)]

        functions.append(LLMTextRepresentation(
            class_name=class_name,
            function_name=function_name,
            visibility=visibility,
            input_args=input_args_list,
            output_args=json.loads(output_args),
            description=description,
            complexity=complexity,
            dependencies=json.loads(dependencies),
            side_effects=json.loads(side_effects),
            error_handling=json.loads(error_handling),
            decorators=json.loads(decorators)
        ))

    return functions


def save_functions_to_cache(file_path: str, file_hash: str, functions: list[LLMTextRepresentation]):
    """Save functions to the cache database."""
    conn = sqlite3.connect('function_cache.db')
    cursor = conn.cursor()

    for func in functions:
        # Convert lists to JSON strings for storage
        input_args_json = json.dumps([{"arg_name": arg.arg_name, "description": arg.description} for arg in func.input_args])

        cursor.execute('''
            INSERT OR REPLACE INTO function_cache
            (file_path, file_hash, class_name, function_name, visibility, input_args,
             output_args, description, complexity, dependencies, side_effects, error_handling, decorators)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_path, file_hash, func.class_name, func.function_name, func.visibility,
            input_args_json, json.dumps(func.output_args), func.description, func.complexity,
            json.dumps(func.dependencies), json.dumps(func.side_effects),
            json.dumps(func.error_handling), json.dumps(func.decorators)
        ))

    conn.commit()
    conn.close()


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
                # First, read just enough to generate a hash and check cache
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                file_hash = get_file_hash(file_content)

                # Check if this exact file content is already cached
                if is_file_cached(str(file_path), file_hash):
                    logfire.info(f"File {file_path} is already cached, retrieving from cache")
                    cached_functions = get_cached_functions(str(file_path), file_hash)
                    if cached_functions:
                        relative_path = str(file_path.relative_to(directory))
                        results[relative_path] = cached_functions
                        continue

                # If not cached, proceed with analysis
                relative_path = str(file_path.relative_to(directory))
                results[relative_path] = summarize_code(file_content, str(file_path))

            except (UnicodeDecodeError, PermissionError) as e:
                logfire.info(f"Skipping {file_path}: {e}")
                continue

    return results


async def process_single_file(file_path: Path, directory: Path, semaphore: asyncio.Semaphore) -> tuple[str, list[LLMTextRepresentation]] | None:
    """Process a single file with semaphore for concurrency control."""
    async with semaphore:
        try:
            # Read file content and generate hash
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            file_hash = get_file_hash(file_content)

            # Check if this exact file content is already cached
            if is_file_cached(str(file_path), file_hash):
                logfire.info(f"File {file_path} is already cached, retrieving from cache")
                cached_functions = get_cached_functions(str(file_path), file_hash)
                if cached_functions:
                    relative_path = str(file_path.relative_to(directory))
                    functions = cached_functions
                else:
                    # If cache check failed, proceed with analysis
                    relative_path = str(file_path.relative_to(directory))
                    functions = await summarize_code_async(file_content, str(file_path))
            else:
                # If not cached, proceed with async analysis
                relative_path = str(file_path.relative_to(directory))
                functions = await summarize_code_async(file_content, str(file_path))

            # Save each function to TurboPuffer with embeddings (happens for both cached and new)
            for func in functions:
                try:
                    await save_function_to_turbopuffer(func, str(file_path))
                except Exception as e:
                    logfire.error(f"Failed to save {func.function_name} to TurboPuffer: {e}")

            return relative_path, functions

        except (UnicodeDecodeError, PermissionError) as e:
            logfire.info(f"Skipping {file_path}: {e}")
            return None


async def summarize_directory_async(directory_path: str, max_concurrent: int = 4) -> dict[str, list[LLMTextRepresentation]]:
    """Async version that analyzes all code files in a directory with configurable concurrency."""
    directory = Path(directory_path)

    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {directory_path} does not exist or is not a directory")

    # Common code file extensions
    code_extensions = {'.py'}

    # Find all code files
    code_files = [
        file_path for file_path in directory.rglob('*')
        if file_path.is_file() and file_path.suffix in code_extensions
    ]

    logfire.info(f"Found {len(code_files)} code files to process with max_concurrent={max_concurrent}")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process files concurrently
    tasks = [
        process_single_file(file_path, directory, semaphore)
        for file_path in code_files
    ]

    # Wait for all tasks to complete
    results_list = await asyncio.gather(*tasks)

    # Convert results to dictionary, filtering out None results
    results = {}
    for result in results_list:
        if result is not None:
            relative_path, functions = result
            results[relative_path] = functions

    return results

async def main():
    """Main async function to run directory summarization."""
    # Summarize all Python files in the pydantic_ai models directory
    directory_path = "/Users/jeremychua/dev/pydantic-ai/pydantic_ai_slim/pydantic_ai/models"

    # Use async version with configurable concurrency (default 4)
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "4"))
    logfire.info(f"Starting analysis with max_concurrent={max_concurrent}")

    results = await summarize_directory_async(directory_path, max_concurrent=max_concurrent)

    for file_path, functions in results.items():
        print(f"\n=== {file_path} ===")
        for func in functions:
            print(f"class name: {func.class_name}")
            print(f"function name: {func.function_name}")
            print(f"input args: {func.input_args}")
            print(f"output args: {func.output_args}")
            print(f"description: {func.description}")
            print("\n-----\n")


if __name__ == "__main__":
    asyncio.run(main())

