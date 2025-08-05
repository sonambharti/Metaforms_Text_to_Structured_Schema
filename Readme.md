# Unstructured Text to JSON Converter

This project is a Python-based system that converts unstructured plain text into a structured JSON format, strictly adhering to a provided JSON schema. It uses a Retrieval-Augmented Generation (RAG) pipeline with a free Large Language Model (LLM) from Groq to handle large and complex inputs efficiently.

## Features

-   **RAG Pipeline:** Uses a sophisticated RAG architecture to handle large schemas that wouldn't fit in a standard LLM context window.
-   **Iterative Extraction:** Processes text by identifying relevant sections and querying a schema index for targeted, accurate data extraction.
-   **Schema Validation:** Strictly validates the final JSON output against the provided schema to guarantee correctness.
-   **Free LLM:** Powered by Groq and Llama 3, providing high-speed and no-cost inference.
-   **CLI Interface:** Simple and easy-to-use command-line interface.


## Project Set up

```
# Install the required packages
pip install -r requirements.txt

# Set GROQ_API_KEY
set GROQ_API_KEY="your_api_key_here"

# Execute this code
python main.py --text-file "./tests/test-case-2/github actions sample input.txt" --schema-file "./tests/test-case-2/github_actions_schema.json"
```