# AI Solution Design: Unstructured Text to Structured JSON

## 1. Goal

To design and prototype a system that converts unstructured plain text into a structured format, strictly adhering to a desired JSON schema. The system must be robust enough to handle large and complex inputs.

## 2. Core Requirements & Constraints

The solution was developed to meet the following key requirements:

* **P1. Logging:** Maintain a log of decisions, experiments, and insights.
* **P2. Functional Correctness:** The solution must work on sample test cases and produce outputs that strictly adhere to their corresponding schemas.
* **P3. Large Context Support:** The system must be able to handle large inputs, including text files up to 50k tokens and schema files up to 100k tokens.
* **Cost Constraint:** Utilize a Free Large Language Model (LLM) API.
* **Architectural Constraint:** Implement the solution using a pure Retrieval-Augmented Generation (RAG) pipeline.

## 3. System Architecture: The RAG Pipeline

To meet the dual constraints of supporting large schemas with a free LLM (which typically have smaller context windows), a Retrieval-Augmented Generation (RAG) architecture was chosen. This approach breaks the problem down into smaller, more manageable pieces, leading to higher accuracy and efficiency.

The system operates in two main stages:

### Stage A: Offline Schema Indexing (Once per Schema)

This stage prepares the JSON schema for efficient retrieval.

1.  **Load Schema:** The input JSON schema file is loaded into memory.
2.  **Custom Chunking:** A custom `JsonSchemaSplitter` traverses the schema. For each top-level property (e.g., `name`, `inputs`, `runs`), it creates a descriptive text document. This document includes the property's name, its `description` from the schema, its `type`, and any sub-properties or `enum` values.
3.  **Embedding:** Each text chunk is converted into a vector embedding using a sentence-transformer model (`all-MiniLM-L6-v2`).
4.  **Store in Vector DB:** The text chunks and their corresponding vector embeddings are stored in a local ChromaDB vector store. This creates a searchable index of the schema's components.

### Stage B: Online Iterative Extraction (Per Input Text)

This is the main application flow for converting a given text file.

1.  **Identify Top-Level Keys:** A preliminary LLM call reads the entire input text and identifies which of the schema's main properties are being described. This creates a worklist (e.g., `['name', 'author', 'runs', 'branding']`).
2.  **Iterate and Extract:** The system loops through each identified key. For each key:
    * **Retrieve (R):** The key itself (e.g., "runs") is used as a query to the ChromaDB vector store. The most relevant schema chunk(s) are retrieved.
    * **Augment (A):** A highly specific, focused prompt is constructed. This prompt contains the full original text, the *retrieved schema chunk* (not the whole schema), and a strict instruction for the LLM.
    * **Generate (G):** The LLM (Groq/Llama 3) receives this augmented prompt and generates **only the JSON value** for that specific key.
3.  **Assemble:** The extracted values are assembled into a single Python dictionary.
4.  **Validate:** The final, complete JSON object is validated against the original, full JSON schema file using the `jsonschema` library. This ensures the "strictly following" requirement is met.

## 4. Technology Stack

* **Language:** Python 3.9+
* **LLM:** Groq API with `llama3-70b-8192` (Provides free, high-speed access to a powerful open-source model).
* **Orchestration:** LangChain (Used for its Expression Language to build the RAG chain).
* **Vector Store:** ChromaDB (An open-source, in-memory vector database, easy to set up locally).
* **Embedding Model:** `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2` (A high-performance, free model that runs locally).
* **CLI:** `click` (For creating a clean and user-friendly command-line interface).
* **Validation:** `jsonschema` (To programmatically validate the final output).