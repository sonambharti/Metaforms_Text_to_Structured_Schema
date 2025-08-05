# Implementation Log

This log documents the decisions, experiments, and insights gained during the development of the text-to-JSON conversion system.

**Date: 2025-08-04**
* **Project Inception:** Initial goal defined: Convert unstructured text to JSON based on a schema.
* **Initial Analysis:** Key challenge identified is P3: supporting massive schema files (100k+ tokens). Initial thought is to use a large context window model like Gemini 1.5 Pro and a single-prompt approach.
* **Constraint Pivot:** User introduced critical new constraints: must use a **Free LLM API** and a **pure RAG pipeline**.
* **Insight:** The "Free LLM" constraint makes the single-prompt approach unfeasible, as free models typically have smaller context windows than the required 150k+ tokens. The RAG architecture is now not just an option but a necessity.
* **Technology Selection:**
    * **LLM:** Chose Groq with Llama 3 for its speed, capability, and free access.
    * **RAG Stack:** Selected LangChain for orchestration, ChromaDB for a simple local vector store, and `Sentence-Transformers` for local embeddings.

**Date: 2025-08-05**
* **Design & Implementation:**
    * Designed the two-stage RAG pipeline (Offline Indexing, Online Extraction).
    * Implemented a custom `get_schema_chunks` function to traverse the JSON schema and create meaningful documents for the vector store.
    * Implemented the iterative extraction loop.

* **Experiment 1: First Run & Failure**
    * **Observation:** The first implementation failed validation. The generated JSON was nonsensically nested. For every key (`name`, `author`, etc.), the LLM was returning the *entire* JSON object.
    * **Insight:** The prompt for the RAG loop was not strict enough. The LLM was trying to be "overly helpful" by completing the whole task at once, ignoring the instruction to focus on a single key.
    * **Action:** Rewrote the RAG prompt to be much more strict, providing clear examples of the expected output format (e.g., "if the key is 'name', respond with just 'My Action'").

* **Experiment 2: Second Run & `AttributeError`**
    * **Observation:** The updated code threw an `AttributeError: 'Chroma' object has no attribute 'page_content'`.
    * **Insight:** The LangChain Expression Language (LCEL) chain was structured incorrectly. The `format_docs` function was being applied at the wrong stage, causing a type mismatch.
    * **Action:** Restructured the `rag_chain` to correctly pipe the output of the `retriever` into the `format_docs` function before it was passed to the prompt.

* **Experiment 3: Third Run & Second `AttributeError`**
    * **Observation:** The code now threw `AttributeError: 'dict' object has no attribute 'replace'`.
    * **Insight:** The traceback showed the error was in the embedding model's code. The `retriever` was being passed the entire input dictionary from the chain, instead of the simple string query it expected.
    * **Action:** Corrected the chain again, using a `RunnableLambda` to select only the `key` from the input dictionary and pass that string to the retriever.

* **Final Success:**
    * **Observation:** The code now successfully processes the given test case, produces a correctly structured JSON, and passes validation.
    * **Insight:** The iterative RAG approach, combined with highly specific and strict prompting, is an effective way to solve this problem under the given constraints. The debugging cycle was essential for refining the LCEL chain structure and prompt engineering.