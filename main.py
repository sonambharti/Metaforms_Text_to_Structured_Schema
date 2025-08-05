import os
import json
import click
from typing import List, Dict, Any

import chromadb
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings # CORRECTED IMPORT

# --- CONFIGURATION ---
os.environ["GROQ_API_KEY"] = "" # Your_GROQ_API_KEY
# Initialize the LLM from Groq. Llama3 70b is powerful and free on Groq.
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)

# Initialize the embedding model. This runs locally and is free.
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# CORRECTED EMBEDDING INSTANTIATION
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- STAGE 1: SCHEMA INDEXING ---

def get_schema_chunks(schema: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
    """
    Recursively chunks a JSON schema into a list of documents.
    Each document describes a property and its sub-properties.

    Args:
        schema: The JSON schema dictionary.
        path: The current JSON path (used for recursion).

    Returns:
        A list of dictionaries, where each is a chunk to be embedded.
    """
    chunks = []
    properties = schema.get("properties", {})

    for key, value in properties.items():
        current_path = f"{path}.{key}" if path else key
        description = value.get("description", "")
        prop_type = value.get("type", "N/A")
        
        chunk_text = f"Property Name: `{key}`. Description: {description}. Type: {prop_type}."

        # Add details for sub-properties if it's an object
        if prop_type == "object" and "properties" in value:
            sub_props = ", ".join(value["properties"].keys())
            chunk_text += f" It contains sub-properties: {sub_props}."
        
        # Add enum values if they exist
        if "enum" in value:
            enum_vals = ", ".join(map(str, value["enum"]))
            chunk_text += f" Accepted values are: {enum_vals}."
            
        chunks.append({
            "path": current_path,
            "key": key,
            "content": chunk_text
        })

        # Recurse for nested objects
        if prop_type == "object":
            chunks.extend(get_schema_chunks(value, current_path))
            
    return chunks

def create_or_load_vector_store(schema_chunks: List[Dict[str, Any]], store_path: str = "./chroma_db"):
    """
    Creates a ChromaDB vector store from schema chunks or loads it if it exists.
    """
    if os.path.exists(store_path):
        print("Loading existing vector store...")
        vector_store = Chroma(persist_directory=store_path, embedding_function=embedding_function)
    else:
        print("Creating new vector store...")
        documents = [chunk["content"] for chunk in schema_chunks]
        metadatas = [{"key": chunk["key"], "path": chunk["path"]} for chunk in schema_chunks]
        
        vector_store = Chroma.from_texts(
            texts=documents,
            embedding=embedding_function,
            metadatas=metadatas,                                                                                                                                                         
            persist_directory=store_path
        )
        vector_store.persist()
    return vector_store

# --- STAGE 2: ITERATIVE EXTRACTION ---

def postprocess_extracted_value(key, extracted_value, schema_dict):
    """
    Ensures that for primitive keys, only the value is extracted, not the whole object.
    """
    expected_type = schema_dict["properties"][key].get("type")
    # Handle array of types (e.g., ["string", "null"])
    if isinstance(expected_type, list):
        if "string" in expected_type:
            expected_type = "string"
        elif "number" in expected_type:
            expected_type = "number"
        elif "boolean" in expected_type:
            expected_type = "boolean"
        elif "object" in expected_type:
            expected_type = "object"
        elif "array" in expected_type:
            expected_type = "array"
    if expected_type in ["string", "number", "boolean"] and isinstance(extracted_value, dict):
        # If the LLM returned an object, but we want a primitive, extract the value
        if key in extracted_value:
            return extracted_value[key]
        # Or, try to get the first primitive value
        for v in extracted_value.values():
            if isinstance(v, (str, int, float, bool)):
                return v
    return extracted_value

def get_top_level_keys(text_content: str, schema: Dict[str, Any]) -> List[str]:
    """
    First-pass LLM call to identify which top-level keys are described in the text.
    """
    print("Identifying relevant top-level keys...")
    
    # Extracting top-level keys from the schema to guide the LLM
    top_level_keys = list(schema.get("properties", {}).keys())
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Your task is to identify which of the provided schema keys are mentioned or described in the user's text. Respond with a simple comma-separated list of the keys."),
        ("user", "Here are the possible top-level schema keys: {keys}\n\nHere is the text:\n---\n{text}\n\n---\nWhich of these keys are present in the text? Respond with ONLY a comma-separated list.")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    
    response = chain.invoke({
        "keys": ", ".join(top_level_keys),
        "text": text_content
    })
    
    # Clean up the response
    keys = [key.strip() for key in response.split(',') if key.strip()]
    print(f"Found keys: {keys}")
    return keys

def extract_json_for_key(text_content: str, key_to_extract: str, retriever) -> Any:
    """
    The core RAG step. Retrieves relevant schema and extracts JSON for a specific key.
    """
    print(f"Running RAG for key: '{key_to_extract}'...")
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a world-class expert in structured data extraction. Based on the full text and the provided relevant schema for a specific key, extract the information for that key. Output ONLY the raw, valid JSON for that key's value. Do not add any commentary, explanations, or markdown formatting like ```json."),
        ("user", """
        **Full Text Context:**
        ---
        {text}
        ---

        **Relevant Schema for key '{key}':**
        ---
        {schema_context}
        ---

        **Instruction:**
        Based on the full text, extract the data for the key **'{key}'**. Output only its value as a valid JSON object, array, or primitive.
        """)
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"schema_context": retriever | format_docs, "text": RunnablePassthrough(), "key": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    # Invoke the chain with the full text content and the key to extract
    response_str = rag_chain.invoke(text_content, config={"run_name": "ExtractJSONForKey", "tags": [f"key:{key_to_extract}"]})
    
    # Try to parse the LLM's string response into a Python object
    try:
        # Basic cleanup for markdown
        if response_str.startswith("```json"):
            response_str = response_str[7:].strip()
        if response_str.endswith("```"):
            response_str = response_str[:-3].strip()
            
        return json.loads(response_str)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON for key '{key_to_extract}'. Raw response: {response_str}")
        return None


# --- MAIN ORCHESTRATION & CLI ---

@click.command()
@click.option('--text-file', required=True, type=click.Path(exists=True), help='Path to the unstructured text file.')
@click.option('--schema-file', required=True, type=click.Path(exists=True), help='Path to the JSON schema file.')
def main(text_file, schema_file):
    """
    An end-to-end RAG pipeline to convert unstructured text to structured JSON
    based on a provided schema, using a free LLM.
    """
    # Load inputs
    with open(text_file, 'r') as f:
        text_content = f.read()
    with open(schema_file, 'r') as f:
        schema_dict = json.load(f)

    # --- Stage 1: Indexing ---
    schema_chunks = get_schema_chunks(schema_dict)
    vector_store = create_or_load_vector_store(schema_chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 most relevant chunks

    # --- Stage 2: Extraction ---
    final_json = {}
    keys_to_process = get_top_level_keys(text_content, schema_dict)

    for key in keys_to_process:
        # Check if key is valid in the schema before processing
        if key in schema_dict.get("properties", {}):
            extracted_value = extract_json_for_key(text_content, key, retriever)
            if extracted_value is not None:
                processed_value = postprocess_extracted_value(key, extracted_value, schema_dict)
                final_json[key] = processed_value
        else:
            print(f"Warning: LLM returned key '{key}' which is not in the schema's top-level properties. Skipping.")

    # --- Stage 3: Validation & Output ---
    print("\n--- FINAL VALIDATION ---")
    try:
        validate(instance=final_json, schema=schema_dict)
        print("✅ Success: The generated JSON is valid against the schema.")
    except ValidationError as e:
        print(f"❌ Error: The generated JSON is INVALID against the schema.")
        print(e)
        # Even if invalid, print the generated JSON for debugging
    
    print("\n--- GENERATED JSON OUTPUT ---")
    print(json.dumps(final_json, indent=2))
    # --- This block replaces the original print statement ---

    # 1. Define the output directory
    output_dir = "output"

    # 2. Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 3. Get the current time and format it for the filename
    # Format: YYYY-MM-DD_HH-MM-SS
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"output_{timestamp}.json"

    # 4. Construct the full file path
    file_path = os.path.join(output_dir, file_name)

    # 5. Write the JSON data to the new file
    with open(file_path, 'w') as f:
        json.dump(final_json, f, indent=2)

    # 6. Print a final confirmation message to the console
    print(f"\n✅ Successfully saved output to: {file_path}")


if __name__ == '__main__':
    main()