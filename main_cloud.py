import os
import time
import re
from openai import OpenAI

# --- CLOUD API CONFIGURATION ---
# The OpenAI client is the universal standard for Cloud APIs (OpenRouter, Together, Groq, etc.)
# You can use OpenRouter to access SmolLM3 or Qwen3 for free/cheap.
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "your-cloud-api-key-here")
CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://api.openrouter.ai/v1") 
MODEL_NAME = os.getenv("MODEL_NAME", "huggingface/smollm:1.7b") # Small open-weight LM

# Initialize the Cloud Client
client = OpenAI(
    api_key=CLOUD_API_KEY,
    base_url=CLOUD_BASE_URL
)

def ask_llm(text_chunk, sub_query):
    """
    Cloud-based recursive delegation tool.
    Acts as the 'junior team member' analyzing a specific chunk.
    """
    print(f"    [Cloud Sub-Task] Analyzing chunk ({len(text_chunk)} chars)...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a sub-task worker. Answer strictly and concisely."},
                {"role": "user", "content": f"CONTEXT:\n{text_chunk}\n\nTASK: {sub_query}\n\nINSTRUCTION: If found, output the exact value ONLY. If not found, output 'NOT_FOUND'."}
            ],
            temperature=0.0 # Low temperature for analytical extraction
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [!] Cloud API Error on sub-task: {e}")
        return "NOT_FOUND"

def run_rlm_cloud(document_text, query):
    """
    The RLM Manager node. It writes the inference-time execution strategy.
    """
    print(f"[*] Analyzing document ({len(document_text)} chars) via Cloud API ({MODEL_NAME})...")
    
    # Unified REPL environment containing the Cloud Tool
    repl_env = {
        "document": document_text,
        "ask_llm": ask_llm,
        "result": None,
        "print": print
    }

    # Strict prompt to force the Senior Manager to write code
    system_prompt = (
        "You are a Recursive Language Model Senior Manager. "
        "The document is too large for your memory. Write Python code to:\n"
        "1. Slice the `document` string into 4000-character chunks.\n"
        "2. Call `ask_llm(chunk, 'Search for the secret key')` on each chunk.\n"
        "3. If the response is not 'NOT_FOUND', save it to the `result` variable and break the loop.\n"
        "Output ONLY valid Python code. No text, no markdown backticks."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1
        )
        raw_content = response.choices[0].message.content
    except Exception as e:
        return f"Manager API Error: {e}"

    # Robust parsing to handle any markdown the cloud model might inject
    clean_code = re.sub(r"```(?:python)?\n?", "", raw_content).replace("```", "").strip()

    print(f"--- EXECUTING CLOUD RLM STRATEGY ---\n{clean_code}\n---")

    try:
        # Execute the generated logic in the isolated REPL environment
        exec(clean_code, repl_env)
        return repl_env.get("result")
    except Exception as e:
        return f"Execution Error: {e}\nGenerated Code:\n{clean_code}"

if __name__ == "__main__":
    print("--- RLM Cloud Implementation ---")
    
    # Toy "needle in a haystack" experiment
    # Creating a massive document that bypasses standard context windows
    haystack = "Irrelevant data. " * 2000 + "KEY_FOUND: IMT_ATL_2026_CLOUD" + " More noise. " * 2000
    
    task = "Find the KEY_FOUND value in the document."
    
    start_time = time.time()
    final_answer = run_rlm_cloud(haystack, task)
    elapsed = time.time() - start_time
    
    print(f"\n[!] FINAL RLM RESULT: {final_answer}")
    print(f"[*] Total Cloud Inference Time: {elapsed:.2f} seconds")
