import ollama
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
# Choose your model (SmolLM3 3B is recommended for the project) [cite: 31, 38]
CHOSEN_MODEL = 'qwen3:4b'  # Local model
# CHOSEN_MODEL = 'qwen3-coder:480b-cloud'  # For Ollama Cloud 

# Choose your host
# Local Docker: http://ollama_server:11434 
# Ollama Cloud: https://api.ollama.com (example endpoint) 
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://ollama_server:11434')

client = ollama.Client(host=OLLAMA_HOST)

def test_api_connection():
    """Diagnostic test to verify the API and model availability."""
    print(f"[*] Testing connection to: {OLLAMA_HOST}")
    try:
        # Check if service is up 
        models = client.list()
        print("[+] API is reachable.")
        
        # Check if the specific model is pulled 
        available_models = [m['name'] for m in models['models']]
        if CHOSEN_MODEL in available_models or "cloud" in CHOSEN_MODEL:
            print(f"[+] Model '{CHOSEN_MODEL}' is ready for use.")
            return True
        else:
            print(f"[-] Model '{CHOSEN_MODEL}' not found locally. Please run: ollama pull {CHOSEN_MODEL}")
            return False
    except Exception as e:
        print(f"[!] Connection failed: {e}")
        return False

def ask_llm(text_chunk, sub_query):
    """Recursive tool for processing chunks[cite: 17, 21]."""
    print(f"    [Task] Analyzing {len(text_chunk)} characters...")
    resp = client.chat(
        model=CHOSEN_MODEL,
        messages=[{'role': 'user', 'content': f"Text: {text_chunk}\n\nTask: {sub_query}"}]
    )
    return resp['message']['content']

def run_rlm_logic(document_text, query):
    """Executes the RLM inference-time strategy[cite: 18, 20]."""
    print(f"[*] Starting RLM on {CHOSEN_MODEL}...")
    
    repl_env = {
        "document": document_text, 
        "ask_llm": ask_llm, 
        "result": None
    }
    
    # Prompting for code-only output to avoid syntax errors [cite: 20]
    system_prompt = (
        "You are an RLM Python agent. Access the 'document' variable. "
        "Use 'ask_llm(chunk, query)' to solve the problem. "
        "Output ONLY Python code. No text. Store the answer in 'result'."
    )

    response = client.chat(
        model=CHOSEN_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ]
    )
    
    # Clean the code block [cite: 20]
    clean_code = re.sub(r"```(?:python)?", "", response['message']['content']).replace("```", "").strip()
    
    try:
        exec(clean_code, repl_env)
        return repl_env.get("result")
    except Exception as e:
        return f"Execution Error: {e}"

if __name__ == "__main__":
    if test_api_connection():
        # Needle in a haystack test 
        test_doc = "noise " * 2000 + "SECRET: IMT_2026" + " noise " * 2000
        print("\n[*] Running RLM Logic...")
        res = run_rlm_logic(test_doc, "Search the document for the SECRET value.")
        print(f"\n[!] FINAL RESULT: {res}")
