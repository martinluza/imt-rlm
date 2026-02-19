import ollama
import os
import time
import sys
import re

# Connection Setup
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://ollama_server:11434')
client = ollama.Client(host=OLLAMA_HOST)
# Using Qwen3 as the reasoning engine
MODEL = 'qwen3:4b' 

def wait_for_ollama():
    print(f"[*] Connecting to Ollama at {OLLAMA_HOST}...")
    for i in range(10):
        try:
            client.list()
            print("[+] Connected successfully!")
            return
        except Exception:
            print(f"[-] Waiting for Ollama... ({i+1}/10)")
            time.sleep(3)
    sys.exit(1)

def ask_llm(text_chunk, sub_query):
    """The recursive delegation tool: LLM calling itself on chunks."""
    print(f"    [Recursive Call] Processing {len(text_chunk)} chars...")
    resp = client.chat(
        model=MODEL,
	messages=[{
            'role': 'user', 
            'content': f"CONTEXT: {text_chunk}\n\nTASK: {sub_query}\n\nINSTRUCTION: If found, output the value ONLY. If not found, output 'NOT_FOUND'."
        }]
    )
    return resp['message']['content']


def run_rlm_recursive_logic(document_text, query):
    print(f"[*] Analyzing document ({len(document_text)} chars) using {MODEL}...")
    
    # Unified REPL environment
    repl_env = {
        "document": document_text, 
        "ask_llm": ask_llm, 
        "result": None,
        "print": print  # Allow the model to print for debugging
    }
    
    system_prompt = (
	"You are a Recursive Language Model Senior Manager. "
        "The document is too large for your memory. Write Python code to:\n"
        "1. Slice `document` into chunks.\n"
        "2. Call `ask_llm(chunk, 'Search for the secret key')` on each.\n"
        "3. If the response is not 'NOT_FOUND', save it to the `result` variable and BREAK the loop.\n"
        "Output ONLY code."
    )

    response = client.chat(
        model=MODEL, 
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ]
    )
    
    clean_code = re.sub(r"```(?:python)?", "", response['message']['content']).replace("```", "").strip()
    
    print(f"--- EXECUTING QWEN3 STRATEGY ---\n{clean_code}\n---")

    try:
        # Using a single dict solves the 'NameError' for variables like 'document'
        exec(clean_code, repl_env) 
        return repl_env.get("result")
    except Exception as e:
        return f"Execution Error: {e}"

if __name__ == "__main__":
    wait_for_ollama()
    
    # Toy 'needle in a haystack' experiment [cite: 31]
    haystack = "Irrelevant text. " * 1500 + "KEY_FOUND: IMT_ATLANTIQUE" + " More noise. " * 1500
    
    # Task designed to trigger recursive behavior 
    task = "Find the KEY_FOUND value. Use a loop to slice the document into 4000-char chunks and use ask_llm on each."
    
    final_answer = run_rlm_recursive_logic(haystack, task)
    print(f"\n[!] RLM RESULT: {final_answer}")
