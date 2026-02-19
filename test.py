import ollama
import os
import sys

# Configuration - Match your docker-compose or cloud settings
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
# Use the model you integrated (qwen3:4b or smollm:1.7b)
MODEL_NAME = 'qwen3:4b' 

def run_diagnostics():
    print(f"--- RLM Environment Diagnostic ---")
    print(f"[*] Target Host: {OLLAMA_HOST}")
    print(f"[*] Target Model: {MODEL_NAME}")
    
    client = ollama.Client(host=OLLAMA_HOST)

    # 1. Check API Reachability
    try:
        models_info = client.list()
        print("[+] SUCCESS: Ollama API is reachable.")
    except Exception as e:
        print(f"[!] ERROR: Cannot connect to Ollama. Is the container running?")
        print(f"    Details: {e}")
        sys.exit(1)

    # 2. Check if Model is Pulled
    available = [m['name'] for m in models_info['models']]
    if MODEL_NAME in available or any(MODEL_NAME in m for m in available):
        print(f"[+] SUCCESS: Model '{MODEL_NAME}' is available locally.")
    else:
        print(f"[-] WARNING: Model '{MODEL_NAME}' not found in {available}.")
        print(f"    Action: Run 'docker exec -it ollama_server ollama pull {MODEL_NAME}'")

    # 3. Test RLM-style "Code Only" Generation
    print(f"[*] Testing RLM Code Generation (Inference-Time Strategy)...")
    test_prompt = "Write a one-line Python script to set result = 40 + 2. Output ONLY code."
    
    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': test_prompt}]
        )
        content = response['message']['content'].strip()
        print(f"[+] Raw Model Output: {content}")
        
        # Validation for RLM paradigm
        if "result" in content and "42" not in content: # Checking if it wrote logic vs just the answer
             print("[+] SUCCESS: Model produced executable logic.")
        else:
             print("[!] NOTE: Model might need stricter system prompting for RLM tasks.")

    except Exception as e:
        print(f"[!] ERROR: Inference failed: {e}")

if __name__ == "__main__":
    run_diagnostics()
