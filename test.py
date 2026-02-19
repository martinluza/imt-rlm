import ollama
import os

# CONFIGURATION
MODEL = "qwen3:4b"
HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434') # Use localhost if running outside docker
client = ollama.Client(host=HOST)

def test_pipeline():
    print(f"--- RLM System Check ---")
    
    # Test 1: Connectivity
    try:
        client.list()
        print("[1/3] Connection: OK")
    except:
        print("[1/3] Connection: FAILED (Check if Docker is running)")
        return

    # Test 2: Basic Inference
    print(f"[2/3] Testing {MODEL} Inference...", end="", flush=True)
    resp = client.chat(model=MODEL, messages=[{'role': 'user', 'content': 'Say "RLM_READY"'}])
    if "RLM_READY" in resp['message']['content']:
        print(" OK")
    else:
        print(f" ERROR (Got: {resp['message']['content']})")

    # Test 3: RLM Tool-Use Logic
    print(f"[3/3] Testing Logic Scoping...")
    env = {"document": "SECRET: 123", "result": None}
    code = "result = document.split(':')[1].strip()"
    try:
        exec(code, {}, env)
        if env["result"] == "123":
            print("      REPL Scope: OK")
        else:
            print(f"      REPL Scope: FAILED (Got {env['result']})")
    except Exception as e:
        print(f"      REPL Scope: CRASHED ({e})")

if __name__ == "__main__":
    test_pipeline()
