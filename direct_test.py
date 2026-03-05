import ollama
import random
from datasets import load_dataset

# --- Configuration ---
OLLAMA_HOST = "http://ollama_server:11434"
MODEL = "gemma3:4b" 
client = ollama.Client(host=OLLAMA_HOST)

# --- Chargement des données ---
print("Chargement de l'exemple Oolongbench pour test DIRECT...")
dataset = load_dataset("oolongbench/oolong-real", "dnd", split="validation")
#example = next(iter(dataset))
example = random.choice(dataset)

context_data = example.get('context_window_text', "")
target_query = example.get('question', "")

print(f"Dataset loaded. Question: {target_query}")
ground_truth = example.get('answer', "")

# --- Test Direct (Zero-Shot) ---
def run_direct_test():
    # On met tout le contexte dans le prompt d'un coup
    prompt = f"""CONTEXT:
{context_data}

QUESTION: {target_query}

Provide a concise answer based ONLY on the context provided.
"""
    
    print(f"\n--- Envoi de la requête directe ({len(context_data)} caractères) ---")
    try:
        response = client.generate(model=MODEL, prompt=prompt)
        answer = response['response']
        
        print("\nREPONSE DU LLM:")
        print(answer)
        print("\n--- VALIDATION ---")
        print(f"Réponse attendue (Dataset) : {ground_truth}")
        
            
    except Exception as e:
        print(f"Erreur : {e} (Probablement un dépassement de contexte)")

if __name__ == "__main__":
    run_direct_test()