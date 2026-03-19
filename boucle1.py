import ollama
import re
import io
import sys
import os
from contextlib import redirect_stdout
from datasets import load_dataset
from ollama import Client


# --- Configuration ---
api_key = os.getenv("OLLAMA_API_KEY")
endpoint = os.getenv("OLLAMA_ENDPOINT")

MODEL = "gpt-oss:120b"

client = Client(
    host=endpoint,
    headers={'Authorization': 'Bearer ' + api_key}
)

# --- Load Oolong-real Haystack ---
print("Loading Oolongbench data...")
# We load the 'dnd' subset which contains Critical Role transcripts
dataset = load_dataset("oolongbench/oolong-real", "dnd", split="validation", streaming=True)
example = next(iter(dataset))

# Print keys to console for debugging (helpful for your logs)
print(f"Available keys in dataset: {list(example.keys())}")

# The 'dnd' subset uses 'document' for the transcript
context_data = example.get('context_window_text', "")
#target_query = example.get('question', "No question found.")
target_query = "Find the very first time Liam (Vax/Caleb) makes a roll. What was the number rolled and what was it for?"

print(f"Dataset loaded. Question: {target_query[:50]}...")

def llm_query(query_text):
    response_text = ""

    for part in client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": query_text}],
        stream=True
    ):
        response_text += part.message.content

    return response_text



if __name__ == "__main__":
    # Dividir el contexto en trozos más pequeños
    chunks = [context_data[i:i+20000] for i in range(0, len(context_data), 20000)]
    answers = []

    for chunk in chunks:
        prompt = f"\n\nCONTEXT:\n{chunk}\n\nQuestion: {target_query}"
        response = llm_query(prompt)
        answers.append(response)
    
    # Combinar las respuestas de todos los chunks en una respuesta final
    final_prompt = f"Combine the following answers into one final answer:\n\n{''.join(answers)}"
    final_answer = llm_query(final_prompt)
    print("FINAL:", final_answer)