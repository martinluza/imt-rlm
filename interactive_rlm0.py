import ollama
import re
import io
import random
import sys
from contextlib import redirect_stdout
from datasets import load_dataset

# --- Configuration ---
OLLAMA_HOST = "http://ollama_server:11434"
MODEL = "gemma3:4b" 
client = ollama.Client(host=OLLAMA_HOST)

print("Loading Oolongbench dataset into memory...")
dataset = load_dataset("oolongbench/oolong-real", "dnd", split="validation")

def run_interactive_depth_0():
    # 1. Select a random example
    example = random.choice(dataset)
    context_data = example.get('context_window_text', "")
    target_query = example.get('question', "No question found.")
    ground_truth = example.get('answer', "No answer found.")

    print("\n" + "="*60)
    print("🎯 NEW RANDOM EXAMPLE LOADED")
    print(f"Context size : {len(context_data)} characters")
    print(f"Query        : {target_query}")
    print(f"True Answer  : {ground_truth}")
    print("="*60 + "\n")

    # 2. Setup Depth 0 Environment (No sub-LLMs, just Python)
    repl_globals = {
        "context": context_data,
        "re": re,
        "len": len,
        "print": print,
        "type": type
    }

    # 3. System Prompt telling the LLM to use the variable
    prompt = f"""You are an AI assistant solving a task. The data you need is too large for your prompt.
Instead, it has been loaded into a Python variable named `context` (a raw string).

YOUR TASK: {target_query}

RULES:
1. You cannot see the `context` string directly. You MUST write Python code to analyze it.
2. Wrap your Python code in ```repl ... ``` blocks.
3. I will execute your code and give you the printed output.
4. Use standard Python string methods or the `re` module to find the answer.
5. Once you know the final answer, say FINISHED and provide it.
"""

    messages = [{'role': 'user', 'content': prompt}]

    print("Starting interactive RLM Depth 0 loop. The LLM will now attempt to write code.")
    print("You can interrupt or guide it after each step.\n")

    while True:
        try:
            # Call the LLM
            response = client.chat(model=MODEL, messages=messages)
            content = response['message']['content']
            messages.append({'role': 'assistant', 'content': content})
            
            print(f"\n[🤖 LLM]:\n{content}")
            
            # Check if it reached the end
            if "FINISHED" in content.upper():
                print("\n✅ LLM claims to be finished!")
                break

            # Look for REPL code
            code_match = re.search(r'```repl\n(.*?)\n```', content, re.DOTALL)
            
            if code_match:
                code = code_match.group(1)
                print(f"\n[⚙️ Executing REPL Code...]")
                output_capture = io.StringIO()
                try:
                    with redirect_stdout(output_capture):
                        exec(code, repl_globals)
                    result = output_capture.getvalue()
                except Exception as e:
                    result = f"Python Error: {str(e)}"
                
                print(f">> Output:\n{result.strip()}")
                
                # Auto-feed the result back into the chat history
                feed_message = f"Execution Output:\n{result}\nWhat is your next step?"
                
                # Give the user a chance to intervene or just press Enter to let the LLM continue
                user_input = input("\n[Press Enter to feed this output to the LLM, or type a message to guide it] (type 'quit' to exit): ")
                if user_input.lower() in ['quit', 'exit']:
                    break
                if user_input.strip():
                    feed_message = f"Execution Output:\n{result}\nUser Note: {user_input}"
                
                messages.append({'role': 'user', 'content': feed_message})
                
            else:
                # If no code was generated, prompt the user to guide the LLM
                user_input = input("\n[You] (type 'quit' to exit): ")
                if user_input.lower() in ['quit', 'exit']:
                    break
                messages.append({'role': 'user', 'content': user_input})

        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    run_interactive_depth_0()