import ollama
import re
import io
import sys
from contextlib import redirect_stdout
from datasets import load_dataset

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

# System prompt for the REPL environment with explicit final answer checking
REPL_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""

# --- Configuration ---
OLLAMA_HOST = "http://ollama_server:11434"
MODEL = "gemma3:4b" 
client = ollama.Client(host=OLLAMA_HOST)

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
    """Sub-LLM function called by the REPL loop."""
    response = client.chat(model=MODEL, messages=[{'role': 'user', 'content': query_text}])
    return response['message']['content']

def run_rlm_system(user_prompt, context):
    messages = [{'role': 'user', 'content': user_prompt}]
    repl_globals = {
        "context": context,  # Assurez-vous que c'est bien la string brute
        "llm_query": llm_query,
        "print": print,
        "re": re,
        "type": type,  # Ajoutez ceci pour qu'il puisse faire print(type(context))
        "len": len
    }
    for iteration in range(10):
        response = client.chat(model=MODEL, messages=messages)
        content = response['message']['content']
        messages.append({'role': 'assistant', 'content': content})
        
        print(f"\n--- Iteration {iteration} ---\n{content}")

        if "FINAL(" in content or "FINAL_VAR(" in content:
            print("\n[SYSTEM] Termination keyword 'FINISHED' detected.")
            break

        code_match = re.search(r'```repl\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            output_capture = io.StringIO()
            try:
                with redirect_stdout(output_capture):
                    exec(code, repl_globals)
                result = output_capture.getvalue()
            except Exception as e:
                result = f"Error in REPL: {str(e)}"
            
            messages.append({'role': 'user', 'content': f"REPL Output:\n{result}"})
        else:
            messages.append({'role': 'user', 'content': "Please use a ```repl block or provide a FINAL() answer."})

# --- Prepare the Prompt ---
prompt_instructions = f"""
{REPL_SYSTEM_PROMPT}

---
USER QUERY: {target_query}
IMPORTANT: The variable `context` is a raw STRING (not a dict). 
Do NOT use context["content"]. Use it directly as a string.
Example: matches = re.findall(r'Roll', context)
---
"""

if __name__ == "__main__":
    run_rlm_system(prompt_instructions, context_data)