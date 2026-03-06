from ollama import chat
import re
import io
import contextlib
from datasets import load_dataset

print("Imported libraries successfully.")

answer_LLM = False
output_str = ""

REPL_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively by giving me python code to execute. The context variable is stored in a str variable called "long_str". DO NOT CREATE A NEW VARIABLE CALLED "context" OR ANYTHING ELSE. You must use the variable "long_str" to access the context.
I have access to that variable and can execute any Python code you give me to analyze it. You can use this to help you understand the context, especially if it is huge. Remember that the context variable can be very long, so don't be afraid to analyze it piece by piece. You can use the `print()` function to view the output of your code and continue your reasoning.

The context variable contains extremely important information about your query. You should check the content of the context variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query. 
This variable is very long and goes beyond the context window, so you should chunk it and analyze it piece by piece. You can use the `print()` function to view the output of your code and continue your reasoning.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. Those Python functions can take the variable "long_str" as an input, and you can create new variables to store your outputs. For example, say we want to chunk the context variable and look at the first 10000 characters:
```repl
chunk = long_str[:10000]
print(chunk)
```

I'm going to execute the code you give me and return the output to you, so make sure to use print statements to see the output of your code and continue your reasoning. 

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer by saying FINAL ANSWER : [your answer] when you have completed your task, NOT in code. Do not use these tags unless you have completed your task.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment as much as possible. Remember to explicitly answer the original query in your final answer.
"""

# Str that goes past context window
long_str = "This is a very long string. " * 15000  # Adjust the multiplier to increase the length as needed
long_str += "The chicken is pink"
long_str += "This is a very long string. " * 15000  # Adjust the multiplier to increase the length as needed

messages = [
  {
    'role': 'user',
    'content': f"{REPL_SYSTEM_PROMPT}. Remember that the name of the context variable is 'long_str'. Now, please answer the following query based on the context: What color is the chicken?"
  }
]

while True:
  if answer_LLM:
     messages += [
       {'role': 'user', 'content': output_str},
     ]
     user_input = input('Chat with history: ')
     response = chat(
        'gemma3',
        messages=[*messages, {'role': 'user', 'content': user_input}],
        stream=True
      )
  else:
    user_input = input('Chat with history: ')
    response = chat(
        'gemma3',
        messages=[*messages, {'role': 'user', 'content': user_input}],
        stream=True
    )

  full_assistant_response = ""

  for chunk in response:
        word = chunk.message.content
        full_assistant_response += word # Add the word to our complete sentence

  print(full_assistant_response) # Print the complete response after the stream ends
  print("\n")

  # Add the response to the messages to maintain the history
  if answer_LLM:
      messages += [
          {'role': 'assistant', 'content': full_assistant_response},
      ]
      answer_LLM = False
      output_str = ""
  else:
    messages += [
        {'role': 'user', 'content': user_input},
        {'role': 'assistant', 'content': full_assistant_response},
    ]
  
  if "python" in full_assistant_response.lower() or "repl" in full_assistant_response.lower():
    print("Detected code block in assistant response. Attempting to execute...")
    try:
      match = re.search(r"```python(.*?)```", full_assistant_response, re.DOTALL)
      match_repl = re.search(r"```repl(.*?)```", full_assistant_response, re.DOTALL)
      if match or match_repl:
          print("Found Python code block:")
          code_string = match.group(1) if match else match_repl.group(1)
      code_string = f"long_str = {long_str} \n" + code_string
      output_buffer = io.StringIO()

      with contextlib.redirect_stdout(output_buffer):
        exec(code_string)

      output_str = output_buffer.getvalue()
      answer_LLM = True

      print("Captured output:")
      print(repr(output_str)) 
    except Exception as e:
      print("An error occurred while executing the code:", e)