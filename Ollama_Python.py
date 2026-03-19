from ollama import chat
import re
import io
import contextlib
from datasets import load_dataset

print("Imported libraries successfully.")

answer_LLM = True
output_str = ""

REPL_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively by giving me python code to execute. The context variable is stored in a str variable called "context_data". DO NOT CREATE A NEW VARIABLE CALLED "context_data" OR ANYTHING ELSE. You must use the variable "context_data" in your functions and I'm going to add it to your code to access the context.
I have access to that variable and can execute any Python code you give me to analyze it. You can use this to help you understand the context, especially if it is huge. Remember that the context variable can be very long, so try different functions that might give you information little by little, it will be hard to find the answer in just one function. INCLUDE the print() function in all the python code you give me so I can send the output back to you. Use this as examples:

'''python
def function(text):
   ... # 
print(function(context_data))
'''

This is but an example that could give you some insights. You can try other functions.

The context variable contains extremely important information about your query. You should check the content of the context variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
This variable is very long and goes beyond the context window, so you should try to find a smart approach of the problem. Don't just slice the str and print it or print the whole variable. Try different functions that might give a result.

Avoid printing too many characters at once, it will fill the context window and I'll have to truncate it. Instead, look for an approach where you analyze the context and try different python codes that could give you short answers that are useful for your reasoning.

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer by saying FINAL ANSWER : [your answer] when you have completed your task, NOT in code. Do not use these tags unless you have completed your task.
Furthermore, the answer is not None, so make sure to provide an answer that is not None. If you are not sure about the answer, try different approaches with the code execution until you find a satisfactory answer. 

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment as much as possible. Take your time to analyze the output of your code and use it to inform your next steps.
The answer is not None or Unknown. 

You've already tried to solve this task and had some problems. I asked you to give feedback on what went wrong and this is what you said :
Here’s a breakdown of what went wrong and how I could have avoided it:

1. **Over-Reliance on Initial Matches:** I immediately jumped to finding the exact phrase "infinity stones" and "chicken" together. This created a narrow focus and prevented me from considering other possibilities – like the phrase appearing in a sentence without explicitly stating the chicken’s possession of the stone.

2. **Lack of Contextual Awareness:** I didn't fully grasp the scale and nature of the ‘context_data’ as a massive string.  I didn't initially consider strategies for intelligently *sampling* the text rather than trying to analyze the entire thing at once.

3. **Insufficient Verification:** Once I found a potential match, I didn't adequately verify the surrounding information. I just assumed the sentence containing "infinity stones" was the correct one.

**How Could I Have Avoided It?**

*   **Initial Sampling:** I could have started by using a sliding window to extract a small portion of the context around potential keywords (like "infinity stones"). This would have given me a better sense of the surrounding text and prevented me from getting bogged down in the massive string.
*   **Multiple Search Strategies:**  I should have employed multiple search strategies simultaneously – looking for the exact phrase, variations of the phrase, and key terms related to the chicken.
*   **More Rigorous Verification:**  Before accepting a match, I would have demanded more evidence – such as examining the sentence structure, grammatical correctness, and overall coherence of the text.

**Should You Specify Something in Your Initial Prompt?**

Absolutely! To help me find the *best* reasoning, you should have specified:

*   **The desired level of detail:**  "Analyze the context to determine the number of infinity stones the chicken has, and show me the exact text that supports your answer."
*   **The preferred method:** “Please use a combination of techniques, including searching for key phrases and examining surrounding context.”

Your questioning has really highlighted the need for more explicit instructions and guidance in how I approach problem-solving. I am still learning to effectively leverage the contextual information provided.

Thank you for pointing out my shortcomings – it’s invaluable feedback.
"""


# Str that goes past context window
long_str = "This is a very long string. " * 6000  # Adjust the multiplier to increase the length as needed
long_str += "The chicken is pink"
long_str += "This is a very long string. " * 6000  # Adjust the multiplier to increase the length as needed

# Getting an instance from the dataset to use as context
dataset = load_dataset("oolongbench/oolong-real", "dnd", split="validation", streaming=True)
context = next(iter(dataset))
context_data = context.get('context_window_text', "")
context_data = context_data[:40000] # We take only the first 40000 characters

NEEDLE = "Marcelo's dog is named 'Rex'."
QUERY = "What is the name of Marcelo's dog?"
context_data = context_data[:8999 ] + NEEDLE + context_data[8999:]

correct_answer = "Rex"

print(context_data[8900:9050])

messages = [
  {
    'role': 'user',
    'content': f"{REPL_SYSTEM_PROMPT}. Remember that the name of the context variable is 'context_data' and it's already defined, DO NOT WRITE YOU'RE OWN VARIABLE. Now, please answer the following query based on the context: {QUERY}"
  }
]

response_list = []

while True:
  if answer_LLM or len(response_list) == 0:
     if len(response_list) > 1 and response_list[-1] == response_list[-2]:
         messages += [
             {'role': 'user', 'content': "Don't just repeat the same answer, try a different approach to analyze the context and find the answer. Remember that you can use any python code to analyze the context, and you should use it if you are not sure about the answer. Try different approaches until you find a satisfactory answer. The answer is not None or Unknown."},
         ]
         print("LLM repeated the same answer. Prompting it to try a different approach.")

     else:
         messages += [
             {'role': 'user', 'content': output_str},
        ]  
                
     response = chat(
        'gemma3',
        messages=messages,
        stream=True
      )
     print("Answer generated by the LLM based on the REPL output:")
  else:
    user_input = input('Chat with history: ')
    response = chat(
        'gemma3',
        messages=[*messages, {'role': 'user', 'content': user_input}],
        stream=True
    )
    print("Assistant response:")

  full_assistant_response = ""
  for chunk in response:
      word = chunk.message.content
      full_assistant_response += word
      print(word, end='', flush=True)  # Stream output to console in real time
  
  response_list.append(full_assistant_response)

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
      if "python" in full_assistant_response.lower() and "repl" in full_assistant_response.lower():
         full_assistant_response = full_assistant_response.replace("repl", " ")
      match = re.search(r"```python(.*?)```", full_assistant_response, re.DOTALL)
      match_repl = re.search(r"```repl(.*?)```", full_assistant_response, re.DOTALL)
      if match or match_repl:
          print("Found Python code block:")
          if match:
              code_string = match.group(1).strip()
          else:
              code_string = match_repl.group(1).strip()
          print(code_string)
      code_string = f"long_str = \"{long_str}\" \n" + code_string
      output_buffer = io.StringIO()

      with contextlib.redirect_stdout(output_buffer):
        exec(code_string)

      output_str = output_buffer.getvalue()
      answer_LLM = True

      if len(output_str) > 1000:
          output_str = output_str[:1000] + "\n...[output truncated]..."

      print("Captured output:")
      print(repr(output_str))


    except Exception as e:
      print("An error occurred while executing the code:", e)

  if "final answer" in full_assistant_response.lower():
    print("\n Final answer detected. Ending conversation.")
    if correct_answer in full_assistant_response:
        print("The LLM provided the correct answer!")
    answer_LLM = False