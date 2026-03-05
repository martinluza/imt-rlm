from ollama import chat
import re
import io
import contextlib

answer_LLM = False
output_str = ""

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
  {
    'role': 'assistant',
    'content': "The sky is blue because of the way the Earth's atmosphere scatters sunlight.",
  },
  {
    'role': 'user',
    'content': 'What is the weather in Tokyo?',
  },
  {
    'role': 'assistant',
    'content': """The weather in Tokyo is typically warm and humid during the summer months, with temperatures often exceeding 30°C (86°F). The city experiences a rainy season from June to September, with heavy rainfall and occasional typhoons. Winter is mild, with temperatures
    rarely dropping below freezing. The city is known for its high-tech and vibrant culture, with many popular tourist attractions such as the Tokyo Tower, Senso-ji Temple, and the bustling Shibuya district.""",
  },
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
  
  if "python" in full_assistant_response.lower():
    try:
      match = re.search(r"```python(.*?)```", full_assistant_response, re.DOTALL)
      if match:
          print("Found Python code block:")
          code_string = match.group(1)
      output_buffer = io.StringIO()

      with contextlib.redirect_stdout(output_buffer):
        exec(code_string)

      output_str = output_buffer.getvalue()
      answer_LLM = True

      print("Captured output:")
      print(repr(output_str)) 
    except Exception as e:
      print("An error occurred while executing the code:", e)