import ollama
import os
import time
import sys
import re

import requests
from ollama import Client

api_key = os.getenv("OLLAMA_API_KEY")
endpoint = os.getenv("OLLAMA_ENDPOINT")


client = Client(
    host=endpoint,
    headers={'Authorization': 'Bearer ' + api_key}
)

#sin memoria de contexto
while True:
  prompt = input("Enter your prompt: ")
  messages = [
    {
      'role': 'user',
      'content': prompt,
    },
  ]  
  ## deepseek-v3.1:671b-cloud, gpt-oss:20b-cloud, gpt-oss:120b-cloud, kimi-k2:1t-cloud, qwen3-coder:480b-cloud, kimi-k2-thinking
  for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
    print(part.message.content, end='', flush=True)
    
  
#Con memoria de contexto
messages = []

while True:
  prompt = input("Enter your prompt: " )
  messages.append({
      "role": "user",
      "content": prompt
  }) 
  response_text = ""
  for part in client.chat(
      model='gpt-oss:20b',
      messages=messages,
      stream=True
  ):
      print(part.message.content, end='', flush=True)
      response_text += part.message.content
  print()
  messages.append({
      "role": "assistant",
      "content": response_text
  })

