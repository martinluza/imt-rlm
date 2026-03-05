import os
from ollama import Client

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

# deepseek-v3.1:671b-cloud, gpt-oss:20b-cloud, gpt-oss:120b-cloud, kimi-k2:1t-cloud, qwen3-coder:480b-cloud, kimi-k2-thinking

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part.message.content, end='', flush=True)
