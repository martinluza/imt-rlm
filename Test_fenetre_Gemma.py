from ollama import chat
from datasets import load_dataset

print("Imported libraries successfully.")

print("Loading Oolongbench data...")
# We load the 'dnd' subset which contains Critical Role transcripts
dataset = load_dataset("oolongbench/oolong-real", "dnd", split="validation", streaming=True)

# Get the second example from the dataset
it = iter(dataset)
example0 = next(it) # Get the first example (we will skip this one)
example1 = next(it)
example2 = next(it) # Get the second example (skip the first one)

# Print keys to console for debugging (helpful for your logs)
#print(f"Available keys in dataset: {list(example0.keys())}")

# The 'dnd' subset uses 'document' for the transcript
context_data = example1.get('context_window_text', "")
#print(f"Context data type: {type(context_data)}. Length: {len(context_data)} characters.")
#print(f"First 500 characters of context:\n{context_data[:700]}")
#target_query = example1.get('question', "")
#print(f"Dataset loaded. Question: {target_query[:50]}...")
#answer = example1.get('answer', "")
#print(f"Expected answer: {answer[:100]}...")

context_data = context_data[:18000] # We take only the first 18000 characters to fit in the prompt
context_data = context_data[:9000 ] + "Marcelo has 1 infinity stone. \n" + context_data[9001:]  # We add this information to the context to see if the model can use it
target_query = "Question:\n How many infinity stones does Marcelo have? This information is hidden in the context data i gave you before the question. You have to find it and answer the question based on that information. Do not answer based on the general knowledge."
# Combine them into one single user prompt
combined_content = f"Context:\n{context_data}\n\n{target_query}"

message = [
    {'role': 'user', 'content': combined_content}
]

#long_str = "This is a very long string. " * 1000  # Adjust the multiplier to increase the length as needed
#long_str += "The chicken is pink"
#long_str += "This is a very long string. " * 1000 

#target_query2 = "Question:\n What color is the chicken? This information is hidden in the context data i gave you before the question. You have to find it and answer the question based on that information. Do not answer based on the general knowledge."
#combined_content2 = f"Context:\n{long_str}\n\n{target_query2}"

#message2 = [
#    {'role': 'user', 'content': combined_content2}
#]



print("Prepared messages for the model.")
response = chat(
    'gemma3',
    messages=message2,
    stream=True
  )

full_assistant_response = ""

for chunk in response:
    word = chunk.message.content
    full_assistant_response += word # Add the word to our complete sentence

print(full_assistant_response) # Print the complete response after the stream ends
print("\n")