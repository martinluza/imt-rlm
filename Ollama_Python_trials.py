from ollama import chat
import re
import io
import contextlib
from datasets import load_dataset


def run_ollama_needel_haystack():

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

    The context variable contains extremely important information about your query. You should check the content of the context variable to understand what you are working with. Make sure that you look through it sufficiently as you answer your query.
    This variable is very long and goes beyond the context window, so you should try to find a smart approach of the problem. Don't just slice the str and print it or print the whole variable. Try different functions that might give a result.

    Avoid printing too many characters at once, it will fill the context window and I'll have to truncate it. Instead, look for an approach where you analyze the context and try different python codes that could give you short answers that are useful for your reasoning.

    IMPORTANT: When you are done with the iterative process, you MUST provide a final answer by saying FINAL ANSWER : [your answer] when you have completed your task, NOT in code. Do not use these tags unless you have completed your task.
    Furthermore, the answer is not None, so make sure to provide an answer that is not None. If you are not sure about the answer, try different approaches with the code execution until you find a satisfactory answer. 

    Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment as much as possible. Take your time to analyze the output of your code and use it to inform your next steps.
    The answer is not None or Unknown. 

    You've already tried to solve this task and had some problems. I asked you to give feedback on what went wrong and this is what you said :
    ..."""

    # Getting an instance from the dataset to use as context
    dataset = load_dataset("oolongbench/oolong-real", "dnd", split="validation", streaming=True)
    context = next(iter(dataset))
    context_data = context.get('context_window_text', "")

    NEEDLE = "Marcelo's dog is named Rex. "
    QUERY = "What is the name of Marcelo's dog?"
    context_data = context_data[:8999] + NEEDLE + context_data[8999:]

    correct_answer = "Rex"

    messages = [
        {
            'role': 'user',
            'content': f"{REPL_SYSTEM_PROMPT}. Remember that the name of the context variable is 'context_data' and it's already defined, DO NOT WRITE YOU'RE OWN VARIABLE. Now, please answer the following query based on the context: {QUERY}"
        }
    ]

    response_list = []
    LLM_characters_count = 0
    Max_LLM_characters = 9000  # Set a maximum character count for LLM responses to avoid infinite loops

    while True:
        if answer_LLM or len(response_list) == 0:
            if len(response_list) > 1 and response_list[-1] == response_list[-2]:
                messages += [
                    {
                        'role': 'user',
                        'content': "Don't just repeat the same answer, try a different approach to analyze the context and find the answer. Remember that you can use any python code to analyze the context, and you should use it if you are not sure about the answer. Try different approaches until you find a satisfactory answer. The answer is not None or Unknown."
                    },
                ]

            else:
                messages += [
                    {'role': 'user', 'content': output_str},
                ]

            response = chat(
                'gemma3',
                messages=messages,
                stream=True
            )

        full_assistant_response = ""
        for chunk in response:
            word = chunk.message.content
            full_assistant_response += word
            LLM_characters_count += len(word)
        
        if LLM_characters_count > Max_LLM_characters:
            return (0,Max_LLM_characters)

        response_list.append(full_assistant_response)

        messages += [
            {'role': 'assistant', 'content': full_assistant_response},
        ]
        output_str = ""


        if "context_data =" in full_assistant_response:
            messages += [
                {'role': 'user', 'content': "You are trying to redefine the context variable, which is not allowed. Please use the existing 'context_data' variable to access the context. Do not create a new variable."},
            ]
            answer_LLM = True  


        elif "python" in full_assistant_response.lower() or "repl" in full_assistant_response.lower():
            try:
                if "python" in full_assistant_response.lower() and "repl" in full_assistant_response.lower():
                    full_assistant_response = full_assistant_response.replace("repl", " ")
                match = re.search(r"```python(.*?)```", full_assistant_response, re.DOTALL)
                match_repl = re.search(r"```repl(.*?)```", full_assistant_response, re.DOTALL)
                if match or match_repl:
                    if match:
                        code_string = match.group(1).strip()
                    else:
                        code_string = match_repl.group(1).strip()
                # Safely embed the context_data as a Python literal to avoid syntax errors
                safe_context = repr(context_data)
                code_string = f"context_data = {safe_context}\n" + code_string
                output_buffer = io.StringIO()

                with contextlib.redirect_stdout(output_buffer):
                    exec(code_string)

                output_str = output_buffer.getvalue()

                if len(output_str) > 1000:
                    output_str = output_str[:1000] + "\n...[output truncated]..."

            except Exception as e:
                print("An error occurred while executing the code:", e)
                return (0, LLM_characters_count)

        if "final answer" in full_assistant_response.lower():
            if correct_answer in full_assistant_response:
                print("The LLM provided the correct answer! Stopping the loop.")
                return (1, LLM_characters_count)

            else:
                messages += [
                    {'role': 'user', 'content': "That is not the correct answer."},
                ]


if __name__ == '__main__':
    number_trials = 20
    results = []
    for i in range(number_trials):
        print(f"\n--- Trial {i+1}/{number_trials} ---")
        result = run_ollama_needel_haystack()
        results.append(result)
    number_of_successes = sum(1 for r in results if r[0] == 1)
    avg_characters = sum(r[1] for r in results) / len(results)
    print(f"\nSummary: {number_of_successes}/{number_trials} successes, avg LLM characters {avg_characters:.0f}")
