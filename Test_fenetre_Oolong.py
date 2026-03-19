from ollama import chat
from datasets import load_dataset
import time
import sys

print("Imported libraries successfully.")

print("Loading Oolongbench data...")
# We load the 'dnd' subset which contains Critical Role transcripts
dataset = load_dataset("oolongbench/oolong-real", "dnd", split="validation", streaming=True)

# Get the second example from the dataset
it = iter(dataset)
example0 = next(it) # Get the first example (we will skip this one)
example1 = next(it)
example2 = next(it) # Get the second example (skip the first one)

context_data = example1.get('context_window_text', "")

# Keep an immutable original copy for experiments
original_context = context_data[:60000]  # base document used for tests (trimmed)

# Experiment configuration
MODE = 'context_length'  # 'insertion_index' or 'context_length'
MAX_ATTEMPTS = 30

# Needle and query
NEEDLE = "Marcelo has 5 infinity stones. \n"
Expected_answer = "5"
target_query = (
    "Question:\n How many infinity stones does Marcelo have? "
    "This information is hidden in the context data i gave you before the question. "
    "You have to find it and answer the question based on that information. Do not answer based on the general knowledge."
)

# Helper to build prompt by inserting the needle at a given index
def build_prompt(insert_index: int) -> tuple[str, int]:
    # clamp index
    insert_index = max(0, min(insert_index, len(original_context)))
    test_context = original_context[:insert_index] + NEEDLE + original_context[insert_index:]
    prompt = f"Context:\n{test_context}\n\n{target_query}"
    return prompt, len(test_context)

# Helper to query the model and assemble streamed response
def query_model(prompt: str, model_name: str = 'gemma3', timeout: float = 60.0) -> tuple[str, float]:
    messages = [{'role': 'user', 'content': prompt}]
    t0 = time.perf_counter()
    try:
        response = chat(model_name, messages=messages, stream=True)
    except Exception as e:
        return f"[ERROR contacting model: {e}]", 0.0

    answer = ""
    try:
        for chunk in response:
            # chunk.message.content may be incremental text
            part = chunk.message.content
            answer += part
            # optional: stream to stdout
            print(part, end='', flush=True)
    except Exception as e:
        print(f"\n[!] Streaming error: {e}")

    elapsed = time.perf_counter() - t0
    print()  # newline after streaming
    return answer.strip(), elapsed


print("Prepared base context (trimmed to 18k chars). Starting a single test run to verify pipeline...")
# quick smoke test
prompt, ctx_len = build_prompt(len(original_context)//2)
print(f"Smoke test: inserting needle at index {len(original_context)//2} (ctx ≈ {ctx_len} chars)")
resp, elapsed = query_model(prompt)
print(f"Smoke test done in {elapsed:.2f}s. Response starts: {resp[:200]!r}\n")


# Experiment mode selector:
# 'insertion_index' -> binary search over insertion index (existing behavior)
# 'context_length'  -> binary search over total context length (new)


def run_insertion_index_search(max_attempts: int = 30):
    """Existing binary search over insertion index (keeps original semantics)."""
    lo = 0
    hi = len(original_context)
    last_success = None
    attempt = 0

    print("\nStarting binary search over insertion index (0..{}).".format(hi))

    while lo <= hi:
        mid = (lo + hi) // 2
        attempt += 1
        prompt, ctx_len = build_prompt(mid)
        approx_tokens = ctx_len // 4

        print(f"\n[Attempt {attempt}] insert_index={mid} | ctx={ctx_len} chars (~{approx_tokens} tokens) -> querying...")
        resp, elapsed = query_model(prompt)

        found = Expected_answer in resp
        print(f"Result: {'FOUND' if found else 'NOT FOUND'} (elapsed {elapsed:.1f}s)")

        if found:
            last_success = mid
            lo = mid + 1
        else:
            hi = mid - 1

        if attempt >= max_attempts:
            print("Reached max attempts, stopping.")
            break

    if last_success is not None:
        print(f"\n✅ Model SUCCESS up to insertion index {last_success} (ctx ≈ {last_success + len(NEEDLE):,} chars).")
    else:
        print("\n❌ Model failed to find the needle at any tested position.")


def run_context_length_search(max_attempts: int = 30):
    """Two-phase search over total context length as requested by the user.

    Phase A (halve-down): Start with the full document length. If the model does not
    find the needle, cut the context in half and try again, repeating until the
    needle is found or we reach length 0.

    Phase B (refine / binary search): Once we have a lower bound L_found (needle
    found) and an upper bound L_failed (the first larger length that failed),
    perform a binary search between them to find the maximal context length
    where the needle is still found.
    """
    full_len = len(original_context)
    print(f"\nStarting halve-down search from full length = {full_len} chars.")

    attempts = 0
    candidate = full_len
    prev_failed = None
    found_len = None

    # Phase A: repeatedly cut the document in half until the model finds the needle
    while candidate > 0 and attempts < max_attempts:
        attempts += 1
        truncated = original_context[:candidate]
        insert_at = len(truncated) // 2
        test_context = truncated[:insert_at] + NEEDLE + truncated[insert_at:]
        prompt = f"Context:\n{test_context}\n\n{target_query}"
        ctx_len = len(test_context)
        approx_tokens = ctx_len // 4

        print(f"\n[Phase A - Attempt {attempts}] trying length={ctx_len} chars (~{approx_tokens} tokens)...")
        resp, elapsed = query_model(prompt)
        found = Expected_answer in resp
        print(f"Result: {'FOUND' if found else 'NOT FOUND'} (elapsed {elapsed:.1f}s)")

        if found:
            found_len = ctx_len
            break
        else:
            prev_failed = candidate
            candidate = candidate // 2

    if found_len is None:
        print("\n❌ Needle not found even at the smallest halved context. Aborting.")
        return

    # If we found at the original full length and there is no prev_failed, we are done
    if prev_failed is None:
        print(f"\n✅ Found needle at full length {found_len} chars. No refinement needed.")
        return

    # Phase B: binary search between found_len (low) and prev_failed (high)
    lo = found_len
    hi = prev_failed
    attempts_phase_b = 0
    print(f"\nEntering Phase B: refine between {lo} (found) and {hi} (failed).")

    while lo + 1 < hi and (attempts + attempts_phase_b) < max_attempts:
        attempts_phase_b += 1
        mid_len = (lo + hi) // 2

        truncated = original_context[:mid_len]
        insert_at = len(truncated) // 2
        test_context = truncated[:insert_at] + NEEDLE + truncated[insert_at:]
        prompt = f"Context:\n{test_context}\n\n{target_query}"
        ctx_len = len(test_context)
        approx_tokens = ctx_len // 4

        print(f"\n[Phase B - Attempt {attempts_phase_b}] testing mid_len={ctx_len} chars (~{approx_tokens} tokens)...")
        resp, elapsed = query_model(prompt)
        found = Expected_answer in resp
        print(f"Result: {'FOUND' if found else 'NOT FOUND'} (elapsed {elapsed:.1f}s)")

        if found:
            lo = mid_len
        else:
            hi = mid_len

    print(f"\n✅ Final estimated maximum context length where model succeeds: {lo:,} chars (~{lo//4:,} tokens)")


if __name__ == '__main__':
    # Run the selected experiment
    if MODE == 'insertion_index':
        run_insertion_index_search(MAX_ATTEMPTS)
    elif MODE == 'context_length':
        run_context_length_search(MAX_ATTEMPTS)
    else:
        print(f"Unknown MODE: {MODE}. Choose 'insertion_index' or 'context_length'.")