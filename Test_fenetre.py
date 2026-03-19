from ollama import chat
import time

# ── Configuration ────────────────────────────────────────────────────────────
MODEL          = 'gemma3'
NEEDLE         = "The chicken is named Carlos"
FILLER         = "This is a very long string. "   # 28 chars per repeat
QUESTION       = (
    "Question:\n What is the name of the chicken? "
    "This information is hidden in the context data given before the question. "
    "Answer ONLY based on the context, not on general knowledge."
)
EXPECTED_KW    = "carlos"   # lowercase keyword to check in the response

# Binary-search bounds (in number of filler repetitions on EACH side of the needle)
MIN_REPS = 0
MAX_REPS = 4000   # 4000 * 28 * 2 ≈ 224 000 chars max — adjust if needed

# ── Helpers ──────────────────────────────────────────────────────────────────

def build_prompt(reps: int) -> tuple[str, int]:
    """Build the full prompt and return (prompt_text, total_char_count)."""
    filler_block = FILLER * reps
    context = filler_block + NEEDLE + filler_block
    prompt = f"Context:\n{context}\n\n{QUESTION}"
    return prompt, len(context)


def ask_model(prompt: str) -> tuple[str, float]:
    """Send prompt to the model and return (response_text, elapsed_seconds)."""
    messages = [{'role': 'user', 'content': prompt}]
    t0 = time.perf_counter()
    response = chat(MODEL, messages=messages, stream=True)
    answer = ""
    for chunk in response:
        answer += chunk.message.content
    elapsed = time.perf_counter() - t0
    return answer.strip(), elapsed


def model_found_answer(response: str) -> bool:
    return EXPECTED_KW in response.lower()


# ── Binary search ────────────────────────────────────────────────────────────

def run_binary_search():
    print(f"Model  : {MODEL}")
    print(f"Filler : '{FILLER}' ({len(FILLER)} chars/repeat)")
    print(f"Needle : '{NEEDLE}'")
    print(f"Search range: {MIN_REPS}–{MAX_REPS} reps per side")
    print("=" * 60)

    lo, hi = MIN_REPS, MAX_REPS
    last_success = None   # largest reps for which the model succeeded

    while lo <= hi:
        mid = (lo + hi) // 2
        prompt, ctx_len = build_prompt(mid)
        total_chars = len(prompt)
        approx_tokens = total_chars // 4   # rough estimate

        print(f"\n[reps={mid:>5}] ctx={ctx_len:>8} chars (~{approx_tokens:>6} tokens) | querying...", end=" ", flush=True)

        answer, elapsed = ask_model(prompt)
        found = model_found_answer(answer)

        status = "✅ FOUND" if found else "❌ MISSED"
        print(f"{status}  ({elapsed:.1f}s)")
        print(f"           Response: {answer[:120]!r}")

        if found:
            last_success = mid
            lo = mid + 1   # try larger
        else:
            hi = mid - 1   # try smaller

    print("\n" + "=" * 60)
    if last_success is not None:
        _, ctx_len = build_prompt(last_success)
        print(f"✅ Max context where model SUCCEEDS : reps={last_success}, ctx≈{ctx_len:,} chars (~{ctx_len//4:,} tokens)")
    else:
        print("❌ Model failed even at the smallest context size tested.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_binary_search()