---

# Recursive Language Model (RLM) Implementation

A Python-based implementation of the **Recursive Language Model (RLM)** paradigm as proposed by Alex L. Zhang. This project enables Large Language Models (LLMs) like **Qwen3** and **SmolLM** to process arbitrarily long documents by treating them as external variables within a live Python REPL environment.

##  Key Features

* **Infinite Context Scaling**: Processes documents far beyond built-in context windows (100k+ characters) by avoiding "context rot".
* **Recursive Delegation**: Allows the root model to act as a "Senior Manager," writing code to call sub-instances of itself on specific text chunks.
* **Dockerized Workflow**: Fully isolated execution of model-generated code within a Docker sidecar architecture.
* **Hybrid Inference**: Support for local execution via **Ollama** or high-performance inference via **Ollama Cloud**.
* **Parallel Execution**: Leverages `ThreadPoolExecutor` to process document chunks simultaneously, reducing latency.

---

## 🛠 Prerequisites

* [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).
* NVIDIA Container Toolkit (for local GPU acceleration).
* Python 3.11+.

---

## 📦 Installation & Setup

### 1. Start the Environment

Launch the Ollama server and the Python execution container:

```bash
docker compose up -d

```

### 2. Pull the Models

Download the models recommended in the **PRAMEL** project:

```bash
# Pull Qwen3 (Reasoning Engine)
docker exec -it ollama_server ollama pull qwen3:4b

# Pull SmolLM (Lightweight Recursive calls)
docker exec -it ollama_server ollama pull smollm:1.7b

```

---

## 🖥 Usage

### Running the RLM Logic

To run the main recursive analysis script:

```bash
docker exec -it rlm_executor python main.py

```

### Configuration

You can toggle between local and cloud models in `main.py`:

```python
CHOSEN_MODEL = 'qwen3:4b'  # Local
# CHOSEN_MODEL = 'qwen3-coder-cloud'  # Cloud
OLLAMA_HOST = 'http://ollama_server:11434'

```

---

## 🧪 Experiments

This repository includes "needle in a haystack" validation tests:

1. **Linear Search**: The model writes a loop to scan a 45k+ character document.
2. **Recursive Delegation**: The model uses the `ask_llm` tool to sub-query chunks.
3. **Parallel Scaling**: (Optional) Use `parallel_ask_llm` to process chunks in concurrent threads.

---

## 📚 References

* [1] Alex L. Zhang, *Recursive Language Models* (arXiv:2512.24601).
* [2] Project PRAMEL - TAF MCE, IMT Atlantique.
* [3] SmolLM3 Publication (HuggingFace).

---

[Recursive Language Models: The Future of Long Context](https://www.youtube.com/watch?v=QYchuz6nBR8)

Would you like me to help you add a **"Troubleshooting"** section for common Docker GPU errors?
