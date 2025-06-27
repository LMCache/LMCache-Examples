# KV Cache Blending Demo w/ Chat UI

## Overview

This project demonstrates the benefits of LMCache's Key-Value (KV) cache blending feature for Large Language Models (LLMs) using vLLM. It provides a side-by-side chat interface to compare the performance of a vLLM instance with CacheBlend enabled against a standard vLLM instance.

The UI allows users to dynamically construct long contexts by selecting and reordering text chunks.

## Features

- **Dynamic Context Building:** Select and reorder text chunks to form the input context for the LLMs.
- **Dual LLM Companions:** Interact with two vLLM instances:
    - One with LMCache + KV blending enabled.
    - One standard vLLM instance.
- **Side-by-Side Chat UI:** You can select which instance to run & view responses from both LLMs in parallel for easy comparison.
- Uses `mistralai/Mistral-7B-Instruct-v0.2` by default.

## Prerequisites

- **Docker:** Ensure Docker is installed and running. ([Docker Install Guide](https://docs.docker.com/engine/install/))
- **NVIDIA GPU & Drivers:** 2x NVIDIA GPUs with appropriate drivers is required for vLLM.
- **NVIDIA Container Toolkit:** Necessary for Docker to access GPUs. ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- **Python 3.11+:**
- **Bash environment**
- **Hugging Face Token:** `mistralai/Mistral-7B-Instruct-v0.2` (or any private model) requires authentication.

## Project Structure

```
demo-kv-blending-new/
├── backend/
├── data/                    # Text chunks for context building & sys_prompt.txt
├── frontend/
│   ├── app.py               # Streamlit frontend application
│   └── chat_session.py      # Helper for API communication
├── run-server.sh            # Script to build Docker image and start backend servers
└── README.md                # This file
```

## Setup and How to Run

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd demo-kv-blending-new
    ```

2.  **Configure `run-server.sh` :**
    Open `run-server.sh` and review the configuration section:
    - `LOCAL_HF_HOME`: Defaults to `$HOME/.cache/huggingface`. Change if your Hugging Face cache is elsewhere. This path will be mounted into the Docker containers.
    - `HF_TOKEN`: **Important:** Set it in your shell environment before running the script:
      ```bash
      export HF_TOKEN="your_hf_token_here"
      ```
      The script will check that `HF_TOKEN` is set and exit with an error if it is not.

3.  **Start Backend Servers:**
    This script will:
    - Build the Docker image for the backend (this may take time on the first run).
    - Start two Docker containers: one for vLLM with LMCache (port 8000) and one for standard vLLM (port 8001).
    ```bash
    bash run-server.sh
    ```
    You might need `sudo` for Docker commands depending on your system configuration (`sudo bash ./run-server.sh`).
    
    **To monitor logs for a container, use:**
    ```bash
    docker logs -f <container name>
    ```

5.  **Install Frontend Dependencies:**
    ```bash
    pip install -r frontend/requirements.txt
    ```

6.  **Run the Streamlit Frontend:**
    Once the backend servers are ready (especially after the cache pre-population step in `run-server.sh` completes):
    ```bash
    streamlit run frontend/app.py
    ```
    The application should open in your web browser.

## How to Use the UI

1.  **Configure Settings (Sidebar):**
    - **System Prompt:** The system prompt being used is displayed. You can change it by editing `data/sys_prompt.txt` and restarting the frontend (backend servers don't need a restart for sys prompt changes handled by frontend).
    - **Temperature:** Adjust the generation temperature for the LLMs.
    - **Context Selection:** Use the multiselect dropdown to choose which text chunks from the `data/` directory are included in the context. You can also reorder them. The selected chunks will be concatenated with the system prompt.

2.  **View Context:**
    - The fully constructed context (system prompt + selected chunks + separators for LMCache view) is displayed, along with its token count. This is the input that will be processed by the LLMs.

3.  **Chat:**
    - **Dual Chat Panes:** You'll see two chat areas:
        - **vLLM with LMCache** (left, 🚀 icon)
        - **vLLM without LMCache** (right, 🐢 icon)
    - **Ask a Question:** Type your question in the input box at the bottom ("Ask a question to both LLMs...") and press Enter.
    - **View Responses:** Responses from each LLM instance will stream into their respective panes. Observe differences in response time, especially TTFT (time-to-first-token).

4.  **Chat History:**
    - The conversation is stored and displayed.
