import os
import httpx # Using httpx for async requests
import asyncio

# Configuration
VLLM_SERVER_URL_LMCACHE = "http://localhost:8000" # Assuming LMCache instance runs on port 8000
DATA_DIR = "data/" # Relative to project root, adjust if script is run from elsewhere
TIMEOUT_SECONDS = 120 # Timeout for requests to the LLM server

# Chunks to exclude from pre-caching
EXCLUDE_FILES = ["sys_prompt.txt"]

async def send_chunk_to_vllm(client: httpx.AsyncClient, chunk_name: str, chunk_content: str):
    """Sends a chunk to the vLLM server to populate the cache using OpenAI-compatible endpoint."""
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {"role": "user", "content": chunk_content}
        ],
        "temperature": 0.1,
        "max_tokens": 1
    }
    try:
        print(f"Pre-caching chunk: {chunk_name} ({len(chunk_content)} chars)...")
        response = await client.post(f"{VLLM_SERVER_URL_LMCACHE}/v1/chat/completions", json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status() # Raise an exception for bad status codes

        # Consume the stream to ensure the request is fully processed by the server
        async for _ in response.aiter_text():
            pass # We don't need to do anything with the response content

        print(f"Successfully sent chunk {chunk_name} for pre-caching.")
        return True
    except httpx.RequestError as e:
        print(f"Error sending chunk {chunk_name} to vLLM server: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while pre-caching {chunk_name}: {e}")
    return False

async def main():
    print(f"Starting KV cache pre-population for server at {VLLM_SERVER_URL_LMCACHE}...")
    print(f"Reading chunks from: {os.path.abspath(DATA_DIR)}")

    # Ensure data directory exists
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please ensure this script is run from the project root.")
        return

    chunks_to_cache = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt") and filename not in EXCLUDE_FILES:
            key = filename.removesuffix(".txt")
            filepath = os.path.join(DATA_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    chunks_to_cache[key] = f.read()
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")

    if not chunks_to_cache:
        print("No text chunks found to pre-cache.")
        return

    print(f"Found {len(chunks_to_cache)} chunks to pre-cache: {list(chunks_to_cache.keys())}")

    # The LMCache-enabled vLLM server must be running and accessible at VLLM_SERVER_URL_LMCACHE
    # The original run-server.sh started the dockers, then did pre-calculation.
    # This script assumes the server is already up.

    # Wait for the server to be ready (simple delay, could be a more robust check)
    # This is important if this script is run immediately after starting the server.
    wait_time = int(os.environ.get("PRECACHE_WAIT_SECONDS", "30")) # Allow configuration
    print(f"Waiting {wait_time} seconds for vLLM server to be ready...")
    await asyncio.sleep(wait_time)


    successful_sends = 0
    async with httpx.AsyncClient() as client:
        for chunk_name, chunk_content in chunks_to_cache.items():
            # According to LMCache blending, the chunks are identified by their content
            # separated by LMCACHE_BLEND_SPECIAL_STR.
            # To cache them "individually" such that LMCache can reuse them later,
            # we should send them as they would appear as segments in a blended prompt.
            # For example, if a blended prompt is "sys_prompt # # chunk1 # # query",
            # LMCache should identify "sys_prompt" and "chunk1".
            # Sending the raw chunk content should be sufficient if LMCache's chunking
            # and blending logic can match sub-sequences.

            # The original demo3-KV-blending/README.md says:
            # "The script will first load all the text chunks in data/ folder and calculate the KV cache for each chunk separately."
            # This implies sending each chunk as its own prompt.

            # Optional: If sys_prompt should always be part of the cached unit with a chunk:
            # sys_prompt_content = ""
            # try:
            #     with open(os.path.join(DATA_DIR, "sys_prompt.txt"), "r", encoding="utf-8") as f:
            #         sys_prompt_content = f.read()
            #     prompt_for_cache = f"{sys_prompt_content}{os.environ.get('LMCACHE_BLEND_SPECIAL_STR', ' # # ')}{chunk_content}"
            # except FileNotFoundError:
            #     print("Warning: sys_prompt.txt not found, caching chunk content directly.")
            #     prompt_for_cache = chunk_content
            # However, this might make caches too specific. Caching raw chunks seems more flexible for blending.

            if await send_chunk_to_vllm(client, chunk_name, chunk_content):
                successful_sends += 1
            # Optional: add a small delay between requests if needed
            # await asyncio.sleep(0.1)

    print(f"Pre-caching complete. Successfully sent {successful_sends}/{len(chunks_to_cache)} chunks.")

if __name__ == "__main__":
    # This script should be run after the LMCache-enabled vLLM server instance has started.
    # Example: python demo-kv-blending-new/backend/prepopulate_cache.py
    asyncio.run(main())
