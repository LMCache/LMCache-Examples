#!/usr/bin/env python3
import os
import time
import glob
import asyncio

import httpx
import json

# ───────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
OURS_API_KEY  = os.environ.get("OURS_API_KEY", "")            # can be dummy
OURS_BASE_URL = "http://localhost:8001/v1"                    # your vLLM chat endpoint
OURS_MODEL    = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_TOKENS    = 1

# System prompt & separator
BLEND_SPECIAL_STR = " # # "


# ───────────────────────────────────────────────────────────────────────────────
# 2. EXACTLY YOUR SNIPPET'S Streaming + TTFT LOGIC
# ───────────────────────────────────────────────────────────────────────────────
async def call_api(prompt: str):
    url = f"{OURS_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OURS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    start = time.time()
    first_token_time = None
    full_text = ""
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=data, timeout=60) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        line = line[len("data: ") :]
                    if line.strip() == "[DONE]":
                        break
                    chunk = json.loads(line)
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if first_token_time is None and text:
                        first_token_time = time.time()
                    full_text += text
        end = time.time()
        ttft = (first_token_time - start) if first_token_time else 0.0
        total = end - start
        return ttft, total, full_text.strip()
    except Exception as e:
        return 0.0, 0.0, f"❌ Error: {e}"


# ───────────────────────────────────────────────────────────────────────────────
# 3. BUILD YOUR TWO "BLEND" PROMPTS
# ───────────────────────────────────────────────────────────────────────────────
def build_prompts(sys_prompt, chunks):
    p1 = (
        sys_prompt
        + BLEND_SPECIAL_STR
        + chunks[0].strip()
        + BLEND_SPECIAL_STR
        + chunks[1].strip()
        + BLEND_SPECIAL_STR
        + chunks[2].strip()
        + BLEND_SPECIAL_STR
        + chunks[3].strip()
        + BLEND_SPECIAL_STR
        + chunks[4].strip()
        + BLEND_SPECIAL_STR
        + "summarize the docs in 10 words."
    )

    p2 = (
        sys_prompt
        + BLEND_SPECIAL_STR
        + chunks[1].strip()
        + BLEND_SPECIAL_STR
        + chunks[2].strip()
        + BLEND_SPECIAL_STR
        + chunks[3].strip()
        + BLEND_SPECIAL_STR
        + chunks[4].strip()
        + BLEND_SPECIAL_STR
        + chunks[0].strip()
        + BLEND_SPECIAL_STR
        + "summarize the docs in 10 words."
    )

    return p1, p2

async def main(txt_folder: str):
    sys_prompt_path = os.path.join(txt_folder, "sys_prompt.txt")
    if not os.path.exists(sys_prompt_path):
        print(f"❌ sys_prompt.txt not found in {txt_folder}")
        return
    with open(sys_prompt_path, encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    # Read all .txt files except sys_prompt.txt
    paths = sorted(
        p for p in glob.glob(os.path.join(txt_folder, "*.txt"))
        if os.path.basename(p) != "sys_prompt.txt"
    )
    chunks = [open(p, encoding="utf-8").read() for p in paths]
    
    p1, p2 = build_prompts(sys_prompt, chunks)
    
    # First call
    ttft, total, out = await call_api(p1)
    print(f"TTFT: {ttft:.3f}s, Total generation time: {total:.3f}s, Output: {out!r}")

    # Second call
    ttft, total, out = await call_api(p2)
    print(f"TTFT: {ttft:.3f}s, Total generation time: {total:.3f}s, Output: {out!r}")

if __name__ == "__main__":
    txt_folder = "data"
    asyncio.run(main(txt_folder))
