import time
import os
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import Dict
from transformers import AutoTokenizer
from huggingface_hub import HfFolder, login

hf_token = os.environ.get("HF_TOKEN", "")
if hf_token and HfFolder.get_token() != hf_token:
    login(token=hf_token)

MODEL_NAME     = "mistralai/Mistral-7B-Instruct-v0.2"
PORT_LMCACHE   = 8000   # vLLM with LMCacheBlend
PORT_DEFAULT   = 8001   # vLLM without LMCache
BLEND_SPECIAL_STR = " # # "

if "ordered_selected_chunks" not in st.session_state:
    st.session_state.ordered_selected_chunks = []

@st.cache_resource
def get_tokenizer():
    print(f"Loading tokenizer for model: {MODEL_NAME}")
    return AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer = get_tokenizer()

@st.cache_data
def read_chunks(folder: str) -> Dict[str, str]:
    ret = {}
    for fn in os.listdir(folder):
        if not fn.endswith(".txt") or fn == "sys_prompt.txt":
            continue
        key = fn[:-4]  # drop .txt
        with open(os.path.join(folder, fn), "r") as f:
            ret[key] = f.read()
    return ret

chunks = read_chunks("data/")

try:
    with open("data/sys_prompt.txt", "r") as f:
        sys_prompt = f.read()
except FileNotFoundError:
    sys_prompt = "You are a helpful assistant.\n\n"
    st.warning("data/sys_prompt.txt not found - using default system prompt.")

st.title("RAG + KV Cache Blending")

if "messages" not in st.session_state:
    st.session_state.messages = []  

with st.sidebar:
    st.header("Configuration")
    sys_container = st.container(border=True)
    sys_container.markdown("#### System Prompt")
    sys_container.markdown(f"> {sys_prompt}")

    temperature = st.slider("Temperature:", 0.0, 1.0, 0.0, 0.05)

    st.markdown("#### Context Selection")
    initially_selected = st.multiselect(
        "Select and order context chunks",
        list(chunks.keys()),
        default=st.session_state.ordered_selected_chunks,
        key="chunk_multiselect",
    )

    old = st.session_state.ordered_selected_chunks
    if set(initially_selected) != set(old):
        new_order = [x for x in old if x in initially_selected]
        new_order += [x for x in initially_selected if x not in new_order]
        st.session_state.ordered_selected_chunks = new_order
        st.rerun()

    st.markdown("#### Reorder Selected Chunks")
    if not st.session_state.ordered_selected_chunks:
        st.caption("No chunks selected yet.")
    else:
        for i, ck in enumerate(st.session_state.ordered_selected_chunks):
            c1, c2, c3 = st.columns([0.7, 0.15, 0.15])
            c1.write(ck)
            if c2.button("⬆️", key=f"up_{ck}", use_container_width=True):
                if i > 0:
                    st.session_state.ordered_selected_chunks.insert(
                        i - 1, st.session_state.ordered_selected_chunks.pop(i)
                    )
                    st.rerun()
            if c3.button("⬇️", key=f"down_{ck}", use_container_width=True):
                if i < len(st.session_state.ordered_selected_chunks) - 1:
                    st.session_state.ordered_selected_chunks.insert(
                        i + 1, st.session_state.ordered_selected_chunks.pop(i)
                    )
                    st.rerun()

    st.markdown("#### Model Execution Mode")
    run_both    = st.checkbox("Both", value=True,  key="run_both")
    run_lmcache = st.checkbox("LMCacheBlend",          key="run_lmcache")
    run_default = st.checkbox("vLLM without LMCache",  key="run_default")

context_container = st.container(border=True)

# Build context string
context_parts = [sys_prompt] + [chunks[k] for k in st.session_state.ordered_selected_chunks]
session_lmcache = chat_session.ChatSession(PORT_LMCACHE, blend_special_str=BLEND_SPECIAL_STR)
session_default = chat_session.ChatSession(PORT_DEFAULT, blend_special_str=BLEND_SPECIAL_STR)
for s in (session_lmcache, session_default):
    s.temperature = temperature
    s.set_context(context_parts)

ctx_str = session_lmcache.get_context()
token_count = len(tokenizer.encode(ctx_str)) if ctx_str else 0
context_container.header(f"The Context Given to LLMs ({token_count} tokens):", divider="grey")
context_container.text(ctx_str if ctx_str else "Context is empty. Select some chunks.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ LMCacheBlend", divider="rainbow")
    lmcache_chat_box = st.container(height=500, border=True)
    with lmcache_chat_box:
        for m in st.session_state.messages:
            if "user" in m:
                st.chat_message("user").write(m["user"])
            if m.get("lmcache"):
                st.chat_message("assistant", avatar="🚀").write(m["lmcache"])

with col2:
    st.subheader("🐢 vLLM without LMCache", divider="rainbow")
    default_chat_box = st.container(height=500, border=True)
    with default_chat_box:
        for m in st.session_state.messages:
            if "user" in m:
                st.chat_message("user").write(m["user"])
            if m.get("default"):
                st.chat_message("assistant", avatar="🐢").write(m["default"])

if prompt := st.chat_input("Ask a question…"):
    import queue, threading

    def _stream_to_queue(generator, q: queue.Queue, avatar: str):
        full = ""
        try:
            for chunk in generator:
                if chunk:
                    full += chunk
                    q.put({"type": "chunk", "content": chunk, "full": full, "avatar": avatar})
            q.put({"type": "done", "full": full, "avatar": avatar})
        except Exception as e:
            q.put({"type": "error", "content": str(e), "avatar": avatar})

    def process_user_input(user_msg: str):
        idx = len(st.session_state.messages)
        st.session_state.messages.append({"user": user_msg, "lmcache": "", "default": ""})

        with lmcache_chat_box:
            lmcache_placeholder = st.chat_message("assistant", avatar="🚀").empty()
        with default_chat_box:
            default_placeholder = st.chat_message("assistant", avatar="🐢").empty()

        q_lmcache, q_default = queue.Queue(), queue.Queue()

        if run_both or (run_lmcache and run_default):
            gen_lmcache  = session_lmcache.chat(user_msg)
            gen_default  = session_default.chat(user_msg)
        elif run_lmcache:
            gen_lmcache, gen_default = session_lmcache.chat(user_msg), None
        elif run_default:
            gen_lmcache, gen_default = None, session_default.chat(user_msg)
        else:  # fallback to Both if none selected
            gen_lmcache  = session_lmcache.chat(user_msg)
            gen_default  = session_default.chat(user_msg)

        # Spawn threads
        t_lmcache = (threading.Thread(target=_stream_to_queue, args=(gen_lmcache, q_lmcache, "🚀"), daemon=True)
                     if gen_lmcache else None)
        t_default = (threading.Thread(target=_stream_to_queue, args=(gen_default, q_default, "🐢"), daemon=True)
                     if gen_default else None)
        if t_lmcache: t_lmcache.start()
        if t_default: t_default.start()

        done_lmcache = gen_lmcache is None
        done_default = gen_default is None
        acc_lmcache, acc_default = "", ""

        while not (done_lmcache and done_default):
            if not done_lmcache:
                try:
                    item = q_lmcache.get_nowait()
                    if item["type"] == "chunk":
                        acc_lmcache = item["full"]
                        lmcache_placeholder.markdown(acc_lmcache + "▌")
                    elif item["type"] == "done":
                        acc_lmcache = item["full"]
                        lmcache_placeholder.markdown(acc_lmcache)
                        st.session_state.messages[idx]["lmcache"] = acc_lmcache
                        done_lmcache = True
                    elif item["type"] == "error":
                        lmcache_placeholder.error(f"LMCache error: {item['content']}")
                        st.session_state.messages[idx]["lmcache"] = f"Error: {item['content']}"
                        done_lmcache = True
                    q_lmcache.task_done()
                except queue.Empty:
                    pass
            if not done_default:
                try:
                    item = q_default.get_nowait()
                    if item["type"] == "chunk":
                        acc_default = item["full"]
                        default_placeholder.markdown(acc_default + "▌")
                    elif item["type"] == "done":
                        acc_default = item["full"]
                        default_placeholder.markdown(acc_default)
                        st.session_state.messages[idx]["default"] = acc_default
                        done_default = True
                    elif item["type"] == "error":
                        default_placeholder.error(f"Default error: {item['content']}")
                        st.session_state.messages[idx]["default"] = f"Error: {item['content']}"
                        done_default = True
                    q_default.task_done()
                except queue.Empty:
                    pass
            time.sleep(0.05)

        st.rerun()

    process_user_input(prompt)
