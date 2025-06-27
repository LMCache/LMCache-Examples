import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import List, Dict
from transformers import AutoTokenizer
import requests

from huggingface_hub import HfFolder, login

# Set up the Hugging Face Hub credentials
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token and HfFolder.get_token() != hf_token:
    login(token=hf_token)

# Configuration for backend services and model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # Updated model
PORT_LMCACHE = 8000 # vLLM with LMCache
PORT_DEFAULT = 8001 # vLLM without LMCache

# Initialize session state for ordered selected chunks
if 'ordered_selected_chunks' not in st.session_state:
    st.session_state.ordered_selected_chunks = []

@st.cache_resource
def get_tokenizer():
    print(f"Loading tokenizer for model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


tokenizer = get_tokenizer()


@st.cache_data
def read_chunks(file_folder) -> Dict[str, str]:
    """
    Read all the txt files in the folder and return the filenames
    """
    filenames = os.listdir(file_folder)
    ret = {}
    for filename in filenames:
        if not filename.endswith("txt"):
            continue
        if filename == "sys_prompt.txt":
            continue
        key = filename.removesuffix(".txt")
        with open(os.path.join(file_folder, filename), "r") as fin:
            value = fin.read()
        ret[key] = value

    return ret

chunks = read_chunks("data/")

# Load system prompt
try:
    with open("data/sys_prompt.txt", "r") as f:
        sys_prompt = f.read()
except FileNotFoundError:
    sys_prompt = "You are a helpful assistant.\n\n" # Default system prompt
    st.warning("data/sys_prompt.txt not found, using default system prompt.")

# --- UI Layout ---
st.title("RAG + KV Cache Blending")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [] # Each item: {"user": "...", "lmcache": "...", "default": "..."}

# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    sys_container = st.container(border=True)
    sys_container.markdown("#### System Prompt")
    sys_container.markdown(f"> {sys_prompt}")

    temperature = st.slider("Temperature:", 0.0, 1.0, 0.1, 0.05) # Default 0.1 as in many demos

    st.markdown("#### Context Selection")
    # Let user select chunks using multiselect
    # The order of selection in multiselect can be an initial order.
    initially_selected_chunks = st.multiselect(
        "Select and initially order chunks for the context",
        list(chunks.keys()),
        default=st.session_state.ordered_selected_chunks, # Use current ordered list as default
        key="chunk_multiselect"
    )

    # Update session state based on multiselect changes (add/remove chunks)
    # This logic aims to preserve existing order as much as possible when items are added/removed
    # via the multiselect, and let the buttons handle explicit reordering.

    # Convert to sets for easier comparison of content
    current_ordered_set = set(st.session_state.ordered_selected_chunks)
    multiselect_set = set(initially_selected_chunks)

    if current_ordered_set != multiselect_set:
        new_ordered_list = []
        # Add items from old ordered list if they are still in multiselect, preserving order
        for item in st.session_state.ordered_selected_chunks:
            if item in multiselect_set:
                new_ordered_list.append(item)

        # Add any new items from multiselect that weren't in the old ordered list
        # These are typically appended to the end, respecting multiselect order for new items.
        for item in initially_selected_chunks: # Iterate through multiselect to respect its order for new items
            if item not in current_ordered_set: # Item is new
                 if item not in new_ordered_list: # Ensure not already added (if it was in multiselect twice somehow)
                    new_ordered_list.append(item)

        st.session_state.ordered_selected_chunks = new_ordered_list
        st.rerun()

    st.markdown("#### Reorder Selected Chunks")
    if not st.session_state.ordered_selected_chunks:
        st.caption("No chunks selected yet.")
    else:
        for i, chunk_key in enumerate(st.session_state.ordered_selected_chunks):
            cols = st.columns([0.7, 0.15, 0.15])
            cols[0].write(chunk_key)
            if cols[1].button("⬆️", key=f"up_{chunk_key}", use_container_width=True, help="Move chunk up"):
                if i > 0:
                    st.session_state.ordered_selected_chunks.insert(i - 1, st.session_state.ordered_selected_chunks.pop(i))
                    st.rerun()
            if cols[2].button("⬇️", key=f"down_{chunk_key}", use_container_width=True, help="Move chunk down"):
                if i < len(st.session_state.ordered_selected_chunks) - 1:
                    st.session_state.ordered_selected_chunks.insert(i + 1, st.session_state.ordered_selected_chunks.pop(i))
                    st.rerun()

    # --- New: Option to run only one model at a time ---
    st.markdown("#### Model Execution Mode")
    run_one_model = st.checkbox("Run only one model at a time", value=False, key="run_one_model")
    model_choice = None
    if run_one_model:
        model_choice = st.radio("Which model to run?", ["LMCacheBlend", "vLLM without LMCache"], key="model_choice")

# Context display (above the chat columns)
context_display_container = st.container(border=True)

# --- Initialize Chat Sessions ---
# These will be re-initialized if temperature or context changes, which is acceptable for this demo.
session_lmcache = chat_session.ChatSession(PORT_LMCACHE)
session_lmcache.temperature = temperature
# session_lmcache.separator = "" # LMCache enabled instance uses separator

session_default = chat_session.ChatSession(PORT_DEFAULT)
session_default.temperature = temperature
# session_default.separator = "" # Default vLLM instance, no special separator needed in its own logic

# Construct and set context for both sessions using the ordered list
# Use st.session_state.ordered_selected_chunks instead of the global 'selected_chunks'
current_context_parts = [sys_prompt] + [chunks[key] for key in st.session_state.ordered_selected_chunks]
# The set_context method in chat_session.py builds the final string with separators
session_lmcache.set_context(current_context_parts)
session_default.set_context(current_context_parts) # Uses its own separator logic internally

# Display the constructed context (same for both, show LMCache version)
context_for_display = session_lmcache.get_context() # This will include separators
if context_for_display: # Only encode if context is not empty
    num_tokens = tokenizer.encode(context_for_display) # Tokenize the version with separators
    context_display_container.header(f"The Context Given to LLMs ({len(num_tokens)} tokens):", divider="grey")
    context_display_container.text(context_for_display)
else:
    context_display_container.header("The Context Given to LLMs (0 tokens):", divider="grey")
    context_display_container.caption("Context is empty. Select some chunks.")


# --- Dual Chat Area ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ LMCacheBlend", divider="rainbow")
    lmcache_chat_container = st.container(height=500, border=True)
    with lmcache_chat_container:
        for message_pair in st.session_state.messages:
            if "user" in message_pair: # Display user message
                st.chat_message("user").write(message_pair["user"])
            if "lmcache" in message_pair and message_pair["lmcache"]: # Display LMCache response
                st.chat_message("assistant", avatar="🚀").write(message_pair["lmcache"])

with col2:
    st.subheader("❌ LMCacheBlend", divider="rainbow")
    default_chat_container = st.container(height=500, border=True)
    with default_chat_container:
        for message_pair in st.session_state.messages:
            if "user" in message_pair: # Display user message (again, or structure history better)
                                       # For now, this simple loop is fine.
                st.chat_message("user").write(message_pair["user"])
            if "default" in message_pair and message_pair["default"]: # Display Default response
                st.chat_message("assistant", avatar="🐢").write(message_pair["default"])

# Unified chat input
if prompt := st.chat_input("Ask a question to both LLMs..."):
    import queue
    import threading

    def _process_stream_to_queue(stream_generator, message_queue: queue.Queue, avatar: str):
        full_response = ""
        try:
            for chunk in stream_generator:
                if chunk: # Ensure there's content
                    full_response += chunk
                    message_queue.put({"type": "chunk", "content": chunk, "full_response": full_response, "avatar": avatar})
            message_queue.put({"type": "done", "full_response": full_response, "avatar": avatar})
        except Exception as e:
            print(f"Error in stream processing for {avatar}: {e}")
            message_queue.put({"type": "error", "content": str(e), "avatar": avatar})

    def process_user_input_dual(user_prompt: str):
        current_message_index = len(st.session_state.messages)
        st.session_state.messages.append({"user": user_prompt, "lmcache": "", "default": ""})

        with lmcache_chat_container:
            lmcache_placeholder = st.chat_message("assistant", avatar="🚀").empty()
        with default_chat_container:
            default_placeholder = st.chat_message("assistant", avatar="🐢").empty()

        lmcache_queue = queue.Queue()
        default_queue = queue.Queue()

        # --- New: Only run one model if selected ---
        if 'run_one_model' in st.session_state and st.session_state['run_one_model']:
            if st.session_state.get('model_choice', 'LMCacheBlend') == "LMCacheBlend":
                stream_lmcache = session_lmcache.chat(user_prompt)
                stream_default = None
            else:
                stream_lmcache = None
                stream_default = session_default.chat(user_prompt)
        else:
            stream_lmcache = session_lmcache.chat(user_prompt)
            stream_default = session_default.chat(user_prompt)

        thread_lmcache = None
        thread_default = None
        if stream_lmcache is not None:
            thread_lmcache = threading.Thread(target=_process_stream_to_queue, args=(stream_lmcache, lmcache_queue, "🚀"))
            thread_lmcache.daemon = True
            thread_lmcache.start()
        if stream_default is not None:
            thread_default = threading.Thread(target=_process_stream_to_queue, args=(stream_default, default_queue, "🐢"))
            thread_default.daemon = True
            thread_default.start()

        lmcache_done = stream_lmcache is None
        default_done = stream_default is None
        acc_lmcache_response = ""
        acc_default_response = ""

        while not (lmcache_done and default_done):
            if not lmcache_done and stream_lmcache is not None:
                try:
                    item = lmcache_queue.get(block=False)
                    if item["type"] == "chunk":
                        acc_lmcache_response = item["full_response"]
                        lmcache_placeholder.markdown(acc_lmcache_response + "▌")
                    elif item["type"] == "done":
                        acc_lmcache_response = item["full_response"]
                        lmcache_placeholder.markdown(acc_lmcache_response)
                        st.session_state.messages[current_message_index]["lmcache"] = acc_lmcache_response
                        lmcache_done = True
                    elif item["type"] == "error":
                        lmcache_placeholder.error(f"LMCache stream error: {item['content']}")
                        st.session_state.messages[current_message_index]["lmcache"] = f"Error: {item['content']}"
                        lmcache_done = True
                    lmcache_queue.task_done()
                except queue.Empty:
                    pass
            if not default_done and stream_default is not None:
                try:
                    item = default_queue.get(block=False)
                    if item["type"] == "chunk":
                        acc_default_response = item["full_response"]
                        default_placeholder.markdown(acc_default_response + "▌")
                    elif item["type"] == "done":
                        acc_default_response = item["full_response"]
                        default_placeholder.markdown(acc_default_response)
                        st.session_state.messages[current_message_index]["default"] = acc_default_response
                        default_done = True
                    elif item["type"] == "error":
                        default_placeholder.error(f"Default stream error: {item['content']}")
                        st.session_state.messages[current_message_index]["default"] = f"Error: {item['content']}"
                        default_done = True
                    default_queue.task_done()
                except queue.Empty:
                    pass
            time.sleep(0.05)
        st.rerun()

    process_user_input_dual(prompt)