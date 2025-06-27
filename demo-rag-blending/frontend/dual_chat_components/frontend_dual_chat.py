import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import queue
import threading

# Prevent Streamlit file watcher from monitoring torch.classes
# This fixes the "no running event loop" error during hot-reloading
import streamlit.watcher.path_watcher
original_watch_file = streamlit.watcher.path_watcher.watch_file

def patched_watch_file(path, callback):
    if "torch/_classes.py" in path or "torch\\_classes.py" in path:
        return None
    return original_watch_file(path, callback)

streamlit.watcher.path_watcher.watch_file = patched_watch_file

from dual_chat_session import DualChatSession
import altair as alt
# Removing transformers import to avoid compatibility issues
import conversation_generator

# Set page configuration
st.set_page_config(
    page_title="LLM Performance Comparison",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title
st.title("Production Stack vs Other LLM Performance Comparison")
st.subheader("Comparing Time to First Token (TTFT) and Latency to Insight (ITL) with real documentation")

# Note on how conversation history works
st.info("""
**Note on conversation management:**
This demo sends the same queries to both endpoints (Production Stack and Other) in parallel.
However, to ensure consistent context, only the responses from the Production Stack are added to the conversation history.
This prevents the conversation from branching with different responses from each model.
""")

# Initialize session state
if 'dual_session' not in st.session_state:
    with st.spinner("Loading documentation and initializing conversation history..."):
        st.session_state.dual_session = DualChatSession()
        st.session_state.history, st.session_state.docs = st.session_state.dual_session.initialize_with_conversation(5)
        st.session_state.ps_processing = False
        st.session_state.other_processing = False
        st.session_state.metrics_history = []
        st.session_state.show_docs = False
        st.session_state.ps_message_buffer = ""
        st.session_state.other_message_buffer = ""
        st.session_state.last_update_time = time.time()
        st.success("Initialization complete! Ready to start conversation.")
        # Force rerun to ensure UI is updated
        st.rerun()

# Sidebar with information
with st.sidebar:
    st.subheader("Documentation Loaded as Context")
    if st.session_state.docs:
        # Debug info about docs
        st.write(f"Loaded {len(st.session_state.docs)} documents:")

        # Display all document names
        for i, (doc_key, doc_name) in enumerate(st.session_state.docs.items()):
            # Simply display the full document name without any special formatting
            st.write(f"**Doc {i+1}**: {doc_name}")

        # Button to show/hide documentation previews
        if st.button("Show/Hide Documentation Previews"):
            st.session_state.show_docs = not st.session_state.show_docs

        if st.session_state.show_docs:
            # Create tabs for documentation
            doc_keys = list(st.session_state.docs.keys())
            doc_names = [f"Doc {i+1}" for i in range(len(doc_keys))]

            if doc_keys:  # Only create tabs if we have documents
                doc_tabs = st.tabs(doc_names)

                for i, (tab, doc_key) in enumerate(zip(doc_tabs, doc_keys)):
                    with tab:
                        doc_preview = st.session_state.dual_session.get_doc_preview(doc_key, 500)
                        st.text_area("Preview", doc_preview, height=200)

    # Add a reset button
    if st.button("Reset Conversation"):
        with st.spinner("Resetting conversation and reloading documentation..."):
            # Create new session objects
            st.session_state.dual_session = DualChatSession()
            st.session_state.history, st.session_state.docs = st.session_state.dual_session.initialize_with_conversation(5)
            st.session_state.ps_processing = False
            st.session_state.other_processing = False
            st.session_state.metrics_history = []
            st.success("Conversation reset successfully!")
            st.rerun()

# Create two columns for the chat interfaces
col1, col2 = st.columns(2)

with col1:
    st.header("Production Stack", divider="gray")
    ps_container = st.container(height=600, border=False)

with col2:
    st.header("Other", divider="gray")
    other_container = st.container(height=600, border=False)

# Display metrics
latest_ps_ttft = None
latest_other_ttft = None
latest_ps_itl = None
latest_other_itl = None

# Find the latest metrics for each endpoint
# Print all history items for debugging
print("All history items:")
for i, msg in enumerate(st.session_state.history):
    if msg.get("role") == "assistant":
        print(f"  {i}. {msg.get('endpoint', 'unknown')}: " +
              f"metrics={msg.get('metrics')}, " +
              f"content_len={len(msg.get('content', ''))}")

# Now search for metrics
for msg in reversed(st.session_state.history):
    if msg.get("role") == "assistant":
        if msg.get("endpoint") == "ProductionStack":
            metrics = msg.get("metrics", {})
            if metrics:
                if "ttft" in metrics and metrics["ttft"] is not None and latest_ps_ttft is None:
                    latest_ps_ttft = metrics["ttft"]
                    print(f"Found PS TTFT: {latest_ps_ttft}")
                if "itl" in metrics and metrics["itl"] is not None and latest_ps_itl is None:
                    latest_ps_itl = metrics["itl"]
                    print(f"Found PS ITL: {latest_ps_itl}")

        elif msg.get("endpoint") == "Other":
            metrics = msg.get("metrics", {})
            if metrics:
                if "ttft" in metrics and metrics["ttft"] is not None and latest_other_ttft is None:
                    latest_other_ttft = metrics["ttft"]
                    print(f"Found Other TTFT: {latest_other_ttft}")
                if "itl" in metrics and metrics["itl"] is not None and latest_other_itl is None:
                    latest_other_itl = metrics["itl"]
                    print(f"Found Other ITL: {latest_other_itl}")

    # Break when we have all metrics, or continue if some are still missing
    have_ps_metrics = latest_ps_ttft is not None and latest_ps_itl is not None
    have_other_metrics = latest_other_ttft is not None and latest_other_itl is not None

    # Only break if we have all metrics
    if have_ps_metrics and have_other_metrics:
        break

    # Continue to next message if we're still looking for metrics

# Metrics comparison area in a more compact form
metrics_container = st.container(border=True)
with metrics_container:
    # Use a 4-column layout for TTFT and ITL metrics
    cols = st.columns([2, 2, 2, 2])

    # Display metrics
    with cols[0]:
        st.markdown("**Production Stack TTFT**")
        if latest_ps_ttft is not None:
            st.markdown(f"<h4 style='margin:0'>{latest_ps_ttft:.2f} ms</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='margin:0'>N/A</h4>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("**Production Stack ITL**")
        if latest_ps_itl is not None:
            st.markdown(f"<h4 style='margin:0'>{latest_ps_itl:.2f} ms</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='margin:0'>N/A</h4>", unsafe_allow_html=True)

    with cols[2]:
        st.markdown("**Other TTFT**")
        if latest_other_ttft is not None:
            st.markdown(f"<h4 style='margin:0'>{latest_other_ttft:.2f} ms</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='margin:0'>N/A</h4>", unsafe_allow_html=True)


    with cols[3]:
        st.markdown("**Other ITL**")
        if latest_other_itl is not None:
            st.markdown(f"<h4 style='margin:0'>{latest_other_itl:.2f} ms</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='margin:0'>N/A</h4>", unsafe_allow_html=True)

    # Display comparison as a separate row
    st.markdown("---")
    comp_cols = st.columns(2)

    # TTFT comparison
    with comp_cols[0]:
        if latest_ps_ttft is not None and latest_other_ttft is not None:
            ttft_diff_pct = ((latest_other_ttft / latest_ps_ttft) - 1) * 100 if latest_ps_ttft > 0 else 0
            if ttft_diff_pct > 0:
                st.markdown(f"<b>TTFT Comparison:</b> Production Stack is <b>{ttft_diff_pct:.1f}% faster</b>", unsafe_allow_html=True)
            else:
                st.markdown(f"<b>TTFT Comparison:</b> Production Stack is <b>{abs(ttft_diff_pct):.1f}% slower</b>", unsafe_allow_html=True)
        else:
            st.markdown("<b>Waiting for TTFT metrics...</b>", unsafe_allow_html=True)

    # ITL comparison
    with comp_cols[1]:
        if latest_ps_itl is not None and latest_other_itl is not None:
            itl_diff_pct = ((latest_other_itl / latest_ps_itl) - 1) * 100 if latest_ps_itl > 0 else 0
            if itl_diff_pct > 0:
                st.markdown(f"<b>ITL Comparison:</b> Production Stack is <b>{itl_diff_pct:.1f}% faster</b>", unsafe_allow_html=True)
            else:
                st.markdown(f"<b>ITL Comparison:</b> Production Stack is <b>{abs(itl_diff_pct):.1f}% slower</b>", unsafe_allow_html=True)
        else:
            st.markdown("<b>Waiting for ITL metrics...</b>", unsafe_allow_html=True)

    # Display metrics comparison info if available
    metrics_comparison = st.session_state.dual_session.get_metrics_comparison()
    if metrics_comparison:
        pass  # Remove the redundant metrics display

# Display the chat history
for i, message in enumerate(st.session_state.history):
    if message["role"] == "system":
        # For system messages, display them as notes at the top of both columns
        with ps_container:
            st.info(message["content"])
        with other_container:
            st.info(message["content"])
    elif message["role"] == "user":
        with ps_container:
            st.chat_message("user").write(message["content"])
        with other_container:
            st.chat_message("user").write(message["content"])

    elif message["role"] == "assistant":
        # Check if the message is simulated (part of the initial conversation)
        if message.get("simulated"):
            with ps_container:
                st.chat_message("assistant").write(message["content"])
            with other_container:
                st.chat_message("assistant").write(message["content"])
        # Real responses from endpoints
        elif message.get("endpoint") == "ProductionStack":
            with ps_container:
                st.chat_message("assistant").write(message["content"])
        elif message.get("endpoint") == "Other":
            with other_container:
                st.chat_message("assistant").write(message["content"])

# Create an input field that spans both columns
prompt = st.chat_input("Type your question here")

def process_user_input(prompt):
    # Mark as processing
    st.session_state.ps_processing = True
    st.session_state.other_processing = True

    # Add user message to conversation UI immediately
    with ps_container:
        user_message = st.chat_message("user")
        user_message.write(prompt)
    with other_container:
        user_message = st.chat_message("user")
        user_message.write(prompt)

    # Force an immediate UI update to show the user message
    st.empty()

    # Reset message buffers
    st.session_state.ps_message_buffer = ""
    st.session_state.other_message_buffer = ""
    st.session_state.last_update_time = time.time()

    # Get the streaming generators from both sessions
    ps_stream, other_stream, ps_metrics, other_metrics = st.session_state.dual_session.send_message(prompt)

    # Create placeholder messages for both endpoints
    with ps_container:
        ps_placeholder = st.chat_message("assistant").empty()

    with other_container:
        other_placeholder = st.chat_message("assistant").empty()

    # Force another UI update to show empty assistant placeholders
    st.empty()

    # Initialize message strings
    ps_message = ""
    other_message = ""

    # Create message queues for each stream
    ps_queue = queue.Queue()
    other_queue = queue.Queue()

    # Update frequency - set to 1 for more frequent updates
    update_frequency = 1

    # Thread function to process stream and put chunks into queue
    def process_stream(stream_iter, message_queue, is_ps=True):
        accumulated_chunk = ""
        try:
            for chunk in stream_iter:
                if chunk.startswith("<metric:"):
                    # Log the metric chunk but don't add to the message
                    print(f"Skipping {'PS' if is_ps else 'Other'} metric chunk: {chunk}")
                    continue

                # Accumulate small chunks to reduce queue overhead
                accumulated_chunk += chunk

                # Send chunks in larger batches for efficiency
                if len(accumulated_chunk) >= 5:
                    message_queue.put(accumulated_chunk)
                    accumulated_chunk = ""

            # Send any remaining accumulated text
            if accumulated_chunk:
                message_queue.put(accumulated_chunk)

            # Signal completion
            message_queue.put(None)
        except Exception as e:
            print(f"Error in {'PS' if is_ps else 'Other'} stream: {e}")
            message_queue.put(None)

    # Start threads for processing each stream
    ps_thread = threading.Thread(target=process_stream, args=(ps_stream, ps_queue, True))
    other_thread = threading.Thread(target=process_stream, args=(other_stream, other_queue, False))

    ps_thread.daemon = True
    other_thread.daemon = True

    ps_thread.start()
    other_thread.start()

    # Variables to track UI updates
    ps_complete = False
    other_complete = False

    # Process outputs from both queues
    while not (ps_complete and other_complete):
        update_ui = False
        current_time = time.time()
        # Update at least every 100ms
        force_update = (current_time - st.session_state.last_update_time) >= 0.1

        # Check Production Stack queue
        if not ps_complete:
            try:
                chunk = ps_queue.get(block=False)
                if chunk is None:
                    ps_complete = True
                    ps_placeholder.markdown(ps_message)
                    st.session_state.ps_processing = False
                else:
                    ps_message += chunk
                    st.session_state.ps_message_buffer = ps_message
                    # Update if we have accumulated enough text or after time threshold
                    if len(chunk) > 10 or force_update:
                        ps_placeholder.markdown(ps_message + "▌")
                        update_ui = True
                ps_queue.task_done()
            except queue.Empty:
                pass

        # Check Other queue
        if not other_complete:
            try:
                chunk = other_queue.get(block=False)
                if chunk is None:
                    other_complete = True
                    other_placeholder.markdown(other_message)
                    st.session_state.other_processing = False
                else:
                    other_message += chunk
                    st.session_state.other_message_buffer = other_message
                    # Update if we have accumulated enough text or after time threshold
                    if len(chunk) > 10 or force_update:
                        other_placeholder.markdown(other_message + "▌")
                        update_ui = True
                other_queue.task_done()
            except queue.Empty:
                pass

        # If we updated the UI, reset the timer
        if update_ui or force_update:
            st.session_state.last_update_time = current_time
            # Force streamlit to update - use container to avoid flickering
            st.empty()

        # Smaller delay to increase responsiveness
        time.sleep(0.005)

    # Make sure final content is displayed
    if not ps_complete:
        ps_placeholder.markdown(ps_message)
    if not other_complete:
        other_placeholder.markdown(other_message)

    # Print debug info about current metrics state
    print("Current metrics after processing:")
    for msg in reversed(st.session_state.history):
        if msg.get("role") == "assistant":
            metrics_debug = msg.get('metrics', {})
            print(f"  - {msg.get('endpoint', 'unknown')}: " +
                 f"TTFT = {metrics_debug.get('ttft')}, " +
                 f"ITL = {metrics_debug.get('itl')}")

    # Force a rerun to update metrics
    st.rerun()

if prompt and not st.session_state.ps_processing and not st.session_state.other_processing:
    process_user_input(prompt)