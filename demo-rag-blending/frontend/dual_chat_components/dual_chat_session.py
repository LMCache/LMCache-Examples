import asyncio
import time
import chat_session
import conversation_generator
from typing import Dict, List, Tuple, Optional, Any, Generator
import threading
import os

class DualChatSession:
    """Manages chat sessions with two different endpoints simultaneously"""

    def __init__(self):
        # Initialize both chat sessions with the same model name
        self.ps_session = chat_session.ChatSession("ProductionStack", 30080)
        self.other_session = chat_session.ChatSession("Other", 30081)

        # Make sure both use the same model name for deterministic results
        if self.ps_session.model != self.other_session.model:
            print(f"Warning: Different models detected - PS: {self.ps_session.model}, Other: {self.other_session.model}")
            # Set both to use the same model (use ProductionStack's model)
            self.other_session.model = self.ps_session.model
            print(f"Forced both endpoints to use the same model: {self.ps_session.model}")

        # Initialize history
        self.history = []

        # Dictionary to store document names and contents
        self.docs = {}
        self.doc_contents = {}

    def initialize_with_conversation(self, num_rounds=5):
        """Initialize both sessions with the same conversation history"""
        # Generate a conversation and get doc names
        conversation, *doc_names = conversation_generator.generate_conversation(num_rounds)

        # Get list of available documents
        available_docs = conversation_generator.get_available_docs()

        # Store document names
        for i, doc_name in enumerate(doc_names):
            doc_key = f"doc{i+1}"
            self.docs[doc_key] = doc_name

        # Load document contents
        for doc_key in self.docs:
            self.load_doc_contents(doc_key)

        # Add a system message that we track in our history
        doc_list = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(doc_names)])

        system_msg = {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions about Unix command line tools and programming.
You have access to the following documentation:
{doc_list}

Always keep your responses concise, aiming for around 50 words or less.
Focus on the most essential information and be direct.
Please provide accurate and helpful responses based on this documentation."""
        }
        self.history.append(system_msg)

        # Set the context for both sessions (load the documentation)
        combined_docs = []
        for doc_key in self.docs:
            if self.doc_contents.get(doc_key):
                combined_docs.append(f"# {self.docs[doc_key]} Documentation\n{self.doc_contents[doc_key]}")

        # Set the context in both sessions
        print("Setting context with documentation in both sessions...")
        if combined_docs:
            self.ps_session.set_context(combined_docs)
            self.other_session.set_context(combined_docs)
            print(f"Loaded {len(combined_docs)} documentation files ({sum(len(doc) for doc in combined_docs)} bytes)")

        # First, make sure both sessions have a clean slate
        self.ps_session.messages = []
        self.other_session.messages = []

        # Start by adding system message
        self.ps_session.messages.append({"role": "system", "content": system_msg["content"]})
        self.other_session.messages.append({"role": "system", "content": system_msg["content"]})

        # Add conversation to our history and to both sessions
        print("Adding conversation history to both sessions...")
        for msg in conversation:
            if msg["role"] == "system":
                continue  # Skip system message as we've already added our own

            self.history.append(msg)

            if msg["role"] == "user":
                self.ps_session.messages.append({"role": "user", "content": msg["content"]})
                self.other_session.messages.append({"role": "user", "content": msg["content"]})
                print(f"Added user message: {msg['content'][:30]}...")
            elif msg["role"] == "assistant":
                self.ps_session.messages.append({"role": "assistant", "content": msg["content"]})
                self.other_session.messages.append({"role": "assistant", "content": msg["content"]})
                print(f"Added assistant message: {msg['content'][:30]}...")

        # Verify the conversation was loaded properly
        print(f"Initialized with {len(self.ps_session.messages)} messages in ProductionStack")
        print(f"Initialized with {len(self.other_session.messages)} messages in Other")

        return self.history, self.docs

    def send_message(self, message: str) -> Tuple[Generator, Generator, Dict, Dict]:
        """Send a message to both sessions and return the streaming generators"""
        # Add "in 50 words" to the message if it's not already there
        #if "in 50 words" not in message.lower() and "within 50 words" not in message.lower():
        #    if message.endswith("?"):
        #        # Insert before the question mark
        #        message = message[:-1] + " in 50 words?"
        #    else:
        #        # Add to the end
        #        message = message + " Please answer in 50 words."

        # Add message to history
        self.history.append({"role": "user", "content": message})

        # Create empty assistant responses in history first
        ps_entry = {
            "role": "assistant",
            "endpoint": "ProductionStack",
            "content": "",
            "metrics": {"ttft": None, "itl": None}
        }
        other_entry = {
            "role": "assistant",
            "endpoint": "Other",
            "content": "",
            "metrics": {"ttft": None, "itl": None}
        }

        # Add these entries to history right away
        self.history.append(ps_entry)
        self.history.append(other_entry)

        # Print debug info
        print(f"Added placeholder entries to history: PS={id(ps_entry)}, Other={id(other_entry)}")

        # Store metrics for comparison
        ps_metrics = {"ttft": None, "itl": None, "complete": False, "history_entry": ps_entry}
        other_metrics = {"ttft": None, "itl": None, "complete": False, "history_entry": other_entry}

        # Ensure message history is perfectly synchronized between both sessions
        # This fixes any potential history divergence issues
        print("Synchronizing message history between both endpoints...")
        # Reset both session messages to be identical
        self.ps_session.messages = [msg.copy() for msg in self.other_session.messages
                                   if msg["role"] != "assistant" or msg.get("endpoint") != "Other"]
        self.other_session.messages = [msg.copy() for msg in self.ps_session.messages]
        # Add the new user message to both
        self.ps_session.on_user_message(message, display=False)
        self.other_session.on_user_message(message, display=False)

        # Get the streaming generators from both sessions
        ps_stream = self._wrap_stream(self.ps_session.chat(message), "ProductionStack", ps_metrics)
        other_stream = self._wrap_stream(self.other_session.chat(message), "Other", other_metrics)

        return ps_stream, other_stream, ps_metrics, other_metrics

    def _wrap_stream(self, stream, endpoint_name, metrics_dict):
        """Wrap a streaming generator to extract metrics"""
        content_chunks = []
        history_entry = metrics_dict["history_entry"]

        # Print debug info
        print(f"Starting stream for {endpoint_name}, entry id={id(history_entry)}")

        for chunk in stream:
            # Check for metric markers
            if chunk.startswith("<metric:ttft:"):
                try:
                    ttft_value = chunk.split(":")[2].rstrip(">")
                    ttft = float(ttft_value)
                    metrics_dict["ttft"] = ttft

                    # Update the metrics in our history entry directly
                    history_entry["metrics"]["ttft"] = ttft

                    # Print debug information
                    print(f"Found TTFT metric for {endpoint_name}: {ttft} ms, updated entry id={id(history_entry)}")
                except Exception as e:
                    print(f"Error parsing TTFT metric: {e}, chunk: {chunk}")

                # Skip adding metric chunks to content
                continue
            elif chunk.startswith("<metric:itl:"):
                try:
                    itl_value = chunk.split(":")[2].rstrip(">")
                    itl = float(itl_value)
                    metrics_dict["itl"] = itl

                    # Update the metrics in our history entry directly
                    history_entry["metrics"]["itl"] = itl

                    # Print debug information
                    print(f"Found ITL metric for {endpoint_name}: {itl} ms, updated entry id={id(history_entry)}")
                except Exception as e:
                    print(f"Error parsing ITL metric: {e}, chunk: {chunk}")

                # Skip adding metric chunks to content
                continue
            elif chunk.startswith("<metric:"):
                # Skip other metric markers
                continue

            # Regular content
            content_chunks.append(chunk)
            yield chunk

        # After stream completes, update the complete response in history
        complete_content = "".join(content_chunks)
        history_entry["content"] = complete_content

        # Print debug
        print(f"Completed stream for {endpoint_name}, updated content for entry id={id(history_entry)}")

        # Mark this stream as complete
        metrics_dict["complete"] = True

    def get_metrics_comparison(self):
        """Get a comparison of metrics between the two endpoints"""
        result = {}

        # Find the latest metrics for each endpoint
        latest_ps_metrics = None
        latest_other_metrics = None

        # Find the latest metrics for each endpoint
        for msg in reversed(self.history):
            if msg.get("role") == "assistant":
                if msg.get("endpoint") == "ProductionStack" and not latest_ps_metrics and msg.get("metrics"):
                    latest_ps_metrics = msg["metrics"]
                elif msg.get("endpoint") == "Other" and not latest_other_metrics and msg.get("metrics"):
                    latest_other_metrics = msg["metrics"]

            if latest_ps_metrics and latest_other_metrics:
                break

        # Calculate percentage differences if we have both metrics
        if latest_ps_metrics and latest_other_metrics:
            ps_ttft = latest_ps_metrics.get("ttft")
            other_ttft = latest_other_metrics.get("ttft")
            ps_itl = latest_ps_metrics.get("itl")
            other_itl = latest_other_metrics.get("itl")

            # Calculate percentage differences
            if ps_ttft is not None and other_ttft is not None:
                # For TTFT, lower is better, so (Other/PS - 1) * 100
                ttft_diff_pct = ((other_ttft / ps_ttft) - 1) * 100 if ps_ttft > 0 else 0
                result["ttft_diff_pct"] = ttft_diff_pct
                result["ps_ttft"] = ps_ttft
                result["other_ttft"] = other_ttft

            if ps_itl is not None and other_itl is not None:
                # For ITL, lower is better, so (Other/PS - 1) * 100
                itl_diff_pct = ((other_itl / ps_itl) - 1) * 100 if ps_itl > 0 else 0
                result["itl_diff_pct"] = itl_diff_pct
                result["ps_itl"] = ps_itl
                result["other_itl"] = other_itl

        if not result:
            return None

        return result

    def get_doc_preview(self, doc_key, max_chars=1000):
        """Get a preview of a document's content"""
        if doc_key not in self.doc_contents:
            return "Document not found"

        content = self.doc_contents[doc_key]
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "..."

    def load_doc_contents(self, doc_key):
        """Load document contents from disk"""
        if doc_key not in self.docs:
            return False

        doc_name = self.docs[doc_key]
        doc_path = os.path.join("contexts", doc_name)

        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                self.doc_contents[doc_key] = f.read()
            print(f"Loaded document {doc_name} ({len(self.doc_contents[doc_key])} bytes)")
            return True
        except Exception as e:
            print(f"Error loading document {doc_name}: {e}")
            return False