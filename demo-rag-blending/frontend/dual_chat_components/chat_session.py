from openai import OpenAI
import threading
import sys
from io import StringIO
import time
import random
import os



class ChatSession:
    def __init__(self, name, port, context_separator="###"):
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"
        self.name = name
        self.port = port

        # Create custom HTTP headers with x-user-id: 9999
        custom_headers = {
            "x-user-id": "9999"
        }

        self.client = client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
            default_headers=custom_headers
        )

        try:
            models = client.models.list()
            self.model = models.data[0].id
        except Exception as e:
            print(f"Error fetching models from {openai_api_base}: {e}")
            self.model = "llama3"  # Fallback model name

        self.messages = []
        self.final_context = ""
        self.context_separator = context_separator

        # Performance metrics
        self.metrics = []

    def set_context(self, context_strings):
        contexts = []
        for context in context_strings:
            contexts.append(context)

        self.final_context = self.context_separator.join(contexts)
        self.on_user_message(self.final_context, display=False)
        self.on_server_message("Got it!", display=False)

    def get_context(self):
        return self.final_context

    def on_user_message(self, message, display=True):
        if display:
            print(f"[{self.name}] User message:", message)
        self.messages.append({"role": "user", "content": message})

    def on_server_message(self, message, display=True):
        if display:
            print(f"[{self.name}] Server message:", message)
        self.messages.append({"role": "assistant", "content": message})

    def chat(self, question):
        self.on_user_message(question)

        # Track performance metrics
        start_time = time.perf_counter()
        first_token_time = None
        last_token_time = None
        total_tokens = 0
        ttft_sent = False

        # Track time for each token for accurate ITL calculation
        token_times = []

        # For debugging
        print(f"[{self.name}] Starting chat request")

        try:
            chat_completion = self.client.chat.completions.create(
                messages=self.messages,
                model=self.model,
                temperature=0,
                stream=True,
                seed=42,
            )

            server_message = []
            for chunk in chat_completion:
                chunk_message = chunk.choices[0].delta.content
                if chunk_message is not None:
                    # Record time for this token
                    current_time = time.perf_counter()

                    if first_token_time is None:
                        first_token_time = current_time
                        ttft = first_token_time - start_time
                        ttft_ms = ttft * 1000  # Convert to ms

                        # Send TTFT metric before any content
                        ttft_sent = True
                        yield f"<metric:ttft:{ttft_ms:.2f}>"
                        print(f"[{self.name}] Sent TTFT metric: {ttft_ms:.2f} ms")

                    last_token_time = current_time
                    token_times.append(current_time)
                    total_tokens += 1

                    # Add to server message for complete record
                    server_message.append(chunk_message)

                    # Stream each token immediately for responsive UI
                    yield chunk_message

            complete_message = "".join(server_message)
            self.on_server_message(complete_message)

            # Calculate metrics
            ttft = first_token_time - start_time if first_token_time else 0
            total_time = last_token_time - start_time if last_token_time else 0

            # Calculate ITL (Inter-token Latency) in ms if we have tokens
            itl_ms = 0
            if total_tokens > 0:
                # ITL is the average time between tokens, including the first token
                itl_ms = (last_token_time - start_time) / total_tokens * 1000
                # Send ITL metric after completion
                yield f"<metric:itl:{itl_ms:.2f}>"
                print(f"[{self.name}] Sent ITL metric: {itl_ms:.2f} ms")

            # Store metrics with ITL
            self.metrics.append({
                "ttft": ttft * 1000,  # Convert to ms
                "itl": itl_ms,
                "total_time": total_time,
                "tokens": total_tokens
            })

            # If we never got a first token, send default metrics
            if not ttft_sent:
                yield "<metric:ttft:10000.00>"
                yield "<metric:itl:1000.00>"
                print(f"[{self.name}] No tokens received, sending default metrics")

        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(f"[{self.name}] {error_message}")

            # Send default metrics on error
            if not ttft_sent:
                yield "<metric:ttft:10000.00>"
                yield "<metric:itl:1000.00>"
                print(f"[{self.name}] Error occurred, sending default metrics")

            yield error_message
            self.on_server_message(error_message)

    def get_last_metrics(self):
        if self.metrics:
            return self.metrics[-1]
        return None

    def get_avg_metrics(self, last_n=None):
        if not self.metrics:
            return None

        metrics_to_use = self.metrics[-last_n:] if last_n else self.metrics

        avg_ttft = sum(m["ttft"] for m in metrics_to_use) / len(metrics_to_use)
        avg_itl = sum(m["itl"] for m in metrics_to_use) / len(metrics_to_use)

        return {
            "avg_ttft": avg_ttft,
            "avg_itl": avg_itl
        }