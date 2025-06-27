import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.config import KVTransferConfig

# LMCache specific imports (adapted from blend.py)
from lmcache.integration.vllm.utils import ENGINE_NAME # Assuming lmcache is installed
from lmcache.v1.cache_engine import LMCacheEngineBuilder # Assuming lmcache is installed

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # As decided
LMCACHE_ENABLED = os.environ.get("LMCACHE_ENABLED", "False").lower() == "true"
LMCACHE_CHUNK_SIZE = os.environ.get("LMCACHE_CHUNK_SIZE", "256")
LMCACHE_BLEND_SPECIAL_STR = os.environ.get("LMCACHE_BLEND_SPECIAL_STR", " # # ")
# LMCACHE_USE_LAYERWISE = os.environ.get("LMCACHE_USE_LAYERWISE", "True").lower() == "true" # From blend.py
# LMCACHE_LOCAL_CPU = os.environ.get("LMCACHE_LOCAL_CPU", "True").lower() == "true"
# LMCACHE_MAX_LOCAL_CPU_SIZE = os.environ.get("LMCACHE_MAX_LOCAL_CPU_SIZE", "5")


app = FastAPI()

# Global LLM engine and tokenizer
llm_engine = None
tokenizer = None

def setup_lmcache_env_vars():
    """Sets up environment variables for LMCache if enabled."""
    if LMCACHE_ENABLED:
        os.environ["LMCACHE_CHUNK_SIZE"] = LMCACHE_CHUNK_SIZE
        os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
        os.environ["LMCACHE_BLEND_SPECIAL_STR"] = LMCACHE_BLEND_SPECIAL_STR
        os.environ["LMCACHE_USE_LAYERWISE"] = os.environ.get("LMCACHE_USE_LAYERWISE", "True") # Default to True

        # From blend.py, assuming CPU backend for now
        # Modify as needed for disk or other backends based on original run-server.sh or further config
        os.environ["LMCACHE_LOCAL_CPU"] = os.environ.get("LMCACHE_LOCAL_CPU", "True")
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = os.environ.get("LMCACHE_MAX_LOCAL_CPU_SIZE", "5")

        # Explicitly disable disk if CPU is on, or manage exclusivity if both can be configured
        if os.environ["LMCACHE_LOCAL_CPU"] == "True":
            os.environ["LMCACHE_LOCAL_DISK"] = "False"

        print("LMCache environment variables set for blending.")
    else:
        # Ensure blending is off if LMCache itself is not globally enabled for this instance
        os.environ["LMCACHE_ENABLE_BLENDING"] = "False"
        print("LMCache is disabled for this instance.")


@app.on_event("startup")
async def startup_event():
    global llm_engine, tokenizer

    setup_lmcache_env_vars()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer for {MODEL_NAME} loaded.")

    engine_args_dict = {
        "model": MODEL_NAME,
        "max_model_len": 8000,  # From blend.py example
        "gpu_memory_utilization": 0.8, # From blend.py example
        "disable_log_stats": False, # Enable stats for debugging
        "disable_log_requests": False,
    }

    if LMCACHE_ENABLED:
        # KVTransferConfig setup from blend.py
        # Ensure lmcache.integration.vllm.utils.ENGINE_NAME is correctly defined and LMCacheEngineBuilder is available
        try:
            ktc = KVTransferConfig(
                kv_connector="LMCacheConnectorV1", # This name must match LMCache's registration
                kv_role="kv_both", # Or "kv_sender", "kv_receiver" depending on role
            )
            engine_args_dict["kv_transfer_config"] = ktc
            engine_args_dict["enable_prefix_caching"] = False # As in blend.py
            print(f"LMCacheConnectorV1 configured with role 'kv_both'. ENGINE_NAME: {ENGINE_NAME}")
            # Initialize LMCache Engine (if needed separately, or if vLLM's connector handles it)
            # LMCacheEngineBuilder.build(ENGINE_NAME) # This might be needed here or handled by vLLM
        except Exception as e:
            print(f"Error setting up LMCache KVTransferConfig: {e}. LMCache might not be properly installed or configured.")
            # Fallback or raise error depending on strictness
            # For now, we'll let AsyncEngineArgs creation fail if ktc is problematic

    engine_args = AsyncEngineArgs(**engine_args_dict)
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"vLLM AsyncLLMEngine initialized with model {MODEL_NAME}. LMCache enabled: {LMCACHE_ENABLED}")

    # The ad-hoc changes from README_blend_kv_v1.md about VLLMModelTracker
    # would need to be applied to the vLLM source code or handled by the Docker image.
    # If LMCache is enabled, and those patches are necessary, this server might not work correctly
    # without them. For now, this script assumes they are in place if LMCache is used.

import time # Added for OpenAI compatible timestamps
# import json # For model_dump_json, though Pydantic handles it.

# OpenAI compatible Pydantic models
class OpenAIMessage(BaseModel):
    role: str
    content: str

class GenerationRequest(BaseModel): # Renamed internally from OpenAICompletionRequest for clarity
    messages: list[OpenAIMessage]
    model: str # Model name, though our server uses a fixed one from MODEL_NAME
    temperature: float = 0.7
    max_tokens: int = 512 # Renamed from max_length for OpenAI compatibility
    stream: bool = False
    # Other OpenAI params like top_p, stop can be added if needed

class ModelPermission(BaseModel): # Dummy, as vLLM's OpenAI server provides it
    id: str = "modelperm-dummy"
    object: str = "model_permission"
    created: int = int(time.time())
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "vllm" # Standard for vLLM's OpenAI server
    root: str | None = None
    parent: str | None = None
    permission: list[ModelPermission] = [ModelPermission()]

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = []

# For streaming responses (OpenAI compatible)
class ChatDelta(BaseModel):
    role: str | None = None
    content: str | None = None

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatDelta
    finish_reason: str | None = None

class ChatCompletionStreamResponse(BaseModel):
    id: str # Request ID
    object: str = "chat.completion.chunk"
    created: int = int(time.time())
    model: str # Model name
    choices: list[ChatCompletionStreamChoice]


@app.get("/v1/models", response_model=ModelList)
async def list_models_openai(): # Renamed to avoid conflict if any other list_models existed
    # Provide the model that this server is configured with
    # chat_session.py uses the first model from this list.
    return ModelList(data=[ModelCard(id=MODEL_NAME)])


@app.post("/v1/chat/completions") # Changed endpoint
async def create_chat_completion(request: GenerationRequest): # Request model updated
    if not llm_engine or not tokenizer:
        # OpenAI API returns 500 for server errors
        return {"error": {"message": "LLM Engine not initialized", "type": "server_error", "code": None}}, 500

    if not request.messages:
        return {"error": {"message": "No messages provided", "type": "invalid_request_error", "code": None}}, 400

    # Construct prompt_text using tokenizer.apply_chat_template if available
    prompt_text: str
    try:
        if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
            prompt_text = tokenizer.apply_chat_template(
                [msg.model_dump() for msg in request.messages], # Convert Pydantic models to dicts
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Basic fallback: concatenate content of messages.
            # This might not be robust for all models or chat histories.
            # The old chat_session.py was preparing a single string in messages[0].content for the context
            # and then appending the question. This new structure is more OpenAI-like.
            # If request.messages[0] contains the pre-formatted context + question, use that.
            if len(request.messages) == 1 and request.messages[0].role == "user":
                 prompt_text = request.messages[0].content # Assuming client prepared it
            else: # Attempt a simple join (less ideal)
                full_prompt_parts = []
                for msg in request.messages:
                    full_prompt_parts.append(f"{msg.role}: {msg.content}")
                prompt_text = "\n".join(full_prompt_parts)
                if request.messages and request.messages[-1].role == "user": # Add generation prompt
                    prompt_text += "\nassistant:"
        print(f"Constructed prompt for vLLM: '{prompt_text[:200]}...'")
    except Exception as e:
        print(f"Error applying chat template or constructing prompt: {e}")
        return {"error": {"message": f"Failed to construct prompt: {e}", "type": "invalid_request_error", "code": None}}, 400

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        # stop_sequences=request.stop if hasattr(request, 'stop') and request.stop else None
        # top_p=request.top_p if hasattr(request, 'top_p') and request.top_p else 1.0
    )

    request_id = f"chatcmpl-{os.urandom(12).hex()}" # OpenAI-like request ID
    request_created_time = int(time.time())

    async def streamer():
        full_response_text = ""
        results_generator = llm_engine.generate(prompt_text, sampling_params, request_id)

        async for i, request_output in enumerate(results_generator):
            delta_text = request_output.outputs[0].text[len(full_response_text):]
            full_response_text = request_output.outputs[0].text

            if delta_text or i == 0: # Send an initial empty chunk if no immediate delta, or always send delta
                choice_data = ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatDelta(role="assistant", content=delta_text if delta_text else None), # Send None if empty, OpenAI does
                    finish_reason=None
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=request_created_time,
                    model=request.model, # Use model from request as per OpenAI spec
                    choices=[choice_data]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Send the final chunk with finish_reason
        final_finish_reason = request_output.outputs[0].finish_reason
        choice_data = ChatCompletionStreamChoice(
            index=0,
            delta=ChatDelta(), # Empty delta for the last chunk
            finish_reason=final_finish_reason
        )
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=request_created_time,
            model=request.model,
            choices=[choice_data]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield f"data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(streamer(), media_type="text/event-stream")
    else:
        # Non-streaming response (Simplified: collect full text and return in OpenAI format)
        # This part needs full implementation if non-streaming is actually used by client.
        # For now, returning error as frontend expects stream.
        # A proper implementation would await the full_response_text and structure it.
        return {"error": {"message": "Non-streaming not fully implemented in this custom server", "type": "server_error", "code": None}}, 501


@app.on_event("shutdown")
async def shutdown_event():
    if LMCACHE_ENABLED:
        try:
            LMCacheEngineBuilder.destroy(ENGINE_NAME)
            print(f"LMCache Engine {ENGINE_NAME} destroyed.")
        except Exception as e:
            print(f"Error destroying LMCache Engine {ENGINE_NAME}: {e}")
    print("Application shutdown.")


# It's good practice to allow host and port to be configurable for Docker
if __name__ == "__main__":
    host = os.environ.get("VLLM_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("VLLM_SERVER_PORT", "8000")) # Default to 8000

    # LMCACHE_ENABLED will be set by the run script for the two instances
    print(f"Starting vLLM server on {host}:{port}, LMCache enabled: {LMCACHE_ENABLED}")

    uvicorn.run(app, host=host, port=port)

# Example to run this server:
# For LMCache enabled instance (on port 8000):
# LMCACHE_ENABLED=true VLLM_SERVER_PORT=8000 python demo-kv-blending-new/backend/vllm_server.py
# For default vLLM instance (on port 8001):
# LMCACHE_ENABLED=false VLLM_SERVER_PORT=8001 python demo-kv-blending-new/backend/vllm_server.py
#
# The HF_TOKEN and other model download specifics should be handled by the environment/Dockerfile
# where this server runs.
# The `blend.py` script's `setup_environment_variables` for LMCache chunk size, special string etc.
# are partially integrated here via environment variables at the top.
# The `build_llm_with_lmcache` logic is adapted into the `startup_event`.
#
# Note on streaming:
# The frontend's chat_session.py uses `response.iter_lines()` and expects OpenAI-like chunks.
# The current SSE stream `data: {delta}\n\n` is a common way to do it, but chat_session.py
# might need adjustment if it expects JSON objects like `{"choices": [{"delta": {"content": "text"}}]}`.
# vLLM has an OpenAI-compatible server entrypoint (`python -m vllm.entrypoints.openai.api_server`)
# which could be an alternative to this custom FastAPI server if more precise OpenAI compatibility is needed
# and if it can be integrated with LMCache in the same way.
# For now, this custom server provides a basic streaming endpoint.
# The `blend.py`'s `print_output` was for command line, not for server-client streaming.
#
# Ad-hoc vLLM changes:
# The README_blend_kv_v1.md mentioned:
# - Comment out `ensure_kv_transfer_initialized(vllm_config)` in `vllm/v1/worker/gpu_worker.py::init_worker_distributed_environment`.
# - Add `VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)` and then call `ensure_kv_transfer_initialized`
#   at the end of `vllm/v1/worker/gpu_worker.py::load_model`.
# These changes are crucial for LMCache+vLLM to work and must be present in the vLLM installation used by this server.
