import os
BLEND_SPECIAL_STR = " # # "

from contextlib import asynccontextmanager
from dataclasses import asdict
import time

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def _build_llm():
    engine_cfg = EngineArgs(
        model=MODEL_NAME,
        max_model_len=30000,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        max_num_batched_tokens=20480,
        enforce_eager=True,
    )
    return LLM(**asdict(engine_cfg))

@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = _build_llm()
    app.state.tokenizer = tokenizer
    app.state.llm = llm
    yield

app = FastAPI(lifespan=lifespan)

class GenRequest(BaseModel):
    prompt: list[int] | str          # allow raw text OR token-ids
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 10
    req_str: str = "request"

class GenResponse(BaseModel):
    texts: list[str]
    generation_time: float
    req_str: str

@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest):
    if isinstance(req.prompt, str):
        prompt_ids = app.state.tokenizer.encode(req.prompt)
    else:
        prompt_ids = req.prompt

    sp = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )

    start = time.time()
    outputs = app.state.llm.generate(prompt_token_ids=prompt_ids, sampling_params=sp)
    texts = [o.outputs[0].text for o in outputs]
    elapsed = time.time() - start

    print("-" * 50)
    for t in texts:
        print(f"Generated text: {t!r}")
    print(f"Generation took {elapsed:.2f} s, {req.req_str} request done.")
    print("-" * 50, flush=True)

    return GenResponse(texts=texts, generation_time=elapsed, req_str=req.req_str)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_backend_no_blend:app", host="0.0.0.0", port=8001, log_level="info")
