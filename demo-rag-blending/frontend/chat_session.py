import threading, time, requests
from transformers import AutoTokenizer

MODEL_NAME          = "mistralai/Mistral-7B-Instruct-v0.2"
BLEND_SPECIAL_STR   = " # # "

_tokenizer = None
_tok_lock  = threading.Lock()

def get_tokenizer():
    global _tokenizer
    with _tok_lock:
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return _tokenizer


class ChatSession:
    """
    Talks to FastAPI `/generate`.
    Streams two chunks:
        • generated text
        • metrics block  (server gen_time, round-trip, overhead)
    """

    def __init__(self, port: int, blend_special_str: str = BLEND_SPECIAL_STR):
        tok             = get_tokenizer()
        self.sep_ids    = tok.encode(blend_special_str)[1:]  
        self.temperature = 0.0
        self.context_ids = []
        self.context = ""                                 
        self.url         = f"http://localhost:{port}/generate"

    def set_context(self, context_list):
        tok = get_tokenizer()
        if not context_list:
            self.context_ids = []
            return
        
        # build context string
        self.context = context_list[0]
        for chunk in context_list[1:]:
            self.context += BLEND_SPECIAL_STR + chunk

        ids = tok.encode(context_list[0])            
        for chunk in context_list[1:]:
            ids += self.sep_ids + tok.encode(chunk)[1:] 
        self.context_ids = ids

    def get_context(self):
        return self.context

    def chat(self, question: str):
        tok = get_tokenizer()
        prompt_ids = (
            self.context_ids +
            self.sep_ids +
            tok.encode(question)[1:]          # drop BOS
        )

        payload = {
            "prompt": prompt_ids,
            "temperature": self.temperature,
            "top_p": 0.95,
            "max_tokens": 1,
            "req_str": "chat",
        }

        t0 = time.time()
        r  = requests.post(self.url, json=payload, timeout=120)
        t1 = time.time()

        r.raise_for_status()
        data = r.json()

        gen_time   = data.get("generation_time", 0.0)
        total_time = t1 - t0
        overhead   = total_time - gen_time
        text       = data.get("texts", [""])[0]

        print("-" * 60)
        print(f"Request 'chat'")
        print(f"Generated text          : {text!r}")
        print(f"Server generation_time  : {gen_time:.2f} s")
        print(f"Client round-trip time  : {total_time:.2f} s")
        print(f"⇢ Network / overhead    : {overhead:.2f} s")
        print("-" * 60, flush=True)

        yield text
        yield f"\n\n(TTFT: {gen_time:.2f} s)" 