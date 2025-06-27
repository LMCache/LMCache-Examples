FROM lmcache/vllm-openai:latest

USER root
RUN apt-get update \
 && apt-get install -y sed \
 && rm -rf /var/lib/apt/lists/*

# 1. Comment out the original init call
RUN sed -i \
    's/ensure_kv_transfer_initialized(vllm_config)/# &/' \
    /opt/venv/lib/python*/site-packages/vllm/v1/worker/gpu_worker.py

# 2. Append our LMCache patch right after `self.model_runner.load_model()`
RUN sed -i "/self\.model_runner\.load_model()/a\        from lmcache.v1.compute.models.utils import VLLMModelTracker\n\        from lmcache.integration.vllm.utils import ENGINE_NAME\n\        VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)\n\        ensure_kv_transfer_initialized(self.vllm_config)" \
    /opt/venv/lib/python*/site-packages/vllm/v1/worker/gpu_worker.py

USER 1000
