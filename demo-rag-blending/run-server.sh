#!/usr/bin/env bash
set -e

# --- CONFIGURATION: fill these in before running ---
MODEL="mistralai/Mistral-7B-Instruct-v0.2"        # LLM model name
LOCAL_HF_HOME="$HOME/.cache/huggingface" # your Hugging Face cache dir
DATA_DIR="$(pwd)/data"                            # where your text chunks live
HOST_PORT_LMCACHE=8000
HOST_PORT_DEFAULT=8001
IMAGE="lmcache/vllm-openai:latest"
PATCHED_IMAGE="lmcache/vllm-openai:patched"
CONTAINER_NAME_LMCACHE="kv-blending-server-lmcache"
CONTAINER_NAME_DEFAULT="kv-blending-server-default"
LM_CACHE_CONFIG_FILE="/lmcache/demo/example.yaml" # inside the container

# Check that HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "❌ Error: HF_TOKEN environment variable is not set."
  echo "   Please run: export HF_TOKEN=your_hf_token_here"
  exit 1
fi

# 1. Pull the image
sudo docker pull "${IMAGE}"

# 2. Stop and Remove Existing Containers 
cleanup_container() {
  local name=$1
  if [ "$(sudo docker ps -a -q -f name=${name})" ]; then
    echo "Stopping and removing existing container ${name}..."
    sudo docker stop "${name}" >/dev/null
    sudo docker rm "${name}"   >/dev/null
  fi
}
cleanup_container "${CONTAINER_NAME_LMCACHE}"
cleanup_container "${CONTAINER_NAME_DEFAULT}"

# 3. Check if cache exists
# if [ -f "${DATA_DIR}/cache.pt" ]; then
#   echo "✅ Found existing cache. Skipping precompute step."
# else
#   echo "⚙️  Precomputing KV caches..."
#   sudo docker run \
#     --gpus '"device=0"' \
#     -v "${LOCAL_HF_HOME}:/root/.cache/huggingface" \
#     -v "${DATA_DIR}:/input" \
#     -v "${DATA_DIR}:/data" \
#     --env "HF_TOKEN=${HF_TOKEN}" \
#     --ipc=host --network=host \
#     --entrypoint python3 \
#     "${IMAGE}" /lmcache/demo/precompute.py \
#       --model "${MODEL}" \
#       --lmcache-con# Preprocess the text chunks

# if [ -f "${DATA_DIR}/cache.pt" ]; then
#   echo "✅ Found existing cache. Skipping precompute step."
# else
#   echo "⚙️  Precomputing KV caches..."
#   sudo docker run --gpus '"device=0"' \
#     -v "${LOCAL_HF_HOME}:/root/.cache/huggingface" \
#     -v "${DATA_DIR}:/input" \
#     -v "${DATA_DIR}:/data" \
#     --env "HF_TOKEN=${HF_TOKEN}" \
#     --ipc=host \
#     --network=host \
#     --entrypoint python3 \
#     apostacyh/vllm:lmcache-blend \
#     /lmcache/demo/precompute.py \
#     --model ${MODEL} --lmcache-config-file /lmcache/demo/example.yaml --data-path /input
#   echo "✅ Precompute finished."
# fi

# 4. Build the patched image
echo "🔧  Building patched image…"
sudo docker build -t "${PATCHED_IMAGE}" .

# 5. Start LMCache-enabled vLLM server on GPU 0
sudo docker run -d \
  --name "${CONTAINER_NAME_LMCACHE}" \
  --gpus '"device=0"' \
  -v "${LOCAL_HF_HOME}:/root/.cache/huggingface" \
  -v "${DATA_DIR}:/input" \
  -v "${DATA_DIR}:/data" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --env "LMCACHE_USE_EXPERIMENTAL=True" \
  --env "LMCACHE_CHUNK_SIZE=256" \
  --env "LMCACHE_LOCAL_CPU=True" \
  --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
  --env "LMCACHE_MAX_LOCAL_CPU_SIZE=200" \
  --env "LMCACHE_ENABLE_BLENDING=True" \
  --env "LMCACHE_USE_LAYERWISE=True" \
  --ipc=host --network=host \
  "${PATCHED_IMAGE}" \
  "${MODEL}" \
  --gpu-memory-utilization 0.9 \
  --max-model-len 30000 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 20480 \
  --port "${HOST_PORT_LMCACHE}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
#   --lmcache-config-file "${LM_CACHE_CONFIG_FILE}"

# 6. Start default vLLM server (no LMCache) on GPU 1
sudo docker run -d \
  --name "${CONTAINER_NAME_DEFAULT}" \
  --gpus '"device=1"' \
  -v "${LOCAL_HF_HOME}:/root/.cache/huggingface" \
  -v "${DATA_DIR}:/input" \
  -v "${DATA_DIR}:/data" \
  --env "VLLM_ATTENTION_BACKEND=FLASH_ATTN" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --ipc=host --network=host \
  "${IMAGE}" \
  "${MODEL}" \
  --gpu-memory-utilization 0.9 \
  --max-model-len 30000 \
  --port "${HOST_PORT_DEFAULT}"


echo "✅ All containers started:
  • LMCache server: http://localhost:${HOST_PORT_LMCACHE}
  • Default server: http://localhost:${HOST_PORT_DEFAULT}"
