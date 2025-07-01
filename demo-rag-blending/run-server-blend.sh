#!/usr/bin/env bash
set -e

# --- CONFIGURATION: fill these in before running ---
MODEL="mistralai/Mistral-7B-Instruct-v0.2"        # LLM model name
LOCAL_HF_HOME="/root/.cache/huggingface" # your Hugging Face cache dir
DATA_DIR="$(pwd)/data"                            # where your text chunks live
HOST_PORT_LMCACHE=8000
IMAGE="lmcache/vllm-openai:latest"
PATCHED_IMAGE="lmcache/vllm-openai:patched"
CONTAINER_NAME_LMCACHE="kv-blending-server-lmcache"
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

# 3. Build the patched image
echo "🔧  Building patched image…"
sudo docker build -t "${PATCHED_IMAGE}" .

# 4. Start LMCache-enabled vLLM server on GPU 0
docker run --rm -it \
  --name "${CONTAINER_NAME_LMCACHE}" \
  --gpus '"device=0"' \
  -v "${LOCAL_HF_HOME}:/root/.cache/huggingface" \
  -v "${DATA_DIR}:/input" \
  -v "${DATA_DIR}:/data" \
  -v "$(pwd):/workspace" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --env "LMCACHE_USE_EXPERIMENTAL=True" \
  --env "LMCACHE_CHUNK_SIZE=256" \
  --env "LMCACHE_LOCAL_CPU=True" \
  --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
  --env "LMCACHE_MAX_LOCAL_CPU_SIZE=200" \
  --env "LMCACHE_ENABLE_BLENDING=True" \
  --env "LMCACHE_USE_LAYERWISE=True" \
  --ipc=host --network=host \
  --entrypoint /bin/bash \
  "${PATCHED_IMAGE}" \
  -c "source /opt/venv/bin/activate && python llm_backend_blend.py"

echo "✅ LMCache + Blending server started:
  • LMCache server: http://localhost:${HOST_PORT_LMCACHE}" 
