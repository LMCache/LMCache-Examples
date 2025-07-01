#!/usr/bin/env bash
set -e

MODEL="mistralai/Mistral-7B-Instruct-v0.2"        # LLM model name
LOCAL_HF_HOME="/root/.cache/huggingface" # your Hugging Face cache dir
DATA_DIR="$(pwd)/data"                            # where your text chunks live
HOST_PORT_DEFAULT=8001
IMAGE="lmcache/vllm-openai:latest"
PATCHED_IMAGE="lmcache/vllm-openai:patched"
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
cleanup_container "${CONTAINER_NAME_DEFAULT}"

# 5. Start default vLLM server (no LMCache) on GPU 1
docker run --rm -it \
  --name "${CONTAINER_NAME_DEFAULT}" \
  --gpus '"device=1"' \
  -v "${LOCAL_HF_HOME}:/root/.cache/huggingface" \
  -v "${DATA_DIR}:/input" \
  -v "${DATA_DIR}:/data" \
  -v "$(pwd):/workspace" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --ipc=host --network=host \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -c "source /opt/venv/bin/activate && python llm_backend_no_blend.py"

echo "✅ Default (no blending) server started:
  • Default server: http://localhost:${HOST_PORT_DEFAULT}"
  