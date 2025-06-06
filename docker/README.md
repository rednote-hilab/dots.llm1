# Docker

The docker images are available on [Docker Hub](https://hub.docker.com/repository/docker/rednotehilab/dots1/tags) based on the official images.

## vllm

You can start a server via vllm.

```shell
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    rednotehilab/dots1:vllm-openai-v0.9.0.1 \
    --model redmoe-ai-v1/dots.llm1.test \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --served-model-name dots1
```

Then you can verify whether the model is running successfully in the following way.

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dots1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ],
        "max_tokens": 32,
        "temperature": 0
    }'
```
