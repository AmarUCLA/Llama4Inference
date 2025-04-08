#!/bin/bash

vllm serve "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" \
  --tensor-parallel-size 8 \
  --max-model-len 430000 \
  --limit-mm-per-prompt image=10 \
  --host=0.0.0.0 \
  --port=8080 \
  --kv-cache-dtype fp8
