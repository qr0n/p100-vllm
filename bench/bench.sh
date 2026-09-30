#!/bin/bash
# The fixed throughput workload behind every number in the README: random 512-token
# prompts, 400 output tokens (ignore_eos), seed 0, 4 prompts at concurrency 1 and
# 16 prompts at concurrency 4. Benchmark a FRESH server each time: a second run on
# the same instance hits the prefix cache (seed-0 prompts repeat) and reads high.
#
# usage: MODEL=/path/to/hf/model NAME=<served-model-name> [PORT=8000] [VLLM=vllm] ./bench.sh <label>
set -e
L=${1:?label}; M=${MODEL:?set MODEL to the model dir}; N=${NAME:?set NAME to --served-model-name}
P=${PORT:-8000}; V=${VLLM:-vllm}
curl -sf localhost:$P/v1/models >/dev/null || { echo "no server on :$P"; exit 1; }
mkdir -p runs
for C in 1 4; do
  NP=$(( C == 1 ? 4 : 16 ))
  out=runs/$(date -u +%Y%m%dT%H%M%S)-$L-c$C.txt
  $V bench serve --backend openai --base-url http://localhost:$P \
    --model "$N" --tokenizer "$M" --dataset-name random \
    --random-input-len 512 --random-output-len 400 --num-prompts $NP \
    --max-concurrency $C --ignore-eos --seed 0 > "$out" 2>&1
  g(){ grep -m1 "$1" "$out" | awk -F: '{gsub(/ /,"",$2); print $2}'; }
  echo "$(date -u +%FT%TZ) $L c=$C out_tok_s=$(g 'Output token throughput') medTPOT_ms=$(g 'Median TPOT') meanTTFT_ms=$(g 'Mean TTFT') failed=$(g 'Failed requests')"
done
