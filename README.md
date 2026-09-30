# vLLM on the Tesla P100 (GP100, sm_60)

Upstream vLLM dropped pre-Volta GPUs. These seven patches bring it back on the
P100 and then make it fast for **several concurrent requests**, which is where
vLLM earns its keep over llama.cpp on this card.

Measured on one machine: 2x Tesla P100-PCIE-16GB on different CPU sockets (no
NVLink, no working P2P), tensor parallel 2, fp16.

| model | c=1 tok/s | c=4 aggregate tok/s |
|---|---:|---:|
| Gemma 4 31B, first run | 15.5 | 21.5 |
| **Gemma 4 31B, all patches** | **35.2** | **90.7** |
| Qwen3.8-27B, first run | 28.1 | 63.5 |
| **Qwen3.8-27B, all patches** | **37.4** | **95.9** |

Workload: `bench/bench.sh` (random 512-token prompts, 400 output tokens, seed 0),
fresh server per run. Every row passed `bench/quality.py` (greedy tokens and
logprobs against a stored reference). For scale, the same box running llama.cpp
does roughly 60-73 tok/s aggregate at 4 agents on these models (different
harness, so compare loosely). A single request is still faster on llama.cpp with
MTP speculative decoding; use vLLM for concurrency.

## What each patch does

| patch | what | why |
|---|---|---|
| `0001` | sm_60/61 in the CUDA 12.x arch list, `build-pascal.sh` | vLLM will not compile for Pascal otherwise |
| `0002` | release/acquire stores and `atomicAdd(half*)` emulated on sm_6x | compile fixes |
| `0003` | rebind torch's `has_triton()`, rewrite sm_70-only PTX Triton emits | torch hard-codes capability >= 7.0, so Inductor and CUDA graphs of Triton kernels refuse to run |
| `0004` | host-mapped two-rank all-reduce; IPC custom all-reduce disabled pre-Volta | the IPC all-reduce **hangs** across CPU sockets (spins on peer memory); this one is 19.5 us vs NCCL's 33.5 us per decode all-reduce |
| `0005` | small-M Triton GEMM for large unquantized matrices | cuBLAS pads decode-sized GEMMs to 64-row tiles; matters for the 248k-vocab lm_head |
| `0006` | GPTQ/AutoRound load without Marlin; sm_60 exllama GPTQ kernel | Marlin needs sm_75. The kernel unpacks int4 straight into subnormal fp16 (`q * 2^-24`, exact; HFMA2 is full rate on subnormals on GP100): M=4 GEMM ~300 -> ~390 GB/s. `VLLM_GPTQ_SM60_KERNEL=0` = stock kernel |
| `0007` | attention kernels for head_size >= 256 on pre-Volta | the generic Triton kernel exceeds sm_60's 48 KB shared memory at Gemma's d=512 and pads/spills at d=256. Split-head-dim prefill plus a dot-free decode kernel (`tl.dot` needs M >= 16; decode has 2-8 query rows per KV head). Gemma c=4 21.5 -> 81.8 on its own |

## Install

Tested with CUDA 12.4 (`nvidia-cuda-toolkit`), gcc-13, Python 3.12, a Rust
toolchain in `~/.cargo`. **CUDA 13 dropped Pascal**; stay on 12.x.
`torch 2.13.0+cu126` is the only torch 2.13 wheel that still ships sm_60.

    git clone --filter=blob:none https://github.com/vllm-project/vllm && cd vllm
    git checkout aedaba866
    git am /path/to/p100-vllm/patches/*.patch

    python3.12 -m venv .venv && .venv/bin/pip install uv
    .venv/bin/uv pip install --python .venv/bin/python "torch==2.13.0+cu126" "torchvision==0.28.0+cu126" \
        --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
    .venv/bin/uv pip install --python .venv/bin/python "cmake>=3.26.1" ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" \
        "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel "jinja2>=3.1.6" regex build
    export PATH=$PWD/.venv/bin:$HOME/.cargo/bin:$PATH TORCH_CUDA_ARCH_LIST=6.0 CUDAHOSTCXX=/usr/bin/g++-13 \
        CC=gcc-13 CXX=g++-13 CUDA_HOME=/usr MAX_JOBS=24 NVCC_THREADS=1 VLLM_TARGET_DEVICE=cuda
    .venv/bin/uv pip install --python .venv/bin/python --no-build-isolation -e .
    # torchaudio / torchcodec must also be the cu126 builds, or imports fail:
    .venv/bin/uv pip install --python .venv/bin/python --no-deps --reinstall "torchaudio==2.11.0+cu126" \
        "torchcodec==0.16.0+cu126" --index-url https://download.pytorch.org/whl/cu126

After changing C++/CUDA sources, `./build-pascal.sh` rebuilds incrementally;
copy the rebuilt `build-pascal/*.so` into `vllm/`.

## Serve

    # Qwen3.8-27B, GPTQ int4 (https://huggingface.co/sokada4/Qwen3.8-27B-GPTQ-Int4)
    vllm serve sokada4/Qwen3.8-27B-GPTQ-Int4 --tensor-parallel-size 2 --dtype half \
        --max-model-len 32768 --max-num-seqs 16 --kv-cache-memory-bytes 2800000000

    # Gemma 4 31B, AutoRound int4 (https://huggingface.co/Intel/gemma-4-31B-it-int4-AutoRound)
    vllm serve Intel/gemma-4-31B-it-int4-AutoRound --tensor-parallel-size 2 --dtype half \
        --max-model-len 32768 --max-num-seqs 16 --language-model-only --kv-cache-memory-bytes 4000000000

- `--dtype half`: the P100 has no bf16.
- `--kv-cache-memory-bytes`: the **first** start compiles Inductor/Triton kernels
  (~15 min cold; a few minutes once cached) and that inflates vLLM's memory
  profiling until startup fails with "No available memory for the cache blocks".
  Pinning the KV size skips the profiling. Tune it to your cards.
- Tool calling needs `--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3` (Qwen) or `gemma4` for both (Gemma).
- GPTQ checkpoints must be **symmetric** int4 (exllama is the only kernel that
  runs here). Asymmetric AWQ does not load.

## Caveats

- One machine, two models. The mechanisms should transfer; your numbers will not.
- **P40 / GTX 10xx (sm_61) is untested.** The attention and GPTQ kernels lean on
  fp16 arithmetic, which runs at 1/64 rate on sm_61, so expect them to be slow there.
- fp16 end to end. Quality was gated with greedy-token/logprob comparisons over
  8 prompts per model, not a full eval suite.
- Benchmark only a fresh server: a second run on the same instance hits the
  prefix cache and reads high. The first request of a new kernel shape pays
  Triton JIT time in its TTFT.
- vLLM moves fast; these patches are pinned to `aedaba866` and will need rebasing.

## Benchmark tools

- `bench/bench.sh` - the fixed workload. `MODEL=<hf dir> NAME=<served name> PORT=8000 ./bench.sh <label>`
- `bench/quality.py` - `PORT=8000 NAME=<served name> ./quality.py record` on the
  baseline, then `check` on each candidate. Raw completions; for instruct models
  that drift into repetition on raw text (Gemma), switch it to chat completions.

## License

vLLM is Apache-2.0 (© the vLLM contributors); so are these patches. See
`LICENSE`. Not affiliated with or endorsed by the vLLM project.
