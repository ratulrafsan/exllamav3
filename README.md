
<p align="center">
  <img src="doc/logo.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

[Installation](#installation) · [Supported models](#architecture-support) · [Examples](#examples) · [Quantization](#exl3-quantization) · [Community](#community)

ExLlamaV3 is an inference library for running local LLMs on modern consumer GPUs, with flexible quantization and parallel inference.

- **Quantization** - [EXL3](#exl3-quantization), based on QTIP, plus 2–8 bit cache quantization.
- **Parallel inference** - Flexible tensor-parallel and expert-parallel inference for consumer hardware setups.
- **CPU offloading** - Allows large MoE models to run with limited GPU resources. AVX2 and AVX512 support.  
- **Generation** - Continuous, dynamic batching, speculative decoding, multimodal support.
- **Integrations** - Broad [HF model support](#architecture-support), a [Transformers plugin](examples/transformers_integration.py), and an OpenAI-compatible API via [TabbyAPI](https://github.com/theroyallab/tabbyAPI/).

> [!TIP]
> **Looking for a server?** [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) is the official and recommended backend server. It provides an OpenAI-compatible API for local or remote inference, HF model downloading, embedding model support, and HF Jinja2 chat templates. Its startup script manages and installs prerequisites to help you get started.

## About this fork

This fork adds local patches to ExLlamaV3. The branch `local-patches` has one commit for each patch. The commits are on top of the release tag `v1.6.7`. The patches are not part of upstream ExLlamaV3, and upstream does not support them.

> [!IMPORTANT]
> **This fork is for one unusual hardware combination.** I tune the patches for my hardware only, with one model: Qwen3.8-Flash-Next, 4.05 bpw EXL3. I do not test other hardware or other models. The patches can be slower or not work on your system. Your results can be different (YMMV).

### Hardware and constraints

| Part | Detail |
|---|---|
| GPU 0 | RTX 3090, 24 GB, PCIe 4.0 x16 (20.6 GB/s measured) |
| GPU 1 | RTX 4090, 24 GB, PCIe 3.0 x4 (3.3 GB/s measured). This slot is a B660 chipset slot. |
| CPU | Intel i7-13700K: 8 P-cores and 8 E-cores, AVX2, no AVX-512 |
| RAM | 64 GB |
| Motherboard | ASUS ProArt B660-CREATOR D4. The 4090 and one NVMe share the chipset link to the CPU (DMI 4.0 x4). |

The constraints of this system are:

- **The 4090 has a slow link.** Its x4 link is approximately 6 times slower than the x16 link of the 3090. Thus, the CPU expert split goes only on layers of the 3090. Prefill sends experts only over the x16 link. The layers on the 4090 keep all experts in VRAM.
- **The VRAM is full.** At 215,040 tokens of context, the 3090 has almost no free VRAM. A context of 225,280 tokens does not load.
- **The 3090 has a power limit of 270 W.** Its GDDR6X memory reached 108 °C at full power. 350 W does not give faster decode.
- **The E-cores are slow for the CPU experts.** The CPU expert workers use only the 8 P-cores, with one pinned worker on each core. Workers on E-cores decrease decode from approximately 32 to 19 T/s.
- **The RAM is 64 GB.** Approximately 23 GB of expert weights stay in RAM. The 39 GB engram table streams from an NVMe disk.

The settings for this system are in the TabbyAPI configuration and start script. They are not part of this repository.

| Patch | Function |
|---|---|
| `EXL3_MOE_CPU_SPLIT_DRAFT` | Keeps the MTP layer fully in VRAM when the CPU expert split is on. |
| Expert placement | Saves the dynamic CPU-MoE expert placement to a file. Loads the placement at the next start. |
| Prefill pipeline | Adds engram lookahead, resident-first expert order, stream-ahead and an optional last-page merge. Each item has an `EXL3_*` switch. |
| Control vectors | Applies llama.cpp control vectors to the residual stream. Refer to [Control vectors](#control-vectors). Inkling-2-Mini has its own reader: refer to [Control vector for Inkling-2](#control-vector-for-inkling-2). |
| Pipelined prefill with CPU experts | Lets the two GPUs work on different prefill chunks when the CPU expert split is on. This is a port of [a commit by Auryn Macmillan](https://github.com/auryn-macmillan/exllamav3/commit/b38de78bd0376fcdc4f92344ed405800ac8374b6). The port also adds a lock to the n-gram row table. Prefill of prompts longer than one chunk is approximately 10% faster. `EXL3_PIPELINED_PREFILL=0` turns it off. |
| Late promote hook | Moves the promote-on-miss step off the critical path of decode and makes its budget a setting. Decode is approximately 17% faster on this system. Refer to [Late promote hook](#late-promote-hook). |
| Expert lookahead (experimental, off by default) | Predicts the experts of the next split layer and copies them from RAM to VRAM before the layer needs them. It needs a source build of the extension. On this system it makes decode slower. Refer to [Expert lookahead](#expert-lookahead). |
| Inkling-2-Mini (work in progress) | Adds the looped Inkling-2 model (text and image input): load from the source checkpoint or from an EXL3 conversion of its experts, cache, generator, CUDA-graph decode and decode kernels. It needs a source build of the extension for full speed, and TabbyAPI needs the patch `util/inkling2/tabbyapi_inkling2.patch`. It also has LoopCD decoding and a reader for control vectors with one direction for each layer application; both are off by default. Refer to [Inkling-2-Mini](#inkling-2-mini). |
| SenseNova Loop | Adds the looped SenseNova-Loop-9B model (text and image input) and its MTP head as a draft model. This is a port of [upstream pull request #609](https://github.com/turboderp-org/exllamav3/pull/609) and of the branch `sensenova-loop-mtp`, both by [Maxrio9](https://github.com/Maxrio9). Refer to [SenseNova Loop](#sensenova-loop). |

### Late promote hook

#### Overview

With the CPU expert split, some experts of each split layer are in VRAM and the others are in RAM. Promote-on-miss moves an expert from RAM to VRAM after the CPU worker calculated it, if the model uses it frequently. A less frequently used expert moves from VRAM to RAM.

Stock ExLlamaV3 has two limits here:

- **The budget is 6 promotions for each decode step, for all layers together.** On this system the budget is always fully used, and only 62% of the routed experts of the split layers are in VRAM.
- **The promote step runs at the start of the decode step, in the first split layer.** At that time the GPU has no work in its queue. Thus, the GPU waits while the host does the promote step. The host needs approximately 0.17 ms for each promotion with its demotion. A larger budget makes the GPU wait longer. A budget of 33 has a higher hit rate than a budget of 6, but decode is not faster.

The patch adds a mode that runs the promote step in the last split layer. At that time the host has put all earlier split layers in the queue, and the GPU is still working on them. Thus, the GPU does not wait for the promote step. A larger budget then gives faster decode.

The patch is off by default. Without the variables below, the behavior is the same as stock ExLlamaV3.

#### Settings

| Variable | Value | Default |
|---|---|---|
| `EXL3_MOE_CPU_PROMOTE_LATE` | `1` runs the promote step in the last split layer. `0` runs it in the first split layer. | `0` |
| `EXL3_MOE_CPU_PROMOTE_BUDGET` | The maximum number of promotions for each decode step, for all layers together. Each layer gets a maximum of one promotion for each step. | `6` |
| `EXL3_MOE_CPU_PROMOTE_RATIO` | A missed expert is promoted when its hit count is this many times the hit count of the least used expert of the layer in VRAM. | `1.5` |

On this system the start script sets `EXL3_MOE_CPU_PROMOTE_LATE=1` and `EXL3_MOE_CPU_PROMOTE_BUDGET=33`. 33 is the number of split layers.

To see the hit rate and the host time of the promote step, set `EXL3_MOE_CPU_SWAP_DEBUG=1`. The log then shows one line for each 128 decode steps.

#### How the patch works

The patch changes `exllamav3/modules/block_sparse_mlp_cpu.py`.

- `_split_swap_tick` counts the decode steps in the first split layer, as before. In late mode, it calls `_split_promote_step` from the last split layer of the text model that has spare slots.
- The order of operations stays correct. A transfer goes on the promotion stream after all work that the compute stream has in its queue. This work includes the issue, the expert kernels and the collect of each earlier layer of the same step. The map update of a completed transfer goes on the compute stream before the selection translate of the last layer, and before the translate of all layers in the next step.
- In late mode, the map updates go through pinned staging buffers. A copy from pageable host memory waits until the compute stream is empty. In the first test, this wait was approximately 15 ms for each step.

#### Test results

The test uses the hardware in [Hardware and constraints](#hardware-and-constraints), with TabbyAPI. Each run uses a new server and a saved expert placement. A run has 8 chat requests of 400 tokens on different topics, with greedy decode, thinking off, MTP and hybrid drafting. The decode speed is the mean of the 8 requests. The hit rate is the mean for the full run, and it includes the cold start.

| Condition | Decode, each run (T/s) | Mean | Hit rate | Host time of the promote step |
|---|---|---|---|---|
| Stock: first layer, budget 6 | 64.9 / 64.1 / 63.7 / 63.2 / 63.3 | 63.8 | 62% | 2.1 ms |
| First layer, budget 16 | 62.5 / 63.1 | 62.8 | 70% | 4.0 ms |
| First layer, budget 33 | 62.6 / 61.6 | 62.1 | 75% | 5.4 ms |
| Late, budget 6 | 68.9 / 67.6 | 68.2 | 63% | 2.6 ms |
| Late, budget 16 | 72.9 / 73.2 | 73.0 | 70% | 5.3 ms |
| Late, budget 33 | 75.6 / 73.8 / 74.6 | 74.7 | 75% | 7.1 ms |
| Late, budget 33, ratio 1.2 | 75.5 / 75.4 | 75.5 | 76% | 7.5 ms |

- Late mode with a budget of 33 is approximately 17% faster than stock.
- A ratio of 1.2 gives no clear difference. The difference is smaller than the difference between two runs of one condition.
- A test run with `EXL3_MOE_CPU_PROMOTE_VERIFY=1` compared 37,025 promotions and 37,011 demotions with the checkpoint. All were correct.

#### Limits

- The verify mode makes each transfer synchronous. Thus, it shows that the transferred data is correct, but it does not test the order of operations in late mode. No test covers that order.
- The output quality is not graded. With a larger budget, more experts move between the CPU and the GPU during a generation. The same weights give slightly different results on the two devices.
- `tests/moe_cpu/test_promote_step.py` was not run with this patch.
- The patch is tested with one model, one workload and all split layers on one GPU.
- The host time of the promote step increases with the budget. If the host time becomes longer than the time the GPU needs for the earlier layers, the GPU waits again.

### Expert lookahead

> [!WARNING]
> **This patch is experimental, and it is off by default.** On the system in [Hardware and constraints](#hardware-and-constraints), it makes decode slower: 60 T/s with the patch and 75 T/s without it. The patch is in this branch because the result can be different on other hardware. It is not tested on other hardware.

#### Overview

With the CPU expert split, the CPU worker calculates each routed expert that is not in VRAM. The idea of this patch comes from [TokenWing](https://github.com/runmyai/tokenwing), which uses it to read experts from an SSD before a layer needs them. This patch uses the idea for the transfer from RAM to VRAM:

1. While split layer L calculates, the router of layer L+1 runs on the router input of layer L. The result is a prediction of the experts that layer L+1 selects.
2. A kernel selects the predicted experts that are in RAM and gives each one a free GPU slot.
3. On a second CUDA stream, a kernel copies each of these experts from the pinned RAM arena to its GPU slot. A last kernel changes the placement map and writes the transfer to a log in pinned memory.
4. Layer L+1 then finds these experts in VRAM. The experts stay in VRAM.
5. The promote step of the host reads the log, and it moves less frequently used experts back to RAM to make free GPU slots.

All transfers from RAM to VRAM go through these kernels when the patch is on. The promote step sends its own promotions to the kernels as requests. The host does not wait for the GPU in this procedure.

#### Requirements

- **A source build of the extension.** The patch adds CUDA kernels (`exllamav3/exllamav3_ext/cpu/moe_lookahead.cu`). The release wheels do not contain them. With a wheel, the setting has no effect, and the log shows `Expert lookahead: off (the extension has no moe_la_step ...)`.
- The CPU expert split with the pinned arena (`EXL3_MOE_PINNED_ARENA=1`) and promote-on-miss.
- All split layers on one GPU.
- A router of the standard type, with no norm between the layer input and the router.

To build the extension without a change to the installed package:

```sh
CUDA_HOME=/usr/local/cuda-13.0 python setup.py build_ext --build-lib ../exl3_build/lib --build-temp ../exl3_build/tmp
PYTHONPATH=/path/to/exl3_build/lib:/path/to/this/checkout python main.py
```

#### Settings

| Variable | Value | Default |
|---|---|---|
| `EXL3_MOE_CPU_LOOKAHEAD` | The maximum number of predicted experts that one split layer uploads for each decode step. `0` turns the patch off. | `0` (off) |
| `EXL3_MOE_CPU_LOOKAHEAD_ROWS` | A forward pass with more rows than this value does not use predictions. | `4` |
| `EXL3_MOE_CPU_LOOKAHEAD_BLOCKS` | The number of blocks of the upload kernel for each expert. `0` uses one block for each run of 8 tiles. This is the fastest upload. | `0` |
| `EXL3_MOE_CPU_LOOKAHEAD_DRY` | `1` runs the predictor but uploads none of its experts. Use it to measure the cost of the predictor. | `0` |
| `EXL3_MOE_CPU_PROMOTE` | The number of spare GPU slots for each split layer. This is a stock setting. The lookahead uses more free slots: the tests use `8` with a lookahead of 1 or 2, and `12` with a lookahead of 4. | `4` |

When the patch is on, the log shows a line of this type:

```
 -- Expert lookahead: on, 2 per layer per step, 32 of 33 split layers predicted
```

With `EXL3_MOE_CPU_SWAP_DEBUG=1`, the promote-on-miss line also shows the number of uploads for each step.

#### Test results

The test uses the hardware in [Hardware and constraints](#hardware-and-constraints), with TabbyAPI, the late promote hook and a budget of 33. All runs use the extension from a source build. A run has 8 chat requests of 400 tokens. Each run uses a new server.

| Condition | Decode (T/s) | Hit rate |
|---|---|---|
| Lookahead off, 2 runs | 75.0 / 75.1 | 76% |
| Kernel transfers, no predictions (`EXL3_MOE_CPU_LOOKAHEAD_ROWS=0`) | 68.7 | 74% |
| Predictor on, no uploads of its experts (`EXL3_MOE_CPU_LOOKAHEAD_DRY=1`) | 67.3 | 75% |
| Lookahead 1 | 63.4 | 80% |
| Lookahead 2, 2 runs | 60.3 / 60.4 | 82% |
| Lookahead 4 | 55.2 | 84% |

- **The patch does its function.** With a lookahead of 2, the hit rate is 86% to 89% after the cold start. The CPU worker calculates 0.99 experts for each row of a split layer, and 1.78 without the patch.
- **The data is correct.** A test run with `EXL3_MOE_CPU_PROMOTE_VERIFY=1` compared 108,452 promotions and 108,383 demotions with the checkpoint. All were correct.
- **Decode is slower.** The input of each split layer comes approximately 0.29 ms later to the CPU worker (0.52 ms and not 0.23 ms). This is more than the time that the worker saves (0.15 ms and not 0.22 ms for each layer).
- **The predictor is not the cause.** It costs approximately 2%.

The cause is the upload kernel. It reads the experts from mapped host memory. The kernels of the CPU hand-off also use mapped host memory: one kernel writes the layer input to a pinned slot, and one kernel reads the result. While an upload runs, these kernels wait. A test without the model shows the same effect (`upload_interference_bench.py` in the benchmark files of this system, which are not part of this repository). Beside 200 uploads, a stream of small kernels that use pinned memory is 17.3 ms slower with uploads from mapped memory. It is 3.5 ms slower with uploads by `cudaMemcpyAsync`.

A smaller upload kernel does not help. With fewer blocks, the upload is slower and the wait is longer: 58.5 T/s with 128 blocks, 45.3 T/s with 16 blocks and 19.2 T/s with 4 blocks.

#### When the patch can help

These are estimates. They are not tested.

- **A slow CPU for the expert calculation.** On this system, one expert on the CPU costs approximately 57 µs. An upload must also move one other expert back to RAM. Thus, an upload must prevent approximately two CPU calculations to be useful. If one expert on the CPU costs much more, for example with fewer cores or without AVX2, the balance changes.
- **More memory bandwidth.** The uploads and the CPU worker use the same RAM. This system has DDR4.
- **A different upload path.** An upload by `cudaMemcpyAsync` does not make the hand-off kernels wait as much. It needs a host thread that reads the plan of the select kernel and starts the copies. This is not built. An estimate from the measured costs gives 70 to 77 T/s on this system, against 75 T/s without the patch.

#### How the patch works

- `exllamav3/exllamav3_ext/cpu/moe_lookahead.cu` has the select, upload, commit and host-update kernels. `exllamav3/modules/moe_lookahead.py` has the tables of pointers and the calls. `block_sparse_mlp_cpu.py` calls them from the submit of each split layer and from the promote step.
- The placement map, the set of experts in transfer and the pool of free GPU slots are in device memory. Only kernels on the compute stream change them, in stream order. The one exception is the map update of a completed upload on the second stream.
- That map update is safe at any time. Before it, no router ID points to the GPU slot. After it, the expert is still in its RAM slot until the host sees the log entry. The host then uses the RAM slot for a demotion, and it puts this demotion after all work of the compute stream.
- A demoted expert gives its GPU slot back to the pool in the same kernel that changes its map entry. Thus, no kernel can read the slot when a new upload writes to it.
- If the previous upload of a layer is not complete, the layer does not start a new one in that step.
- Before a forward pass of prefill size, the host waits for all uploads.
- For the lookahead, the arena is registered with the `MAPPED` flag, so that kernels can read it.

#### Limits

- The patch is tested with one model and one system only.
- No unit test covers the patch. The verify mode makes the checks after a synchronization, so it does not test the order of operations.
- The output quality is not graded. More experts move between the CPU and the GPU during a generation than without the patch.
- The patch does not use a trained correction of the prediction. On a recorded session, a correction increased the part of the CPU experts that the prediction finds from 59% to 69%.

### Inkling-2-Mini

> [!NOTE]
> Text and image input work, in the Python API and in TabbyAPI, and the MTP head works as a draft model. The port is still being worked on: refer to [Limits](#limits) and [TODO](#todo). Converted models are on Hugging Face: refer to [Published conversions](#published-conversions).

#### Overview

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is a 26.4B-parameter MoE model from Thinking Machines. It is a looped transformer:

- The model has 24 layers of weights. Layers 6 to 17 run three times for each token, so a token goes through 48 layer applications.
- Passes 2 and 3 start again from the hidden state before the loop. They receive "messages" from the pass before, mixed in through small gates.
- Attention has no RoPE. It uses a relative position bias computed from the hidden state, a sink logit for each token, and short convolutions on K, V and the layer outputs.
- 20 layers use a sliding window of 1,024 tokens. 4 layers are global.
- Each layer has 128 experts with 8 active for each token.

This fork adds the architecture `Inkling2LoopForConditionalGeneration`. It is not in upstream exllamav3.

The layer math and the NVFP4 / MXFP8 dequantizers are adapted from [0xSero/inkling2-exl3](https://github.com/0xSero/inkling2-exl3) (MIT), which supports the larger Inkling-2-Small without the loop. The design of the decode step (graph replay, split attention) follows the notes of [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090) (Apache 2.0).

#### Current state

| Item | State |
|---|---|
| Load from the source checkpoint (experts stay in NVFP4 / MXFP8) | Works. 17.6 GiB of VRAM. Slow: approximately 6 T/s. |
| EXL3 conversion of the experts | Works. Refer to [Convert the model](#convert-the-model). |
| Cache, generator, tabbyAPI | Works. TabbyAPI needs the patch in [Use with TabbyAPI](#use-with-tabbyapi). Each loop pass has its own cache layers. FP16 cache only. |
| Context length | Tested with a 161,543-token prompt on the RTX 4090 + RTX 3090, with and without the draft model. |
| Decode as CUDA graph replays, with decode kernels | Works for one sequence with EXL3 experts. |
| Multi-token graph steps (verify drafted tokens) | Works for a maximum of 8 tokens in one forward pass. The MTP draft model uses them. |
| Prefill | Attention and the block attention residual run through the extension (cuBLAS products on fp16 inputs with fp32 accumulation, own kernels). The rest is plain PyTorch. The pipelined prefill of the generator is off for this model. Refer to [Limits](#limits). |
| MTP head (`mtp.safetensors`, DSpark draft model) | Works as a draft model (`architecture/inkling2_dspark.py`), in the Python API and in TabbyAPI (`draft_mode: model`, `draft_num_tokens: 7`, `dynamic_draft: true`; the model directory comes from `util/inkling2/make_dspark_dir.py`). Refer to the TabbyAPI results below. Not quantized. |
| Vision | Works: the vision tower is the `vision` component of the model (`architecture/inkling2_vision.py`), in the Python API and in TabbyAPI (`vision: true`, with the TabbyAPI patch). Refer to [Images](#images). |
| LoopCD decoding | Works in both modes of the paper (hidden state and logits), uncached, cached and as graph replays. Off by default. In the tests here it does not give a measurable gain for this model. Refer to [LoopCD](#loopcd). |
| Control vectors | Works for GLP files with one direction for each layer application, and for files with one direction for each stored layer. Off by default. Refer to [Control vector for Inkling-2](#control-vector-for-inkling-2). |
| Tensor parallelism | Not supported. |

#### Requirements

- **A source build of the extension for full speed.** The patch adds CUDA kernels (`exllamav3/exllamav3_ext/inkling2.cu`) and the Inkling-2 activation in the expert kernels. The release wheels do not contain them. With a wheel, the model gives correct results on a slow path, and the log shows a line that starts with ` !! Inkling2:`.
- **The native CUDA allocator.** CUDA graph capture does not work with `cudaMallocAsync`. In tabbyAPI, set `cuda_malloc_async: false`. With the async allocator the model decodes without graphs and shows a `!!` line.
- **The FP16 cache.** Quantized cache modes are not supported.
- **A GPU split with free memory.** Attention on the global layers needs working memory that grows with the context. On this system, tabbyAPI uses `autosplit_reserve: [2560, 1024]` with a cache of 163,840 tokens. The peak with a 161,543-token prompt is 21.1 GiB on the first GPU.
- **The loop on one GPU.** Layers 6 to 17 run three times. If one of them is on a different GPU than the others, each pass crosses GPUs and decode is approximately 10% slower. The input embedding table stays in system RAM (0.9 GiB) to make room, and in tabbyAPI the draft model goes to the second GPU (`draft_gpu_split: [0, 8]`). `util/inkling2/graph_profile.py --cache <tokens> --chunk <chunk size>` shows the split and the graph segments.

Build the extension with:

```
# in the root of this repository
python setup.py build_ext --build-lib ../exl3_build/lib --build-temp ../exl3_build/tmp
```

Then put `exl3_build/lib` first on `PYTHONPATH`.

#### Use with TabbyAPI

> [!IMPORTANT]
> TabbyAPI needs a patch to serve this model: `util/inkling2/tabbyapi_inkling2.patch` in this repository. Apply it to the TabbyAPI checkout before you load the model.

```
# in the root of the TabbyAPI checkout
git apply /path/to/exllamav3/util/inkling2/tabbyapi_inkling2.patch
```

The patch is made against TabbyAPI commit `4728158` (2026-11-14). It adds:

- **The Inkling-2 message format.** The model writes each part of its reply as a message with a content type (`<|content_thinking|>`, `<|content_text|>`, `<|content_invoke_tool_json|>`). The patch detects the format from the tokenizer and splits the output into reasoning, answer and tool calls, for streaming and non-streaming requests.
- **The tool call format** (`endpoints/OAI/utils/toolcall_formats/inkling2.py`).
- **Image parts.** TabbyAPI joins the parts of a message into one text. The model needs each image as a message of its own, so the patch gives the parts to the chat template as a list and puts the image tokens where the template writes its image placeholder.

Without the patch, TabbyAPI does not separate reasoning from the answer, does not parse tool calls, and puts images inside a text message, which is not the format of the model.

TabbyAPI must also run the exllamav3 of this branch, with the extension built from source (refer to [Requirements](#requirements)). Settings in the TabbyAPI configuration that this model needs:

| Setting | Value |
|---|---|
| `backend` | `exllamav3` |
| `cache_mode` | `FP16` |
| `cuda_malloc_async` | `false` |
| `vision` | `true`, for image input |
| `autosplit_reserve` | Enough for the whole loop to be on one GPU. On the RTX 4090 + RTX 3090 with a cache of 163,840 tokens: `[2560, 1024]` |
| Draft model (optional) | `draft_mode: model`, `draft_num_tokens: 7`, `dynamic_draft: true`, and a `draft_gpu_split` that puts the draft model on the second GPU (`[0, 8]` on this system) |

#### Published conversions

Conversions made with the script below are on Hugging Face, in the repository [ratulrafsan/Inkling-2-Mini-EXL3](https://huggingface.co/ratulrafsan/Inkling-2-Mini-EXL3). They load only with this branch.

| Branch | Experts | Weight files | State |
|---|---|---|---|
| [`main`](https://huggingface.co/ratulrafsan/Inkling-2-Mini-EXL3/tree/main) | 6 bpw, last 4 layers 8 bpw | 21.7 GiB | The conversion all results in this section refer to, unless a table says otherwise |
| [`4.5bpw`](https://huggingface.co/ratulrafsan/Inkling-2-Mini-EXL3/tree/4.5bpw) | 4.5 bpw: 5 bpw for layers 6 to 15 and 20 to 21, 4 bpw for the others | 16.5 GiB | Smaller and faster, a little further from the source. Checked through TabbyAPI at a context of 262,144 tokens: short chats, images, a 251,619-token prompt |

```
hf download ratulrafsan/Inkling-2-Mini-EXL3 --local-dir Inkling-2-Mini-EXL3-6.0bpw
hf download ratulrafsan/Inkling-2-Mini-EXL3 --revision 4.5bpw --local-dir Inkling-2-Mini-EXL3-4.5bpw
```

Both include the vision tower and `mtp.safetensors` as stored in the source checkpoint. Refer to [Other bit rates](#other-bit-rates) for the quality and speed of each.

#### Convert the model

The source checkpoint has the experts only in NVFP4 (layers 0 to 19) and MXFP8 (layers 20 to 23). There is no full-precision release. The stock converter (`convert.py`) cannot run a looped model, so the conversion has its own script:

```
python util/inkling2/convert_exl3.py <source dir> <output dir> --work <work dir> --rows 250 --bits 6 --bits-mx 8
```

- The script quantizes the experts only (93% of the parameters). All other tensors stay as in the checkpoint.
- The calibration pass collects the Hessians of each expert over all three loop passes. There is no error propagation between layers.
- The conversion takes approximately 65 minutes on this system. The work directory needs approximately 60 GB. You can delete it after the conversion.
- The result at 6 bpw (8 bpw for the MXFP8 layers) is 23 GB on disk and uses 21.2 GiB of VRAM.

Quality against the source checkpoint over 20,480 tokens (`util/inkling2/quality_check.py`):

| Model | KL | Top-1 agreement | Perplexity |
|---|---|---|---|
| Source checkpoint (reference) | - | - | 8.4444 |
| Source checkpoint with bf16 activations (noise floor) | 0.00245 | 97.64% | 8.4478 |
| EXL3, 6 bpw | 0.00282 | 97.49% | 8.4452 |

##### Other bit rates

The same measure on ExLlamaV3's benchmark trace (`eval/qbench_prompts.py`: conversations from a fixed prompt set, sampled from the model itself; 25 conversations, 46,740 scored tokens) with the code of `eval/qbench` (`util/inkling2/qbench_trace.py`, `util/inkling2/qbench_i2.py`). The expert layers were quantized at 4 and at 5 bits from the same calibration data, and assembled with `util/inkling2/assemble_variant.py`. EXL3 has whole bit rates above 3.5, so 4.5 bpw is a mix: 5 bits for layers 6 to 15 and 20 to 21, 4 bits for the others.

| Experts | Weight files | KL to the source (mean) | KL above the noise floor | Perplexity | Decode step |
|---|---|---|---|---|---|
| Source checkpoint | - | - | - | 1.6276 | - |
| Source checkpoint with bf16 activations (noise floor) | - | 0.0017 | - | 1.6273 | - |
| 6 bpw, last 4 layers 8 bpw (published, branch `main`) | 21.7 GiB | 0.0020 | 0.0003 | 1.6261 | 11.2 ms (89 T/s) |
| 5 bpw | 17.9 GiB | 0.0032 | 0.0014 | 1.6305 | 10.7 ms (93 T/s) |
| 4.5 bpw (mix; published, branch `4.5bpw`) | 16.5 GiB | 0.0042 | 0.0025 | 1.6328 | 10.3 ms (97 T/s) |
| 4 bpw | 15.0 GiB | 0.0061 | 0.0044 | 1.6355 | 9.5 ms (105 T/s) |

- The decode step is from `util/inkling2/graph_profile.py` at 2K tokens of context, with a cache of 163,840 tokens on the RTX 4090 + RTX 3090. At 4 bpw the whole model and the cache are on the RTX 4090. The 4.5 bpw conversion was also run through TabbyAPI at a context of 262,144 tokens: 95 T/s on short chats (6 bpw: 83 T/s at that context, where part of its loop is on the second GPU) and a 251,619-token prompt without errors. The 4 and 5 bpw conversions were not run through TabbyAPI.
- The last four layers (MXFP8 in the source) matter little: with them at 8 bpw, the KL is 0.0031 at 5 bpw and 0.0056 at 4 bpw.
- For comparison, upstream pull request #621 reports 0.0064 at 4 bpw, 0.0100 at 3.5 bpw and 0.0156 at 3 bpw with the same prompt set and measure. Its trace was sampled on another system (reference perplexity 1.99), and its conversions also quantize the attention projections and the output head, so the numbers are close but not the same test.

> [!NOTE]
> The top-k expert routing of this model is discontinuous. Two runs that differ only by rounding give logits that differ by 2 to 3%. The "noise floor" line shows this effect. Do not compare outputs token by token without it.

#### Settings

| Variable | Value | Default |
|---|---|---|
| `EXL3_I2_GRAPH` | `0` runs each decode step operation by operation, without CUDA graphs. | `1` |
| `EXL3_I2_GRAPH_WIDTH` | The maximum number of tokens of one sequence in one forward pass that runs as a graph. | `8` |
| `EXL3_I2_KERNELS` | `0` uses the PyTorch forms in the graph and not the decode kernels. | `1` |
| `EXL3_I2_PREFILL_KERNEL` | `0` uses the PyTorch forms of attention and the block attention residual in prefill. | `1` |
| `EXL3_I2_FUSE` | `0` keeps the small PyTorch operations between the decode kernels (norms, casts, residuals, output gate). | `1` |
| `EXL3_I2_EMBED` | `gpu` puts the input embedding table on the first GPU and not in system RAM. | `cpu` |
| `EXL3_I2_PIPELINED_PREFILL` | `1` lets the generator run its pipelined prefill, where the GPUs of the split work at the same time. Refer to [Limits](#limits) before you set it. | `0` |
| `EXL3_I2_MOE` | `dense` runs the EXL3 experts in a slow reference loop. Use it to compare with `kernel`. | `kernel` |
| `EXL3_I2_ADT` | The activation data type: `fp16`, `bf16` or `fp32`. The decode kernels need `fp16`. | `fp16` |
| `EXL3_I2_NAN_CHECK` | `1` shows a line when a forward pass gives logits that are not finite. | `0` |
| `EXL3_I2_VISION_DEVICE` | Where the vision tower loads: `last` (the last GPU), `first` (the device the loader gives it), `cpu` or `cuda:N`. | `last` |
| `EXL3_I2_VISION_FP16_PATCHES` | Images with more patches than this run the tower's attention in fp16. | `12000` |
| `EXL3_I2_LOOPCD` | LoopCD mode: `off`, `hidden` or `logits`. Refer to [LoopCD](#loopcd). | `off` |
| `EXL3_I2_LOOPCD_REF` | The reference pass of LoopCD: `1` or `2`. | `1` |
| `EXL3_I2_LOOPCD_W` | The fixed strength of LoopCD. | `0.3`, or `0` when `EXL3_I2_LOOPCD_WMAX` is set |
| `EXL3_I2_LOOPCD_WMAX` | Logits mode: the cap of the adaptive strength. `0` is a fixed strength only. | `0` |
| `EXL3_I2_CVEC` | The control vector file (safetensors). Refer to [Control vector for Inkling-2](#control-vector-for-inkling-2). | Not set: off |
| `EXL3_I2_CVEC_MODE` | `project` or `add`. | The `glp.mode` of the file, or `project` |
| `EXL3_I2_CVEC_SCALE` | The scale of the control vector. | The `glp.alpha_default` of the file, or `1` |
| `EXL3_I2_CVEC_LAYERS` | `a-b`: use only `direction.a` to `direction.b` of the file. | All directions |
| `EXL3_I2_PASSES` | Runs only the first N loop passes. For tests: the reference logits of LoopCD are the logits of the model with as many passes as the reference pass. | `0` (all passes) |
| `EXL3_I2_DSPARK_WINDOW` | Limits the context of the drafter's full-attention layer to this number of tokens. `0` is all of it. Refer to the drafter results before you set it. | `0` |

#### Test results

All results are for the EXL3 conversion on the RTX 4090 + RTX 3090, one sequence.

Decode (`util/inkling2/graph_check.py`):

| Context (tokens) | Operation by operation | CUDA graph with kernels |
|---|---|---|
| 600 | 26 T/s | 91 T/s |
| 2,000 | 26 T/s | 90 T/s |
| 9,000 | 26 T/s | 88 T/s |
| 30,000 | 26 T/s | 81 T/s |
| 45,000 | 26 T/s | 77 T/s |

Forward passes with more than one token of one sequence (`util/inkling2/graph_spec_check.py`):

| Tokens | Operation by operation | Graph |
|---|---|---|
| 1 | 39.3 ms | 11.2 ms |
| 2 | 47.1 ms | 14.7 ms |
| 4 | 47.2 ms | 18.1 ms |
| 8 | 48.1 ms | 23.3 ms |

At 45K tokens of context a pass of 8 tokens takes 31.8 ms and a pass of one token 13.5 ms: the attention kernel reads K and V once for all tokens of a pass.

Prefill, one chunk of 2,048 tokens (`util/inkling2/component_bench.py`):

| Context (tokens) | PyTorch forms (`EXL3_I2_PREFILL_KERNEL=0`) | With the extension |
|---|---|---|
| 0 | 772 ms (2,650 T/s) | 586 ms (3,500 T/s) |
| 40,960 | 2,443 ms (840 T/s) | 1,064 ms (1,925 T/s) |

Perplexity over 20,480 tokens (`util/inkling2/ppl_check.py`) is 8.4455 with the extension and 8.4470 with the PyTorch forms. The KL between the two is 0.0015, below the noise floor in the table above.

Through tabbyAPI (cache of 163,840 tokens, greedy):

| Test | Result |
|---|---|
| Short chats | 88.7 T/s |
| 22K-token prompt | Prefill 2,863 T/s, decode 75 T/s, correct answer to a question about a sentence at 10% depth |
| 161,543-token prompt | Prefill 1,153 T/s (140 s), decode 48 T/s, correct answer to a question about a sentence at 10% depth |
| 161,543-token prompt, older build before the graph and kernel work | Prefill 183 T/s (883 s), decode 12.9 T/s |

With the DSpark draft model (`dynamic_draft: true`, confidence 0.4, 400 tokens, default reasoning effort), against 88.7 T/s without it:

| Text | Result |
|---|---|
| Code | 139 to 147 T/s |
| List | 107 T/s |
| Short math answer | 106 T/s |
| Prose | 92.6 T/s |
| Essay | 81.8 T/s |
| After a 161,543-token prompt | 34.6 T/s (48 T/s without the draft model) |

The draft model gains on text it predicts well and loses on free prose and at very long context. `util/inkling2/draft_trace.py` shows the cost of each round by draft length. The numbers above are from before the attention kernel read K and V once for each pass; the script bench at 45K tokens of context now gives 98 T/s on code and 68 T/s on essay text against 74 T/s without the draft model.

A context window for the drafter (`EXL3_I2_DSPARK_WINDOW`) is not a gain. At 45K tokens of context it takes 0.2 ms off a 4.3 ms round. On a prompt that quotes text from far back, the drafter gives 102 T/s with its full context and 70.5 T/s with a window of 4,096 or 1,024 tokens.

#### Images

The vision tower is a 23-layer vision transformer (303M parameters, 1.2 GiB in fp32). The image is not resized: it is padded to cells of 40 x 40 pixels, and each cell is one image token. An image of 1024 x 768 pixels is 520 tokens. An image with a side of more than 4,096 pixels (the limit of the model card) is scaled down first.

In the prompt, an image is a message of its own:

```
<|message_user|><|content_image|>{image tokens}<|end_message|><|message_user|><|content_text|>What is in this image?<|end_message|>
```

Python API:

```python
vision = Model.from_config(config, component = "vision")
vision.load()
image = vision.get_image_embeddings(tokenizer, PIL.Image.open("photo.png"))
prompt = effort + "<|message_user|><|content_image|>" + image.text_alias + "<|end_message|>" + question
text = generator.generate(prompt = prompt, embeddings = [image], encode_special_tokens = True, ...)
```

TabbyAPI concatenates the parts of a message into one text, which puts the image inside a text message. The TabbyAPI patch (refer to [Use with TabbyAPI](#use-with-tabbyapi)) gives the parts to the chat template as a list when the vision model names an image placeholder (`image_placeholder`), and replaces each placeholder in the rendered prompt with the alias of an image.

Checks (`util/inkling2/vision_check.py` against `util/inkling2/ref_dump_vision.py`, three test images):

| Check | Result |
|---|---|
| Preprocessing against the reference image processor | Identical (maximum difference 1.2e-7) |
| Tower output against the reference in fp32 | 1.5e-6 to 2.4e-6 relative error |
| The reference in bf16 against the reference in fp32 | 1.7e-2 to 2.5e-2 |
| Questions about test images (shapes and colours, text in an image, numbers at positions of a 3 x 4 board) | Correct, in the Python API and through TabbyAPI, with and without the draft model |

| Image | Image tokens | Tower time (RTX 3090) |
|---|---|---|
| 1024 x 768 | 520 | 0.21 s |
| 2048 x 1536 | 2,028 | 1.6 s |
| 4096 x 4096 | 10,609 | 5.0 s (fp16 attention; 55 s in fp32) |

Decode steps of text tokens stay CUDA graph replays in a sequence that has images (88 T/s). A second turn on the same image reuses the cache (83% of a 619-token prompt).

#### LoopCD

[LoopCD](https://arxiv.org/abs/2610.02185) ("Decoding Looped Transformers Better for (Almost) Free") is contrastive decoding for looped transformers, without training and without a second model. A looped model decodes only the state after its last pass. LoopCD also uses the state after an earlier pass (the reference) and moves the output away from it. [InferMatrixLab/vllm-rlt #101](https://github.com/InferMatrixLab/vllm-rlt/pull/101) implements it in vLLM for the Ouro models. This is an implementation for Inkling-2-Mini. It is off by default.

> [!NOTE]
> In the tests below LoopCD does not give a measurable gain for Inkling-2-Mini. The tests score one answer token and next-token prediction. They do not test generation with reasoning, where the paper reports its largest gains.

The two modes of the paper:

- **Hidden state (`hidden`).** The hidden state that goes into the layers after the loop becomes `h + w·(h − h_ref)`. `h` is the state after the last pass and `h_ref` is the state after the reference pass. This mode has no cost. The layers after the loop, and their caches, see the changed state for all tokens.
- **Logits (`logits`).** The layers after the loop and the output head also run on the state of the reference pass. The logits become `z + w·(z − z_ref)`. The strength `w` is fixed, or adaptive: `w = w_max·(1 − (p1 − p2))`, where `p1` and `p2` are the two largest probabilities of the unguided output. The adaptive strength is largest when the first two tokens have the same probability.

Ouro has no layers after its loop, so its reference logits are one more call of the output head. Inkling-2-Mini has 6 layers after the loop, and they have K/V caches and convolution state. In logits mode this implementation gives the reference readout its own cache layers. Thus, the reference logits are exactly the logits of the model with only the reference pass (or the first two passes) for the same tokens. `util/inkling2/loopcd_check.py` tests this against the model loaded with `EXL3_I2_PASSES`.

Set the mode in the environment of the process that loads the model:

```sh
EXL3_I2_LOOPCD=logits EXL3_I2_LOOPCD_REF=1 EXL3_I2_LOOPCD_W=0.3 python main.py        # fixed strength
EXL3_I2_LOOPCD=logits EXL3_I2_LOOPCD_WMAX=0.5 python main.py                          # adaptive strength
EXL3_I2_LOOPCD=hidden EXL3_I2_LOOPCD_W=0.3 python main.py
```

In Python, call `config.set_loopcd(mode, ref, strength, wmax)` before `Model.from_config`. On a loaded model, `model.set_loopcd(strength = ..., wmax = ...)` changes the strength from the next forward pass. It also changes between `off` and `hidden`. Logits mode is fixed when the model is created, because it adds cache layers. With `params["loopcd_export"] = True`, a forward pass in logits mode also returns the two sets of logits in `params["loopcd_logits"]`.

Quality, 6 bpw conversion (`util/inkling2/loopcd_eval.py`). The multiple-choice test has 4,172 questions (ARC-Challenge test, 1,172; MMLU test, 3,000 from 37 subjects; get them with `util/inkling2/loopcd_fetch_mc.py`). Each question is a chat turn with thinking effort 0, and the answer is the option letter with the highest logit. "Better" and "worse" count the questions whose answer changed to correct or to wrong. `p` is from a sign test on these two counts. The text test is top-1 accuracy of the next token on 20 rows of 1,024 tokens.

| Setting | Multiple choice | Better / worse | p | Text top-1 |
|---|---|---|---|---|
| Plain model | 81.64% | | | 52.13% |
| Reference pass 1 alone (the model with 1 pass) | 23.42% | | | 42.10% |
| Reference pass 2 alone (the model with 2 passes) | 35.31% | | | 46.23% |
| Logits, reference 1, fixed 0.1 | 81.76% | 16 / 11 | 0.44 | 52.19% |
| Logits, reference 1, fixed 0.3 | 81.90% | 37 / 26 | 0.21 | 51.48% |
| Logits, reference 1, fixed 0.5 | 81.86% | 48 / 39 | 0.39 | 50.23% |
| Logits, reference 1, fixed 1.0 | 81.90% | 74 / 63 | 0.39 | 44.53% |
| Logits, reference 1, adaptive, cap 0.5 | 81.88% | 41 / 31 | 0.29 | 51.14% |
| Logits, reference 1, adaptive, cap 1.0 | 81.86% | 54 / 45 | 0.42 | 48.35% |
| Logits, reference 2, fixed 0.3 | 81.71% | 24 / 21 | 0.77 | 51.90% |
| Logits, reference 2, fixed 0.5 | 81.86% | 35 / 26 | 0.31 | 50.89% |
| Logits, reference 2, adaptive, cap 0.5 | 81.81% | 30 / 23 | 0.41 | 51.49% |
| Hidden, reference 1, 0.25 | 81.78% | 11 / 5 | 0.21 | 52.09% |
| Hidden, reference 1, 0.5 | 81.74% | 15 / 11 | 0.56 | 51.99% |
| Hidden, reference 1, 1.0 | 81.71% | 20 / 17 | 0.74 | 51.87% |
| Hidden, reference 2, 0.5 | 81.78% | 21 / 15 | 0.41 | 52.02% |
| Hidden, reference 2, 1.0 | 81.81% | 29 / 22 | 0.40 | 51.80% |

- All settings are 0.0 to 0.3 points above the plain model on the questions. No difference is significant: the smallest p is 0.21. The paper reports +0.3 to +1.6 points for its models on its multiple-choice tests.
- Next-token accuracy on text decreases when the strength increases. In logits mode the decrease is large from a strength of 0.5.
- The early passes are weak alone. With 1 pass the model answers the questions at the level of a guess, and with 2 passes it has 35%. Thus, a lower number of passes with LoopCD, which the paper shows for other models, is not an option for this model.

Cost (`util/inkling2/mode_speed.py`: 8K tokens of context, cache of 16K tokens; TabbyAPI: cache of 163,840 tokens, short chats, greedy):

| Setting | Prefill | Decode | TabbyAPI decode | VRAM |
|---|---|---|---|---|
| Plain model | 3,293 T/s | 11.33 ms (88.2 T/s) | 89.3 T/s | |
| Hidden | 3,290 T/s | 11.34 ms (88.2 T/s) | 88.8 T/s | The same |
| Logits, fixed or adaptive | 2,712 T/s (−18%) | 12.93 ms (77.3 T/s, −12%) | 77.7 T/s (−13%) | +0.7 GiB at a cache of 163,840 tokens, on the GPU of the last layer |

In logits mode each layer after the loop runs two times in a row: on the state of the last pass, then on the reference state. The two readouts thus go to the next GPU together. With the reference readout after the full readout of the last pass, a GPU split inside these layers sent the state across two more times in each forward pass, and prefill was 28% slower than the plain model.

#### Control vector for Inkling-2

A control vector for Inkling-2-Mini can have one direction for each layer application ("execution step"): the loop layers run three times with the same weights, but their activations in the three passes are different. [msuiche/Inkling-2-Mini-abliterated-cyber-GLP-48-L1-48](https://huggingface.co/msuiche/Inkling-2-Mini-abliterated-cyber-GLP-48-L1-48) is a file of this type: a [GLP](https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md) file with 48 directions for the 24 layers, for projection.

The patch in [Control vectors](#control-vectors) cannot apply such a file:

- It changes `TransformerBlock`. This model has its own layer module.
- It counts directions by stored layer, from 0. In this file, `direction.N` is for layer application `N − 1`, and there are 48 of them.
- It does not read the GLP metadata (mode, default strength).
- With `EXL3_CVEC` set, it applies the first directions of the file to the three layers of the DSpark draft model, which are `TransformerBlock` modules.

Thus, the Inkling-2 architecture has its own reader, with its own variables (`EXL3_I2_CVEC`, not `EXL3_CVEC`). It applies the vector to the output of each layer application, for all token positions:

- **Projection (`project`):** `h = h − s·(h·d)·d`. `d` is the unit direction of the layer application and `s` is the scale. `s = 1` removes the component along `d`.
- **Addition (`add`):** `h = h + s·v`.

The changes that this needed, beyond the existing patch:

- **A hook in the layer module of this model.** `Inkling2Layer.forward` applies the vector to the tensor that each layer application returns, in the operation-by-operation path and in the decode graph. The block states, the state that each loop pass starts from and the messages to the next pass all come from the changed tensor. This is the effect of a forward hook on the decoder layers of the transformers model, which is how the file was made.
- **Numbers of layer applications.** The reader maps `direction.N` to application `N − 1` of the forward order: 6 before the loop, 3 × 12 in the loop, 6 after the loop. The file agrees with this numbering. Directions that are next to each other have a cosine of 0.65 to 0.94, except at 7 places where it is 0.17 to 0.32. With this numbering, and only with it, these 7 places are the layer applications where the model starts a new block of 6 layers.
- **The metadata of the file.** `util/cvec_gguf_to_safetensors.py` wrote only the tensors. It now also writes the `glp.*` and `controlvector.*` metadata. The reader takes the mode (`glp.mode`), the default scale (`glp.alpha_default`) and the structure (`glp.structure = per-execution-step`) from it, and follows the rules of the GLP specification for what a reader must not accept.
- **Arithmetic and scale.** The projection is computed in fp32 on the fp16 state. The scale is a tensor on the GPU of the layer, so a captured decode graph follows `set_scale` without a new capture. The setter updates the tensor in inference mode, because the tensor is made during a forward pass.
- **Separate variables** (`EXL3_I2_CVEC*`), so that the patch for `TransformerBlock` does not also read the file.
- **LoopCD.** In logits mode, the second application of a layer after the loop uses the direction of the first.
- **Tools.** `util/inkling2/cvec_check.py` tests the hook in the port. `util/inkling2/cvec_ref_dump.py` runs the transformers reference with the forward hooks and `util/inkling2/cvec_ref_compare.py` compares the port with it. `util/inkling2/loopcd_eval.py --mode plain` and `util/inkling2/mode_speed.py` measure quality and speed with the vector on.

A file with one direction for each stored layer (`direction.k` after layer `k`, as llama.cpp applies it) also works: each direction then applies to all applications of its layer. The reader does not accept a hook point other than the layer output, more than one direction for each layer (GLP rank above 1), or additive use of a file that is marked for projection.

Convert the GGUF file one time. The converter keeps the GLP metadata, and the reader takes the mode, the default scale and the numbering from it:

```sh
python -I util/cvec_gguf_to_safetensors.py glp.inkling2mini-GLP-48-L1-48.gguf inkling2_glp48.safetensors
EXL3_I2_CVEC=/path/to/inkling2_glp48.safetensors python main.py
```

The log shows a line of this type when the vector is active:

```
 -- Inkling2 control vector: mode = project, scale = 0.25, directions 1..48 (one per layer application), 48 layer applications steered, from ...
```

`EXL3_I2_CVEC_SCALE` sets a different scale. In Python, `config.cvec.set_scale(s)` changes it on a loaded model. The variables are in the settings table of this section.

Test results with the file above (6 bpw conversion, except where noted):

| Test | Result |
|---|---|
| `util/inkling2/cvec_check.py` (4 bpw, one GPU) | Scale 0 gives the logits of the plain model bit for bit, uncached, cached and as graph replays. At scale 1 the component along the direction is removed at each of the 48 layer applications (mean `\|h·d\|` 5.87 before, 0.001 after), and the next application receives the changed state. At scale 0.25 the component is 0.750 to 0.751 of its value before. Graph replays and operation-by-operation steps agree as for the plain model. |
| Against the transformers reference with a forward hook on each decoder layer (`util/inkling2/cvec_ref_compare.py`, 5 prompts) | The change of the logits from scale 0 has a cosine of 0.95 to 0.99 with the change in the reference, and 0.78 to 1.04 of its size. The top token changes at the same share of the positions (15 to 26% at scale 0.25, 21 to 63% at scale 1). The rest is the routing noise between two implementations of this model. |
| Multiple choice, 4,172 questions (the test of [LoopCD](#loopcd)) | Plain 81.64%. Scale 0.1: 81.45%. Scale 0.25: 78.88% (MMLU part: 76.80% to 73.57%). |
| Text, top-1 accuracy of the next token | Plain 52.13%. Scale 0.1: 52.15%. Scale 0.25: 51.67%. |
| Speed (`util/inkling2/mode_speed.py`) | Prefill does not change (3,288 T/s against 3,293 T/s). Decode is 11.71 ms against 11.33 ms for each token (−3%). TabbyAPI, short chats: 86.2 T/s against 89.3 T/s. |
| First 12 tokens of the reply, thinking effort 0, greedy: 8 requests that the plain model refuses, 4 harmless requests | The 8 replies start with a refusal at scales 0, 0.1, 0.25 (the default of the file), 0.35 and 0.5. At scale 1, 5 of 8 start with a refusal. The 4 harmless requests get an answer at all scales, and the text is coherent at all scales. |

The model card of the file reports a larger effect at scale 0.25 than these tests show: 67.2% against 80.8% on an MMLU subset, and no refusals. This port and the transformers reference agree on the effect of the vector, so the difference is not in this reader. Its cause is not known. The tests here do not include generation with reasoning, and they do not use the test prompts of the model card.

Limits:

- The vector applies to all requests of the process.
- The DSpark draft model is not steered. The steered model does the verification of all draft tokens, so the output is the output of the steered model. The draft model reads the steered states, which it was not trained on: fewer draft tokens can be accepted. This is not measured.
- The projection is a few small PyTorch operations for each layer application in the decode graph. A kernel can remove most of the 3% decode cost.
- A control vector is specific to one model. Do not use a vector from a different model.

> [!NOTE]
> A refusal vector removes a safety behavior of the model. You are responsible for the output of the model.

#### How the patch works

- `exllamav3/architecture/inkling2_vision.py` has the vision tower in plain PyTorch, in fp32. The text model's embedding module takes the rows of image tokens from the image embedding as they are, after its own norm, as the reference does.
- `exllamav3/modules/inkling2.py` has the text model in plain PyTorch. It is the reference, and it runs prefill.
- `exllamav3/architecture/inkling2_loop.py` has the modules, the cache state and the graph code. The loop uses the layer map of exllamav3: the looped layers occur three times in the forward order, and each occurrence has its own cache layers.
- The global layers keep K/V in the paged cache. The sliding layers keep the last window in a buffer for each sequence. All layers keep the last inputs of their convolutions in a ring.
- A decode step is captured as one CUDA graph for each GPU. All cache indexes are computed on the GPU from two integers, so one captured graph is correct for all positions.
- `exllamav3/exllamav3_ext/inkling2.cu` has six kernels that replace groups of small PyTorch operations in the graph: convolution (with the residual and the fp16 cast), block attention residual, message gate, RMS norm with gain and cast, Q/K/V post-processing and attention (with the output gate). A decode step is approximately 1,170 kernel launches.
- Prefill attention has three steps for each chunk of queries, in the same file. `i2_attn_qk` computes the logits with cuBLAS from the fp16 values of Q and K, with fp32 accumulation. `i2_attn_soft` applies the scale, the relative bias, the masks and the exponentials in one kernel. `i2_attn_pv` multiplies the weights with V; each weight goes in as two fp16 terms, which keeps approximately 22 bits. K and V are read in place from the cache. The results have the accuracy of the fp32 PyTorch form.

- LoopCD keeps the state that the reference pass leaves (the hidden state and the block states) when the next pass starts. Hidden mode mixes it into the hidden state at the first layer after the loop. Logits mode adds a second application of each layer after the loop to the layer map (instance 1 of the layer, with its own cache layers), swaps the two states before each application, and gives both final states to the head, which computes both sets of logits with one product. The strength is a tensor on the GPU, so a captured decode graph follows a change of it.
- The control vector is applied to the tensor that each layer application returns. The directions and the scale are tensors on the GPU of the layer.

The checks in `util/inkling2/` compare the port with the transformers reference (`ref_dump.py`, `port_check.py`), cached decode with an uncached pass (`cache_check.py`) and graph decode with operation-by-operation decode (`graph_check.py`, `graph_seq_check.py`, `graph_spec_check.py`).

#### Limits

- The vision tower's rotary frequencies are exact fp32 values. The transformers reference rounds them to bf16 when the model is loaded in bf16; the difference at the tower's output is 3e-4.
- Video input is not implemented (the tower takes single images).
- LoopCD is tested with single-token scoring and next-token prediction only. It is not tested with a generation benchmark, with reasoning, or with the draft model. The draft model is not guided, so fewer of its tokens can be accepted.
- LoopCD and the control vector are settings of the process. They cannot be set for each request.
- Graph decode is for one sequence. Batches, prefill and forward passes wider than `EXL3_I2_GRAPH_WIDTH` run operation by operation.
- **Pipelined prefill is off.** The generator's prefill pipeline engages when the split is a straight route (first GPU, then second) and then runs the GPUs at the same time. On the development system, the computer power-cycled approximately 2 seconds into a 161K-token prompt, three times out of three, with nothing in the logs; with the pipeline off the same split runs the same prompt. The cause is not established (power delivery, or a fault in the concurrent path), and the pipelined path is not validated against the sequential one for this model. `EXL3_I2_PIPELINED_PREFILL=1` turns it on.
- Prefill slows down with the context. At 40K tokens of context, approximately half of the prefill time is attention on the global layers.
- The checkpoint of the cache state is approximately 162 MiB, because it holds the attention window of all sliding layer applications.
- A few experts that the model uses rarely saw very little calibration data (a minimum of 1 token in layer 22).
- On one 44K-token test prompt, graph decode and operation-by-operation decode gave different answers. All step-by-step comparisons agree, so this is the routing effect in the note above, but it is not proven for all prompts. `EXL3_I2_GRAPH=0` is the switch to compare.

#### TODO

1. **MTP draft model: lower the cost of a round.** A round costs approximately 4.1 ms before the verification pass. About half is memory traffic: the output head for the 7 draft rows (926 MB) and the Markov table 7 times (103 MB each). A sparse head over the frequent tokens (exllamav3 has one for drafters that use its `Linear` head) can remove most of it. The draft block and the Markov chain also still run operation by operation. On essay text, 35% of the rounds end with no draft that passes the confidence cut, and each of those costs 16 ms for one token against 11 ms. Sitting the drafter out after failed rounds was tested and gave no gain.
2. **Prefill attention.** The logits of a chunk of queries are still written to memory and read back (two cuBLAS products with a kernel between them). One kernel with tensor cores that keeps the logits in shared memory can remove this traffic. The experts are the largest part of prefill at short context.
3. **Decode.** Of 11.2 ms for each token at 2K context, approximately 7.6 ms is memory traffic: the cuBLAS projections (2.9 ms), the expert kernels (3.5 ms) and the output head (1.1 ms). The own kernels take approximately 2.5 ms. At very long context the K/V of the 8 global layer applications dominate (5.3 GB read per token at 161K). `util/inkling2/graph_profile.py` shows the list.
4. **Graph steps wider than 8 tokens**, for the n-gram draft verification of tabbyAPI (65 tokens).
5. **Batched decode** as graphs.
6. **Quantized cache modes** for the global layers.
7. **Native NVFP4 expert kernels**, to run the source checkpoint without a conversion.
8. **Conversion with the stock converter**, which needs support for looped models, and more calibration data for the rare experts.

### SenseNova Loop

> [!NOTE]
> This patch is not my work. It is a port of [turboderp-org/exllamav3#609](https://github.com/turboderp-org/exllamav3/pull/609) and of the branch [`sensenova-loop-mtp`](https://github.com/Maxrio9/exllamav3/tree/sensenova-loop-mtp), both by [Maxrio9](https://github.com/Maxrio9). The architecture, the loop-aware conversion, the fused router kernel, the tests and the quantized models are all theirs. When the pull request is in an ExLlamaV3 release, this patch goes away.

#### Overview

[SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) is a 9.7B-parameter model with text and image input. It is a looped transformer:

- The model has 32 layers of weights. Layers 4 to 27 run two times for each token, so a token goes through 56 layer applications.
- The layers follow the Qwen 3.5 pattern: 3 gated-delta-net layers, then 1 full-attention layer.
- Inside the loop, the residual runs as two streams. An FP32 router (mHC, log-space Sinkhorn) mixes the streams before and after each attention and MLP block.

The patch adds the architecture `SenseNovaLoopForConditionalGeneration`:

- **Text model.** The loop uses `layer_map`, with one cache slot for each layer application.
- **Vision tower.** The model uses the Qwen 3.5 vision tower without changes.
- **MTP head.** If the model directory has `mtp.*` tensors, the head is available as the `mtp` component and works as a draft model.
- **Router kernel.** One Triton program computes the router for a token. `EXL3_LOOP_ROUTER_TRITON=0` selects the plain PyTorch router.
- **Conversion.** The converter collects calibration statistics from the two visits of each shared layer. It needs one GPU and online calibration.

For the design and the validation against the BF16 model, refer to the pull request.

#### Differences from the source branches

- **Base.** The source branches are on the upstream `dev` branch. This port is on `v1.6.7` plus the patches of this fork. The pull request applied without changes.
- **GEMM autotune.** The branch `sensenova-loop-mtp` makes the EXL3 GEMM kernels use one tuned configuration for 1 to 8 rows. This port does not include that change. On the RTX 4090, the speed and the greedy output with MTP were the same with the change and without it, and the change has an effect on the kernel tuning of all other models.

#### Requirements

- A quantized model with the vision tower and the MTP head. The tests used the 6.0 bpw model from [Maxrio/SenseNova-Loop-9B-exl3](https://huggingface.co/Maxrio/SenseNova-Loop-9B-exl3) (each bit rate is on its own branch).
- Triton, for the router kernel. Without Triton the model uses the PyTorch router, which is much slower for decode.
- No change to the extension. The prebuilt wheel works.

#### Use with TabbyAPI

TabbyAPI at commit `4728158` needs no changes. These settings are the important ones (they follow the [recipes](https://github.com/Maxrio9/sensenova-loop-9b-exl3-recipes) of the model author):

```yaml
model:
  model_name: SenseNova-Loop-9B-exl3-6.0bpw
  max_seq_len: 131072
  cache_size: 131072
  cache_mode: FP16
  recurrent_state_fp32: true
  warmup: true
  tool_format: qwen3_coder
  vision: true
draft_model:
  draft_mode: mtp
  draft_num_tokens: 4
  hybrid_draft_tokens: 7
  sparse_draft_head: 16384
```

To switch thinking on or off for a request, send `chat_template_kwargs: {"enable_thinking": true}` or `false`.

#### Test results

These are the results on the RTX 4090 with the 6.0 bpw model. With a cache of 131,072 tokens, the vision tower and the MTP head, the model uses 15.7 GiB of VRAM. The RTX 3090 is not used.

Decode speed in T/s, greedy, in the Python API (`util/sensenova_loop/mtp_check.py`):

| Prompt | No draft model | MTP draft model | Ratio |
|---|---:|---:|---:|
| Code | 60.0 | 150.7 | 2.51 |
| Math, with thinking | 59.8 | 147.0 | 2.46 |
| List | 59.8 | 119.6 | 2.00 |
| Prose | 59.9 | 91.2 | 1.52 |
| Edit of a code block | 60.3 | 119.1 | 1.97 |

- **Same output.** For all five prompts, the greedy output with the MTP draft model was the same as the output without it, token for token.
- **Prefill.** Prefill was 2,240 T/s for a prompt of 6,034 tokens in the Python API and 3,141 T/s for a prompt of 39,918 tokens in TabbyAPI.
- **Cache.** For three greedy answers (176, 200 and 66 tokens), one full forward pass gave the same tokens as the generator at all positions but one.
- **Router kernel.** On 600 tokens, the logits with the Triton router and with the PyTorch router agree on the top token at 99.67% of the positions (KL 0.00037). The pull request reports the same size for the difference between two BF16 runs.
- **TabbyAPI.** These tests gave correct results: an answer with thinking off, an answer with thinking on (the reasoning arrives in `reasoning_content`), a tool call, the answer after the tool result, a question about an image, and a code planted at 10% of a 39,918-token prompt. Decode in TabbyAPI was 151 T/s for code and 92 T/s for prose.
- **Tests of the pull request.** The CPU tests pass (32 passed, 4 skipped because they need local model paths). The GPU tests of the router kernel pass with Triton 3.6 (37 passed).

The first run of `mtp_check.py` after the port gave a lower gain (1.55 on average). The two runs after it gave the numbers in the table. I did not find the cause.

#### Check scripts

The scripts are in `util/sensenova_loop/`:

| Script | Function |
|---|---|
| `loop_check.py` | Compares the two routers, generates text, compares the generator with a full forward pass, and tests cache reuse, a long prompt and a batch. |
| `mtp_check.py` | Compares greedy text and decode speed with the MTP draft model and without it. |
| `tabby_test.py` | Tests a running TabbyAPI: thinking, a tool call, an image, speed and a long prompt. |

#### Limits

- **No conversion of my own.** I did not convert the model and did not compare this port with the BF16 model. For the quality of the quantized models, refer to the pull request and the model card.
- **One GPU.** The tests ran on one GPU. A split over two GPUs was not tested.
- **Context.** The longest tested prompt has 39,918 tokens. Quantized cache modes were not tested.
- **Control vectors.** The generic control vector patch does not reach the layers in the loop, because these layers have their own forward function.

### Control vectors

#### Overview

A control vector changes the residual stream of the model during inference. It does not change the model weights. This patch applies control vectors in the llama.cpp format. It is a port of the `--control-vector` function of llama.cpp. It also includes the projection mode of [Cudecnik/Qwen3.8-Flash-Next-refusal-projection](https://huggingface.co/Cudecnik/Qwen3.8-Flash-Next-refusal-projection).

The patch has two modes:

- **Projection (`project`).** The patch removes one direction from the residual stream: `h = h - s·(h·v)·v`. `v` is the unit vector for the layer. `s` is the scale.
- **Addition (`add`).** The patch adds the vector to the residual stream: `h = h + s·v`. This is the standard llama.cpp mode.

The patch is off by default. It has no effect until you set `EXL3_CVEC`.

This patch is for models that use `TransformerBlock`. Inkling-2-Mini has its own reader, for vectors with one direction for each layer application: refer to [Control vector for Inkling-2](#control-vector-for-inkling-2).

#### Prepare the vector file

llama.cpp keeps control vectors in GGUF files. This patch reads safetensors files. Convert the GGUF file one time:

```sh
python -I util/cvec_gguf_to_safetensors.py refusal-projection.gguf refusal-projection.safetensors
```

The script only reads the GGUF file. It does not run code from the file. The script shows the number of directions, the layer range and the norm of the vectors.

#### Settings

| Variable | Value | Default |
|---|---|---|
| `EXL3_CVEC` | The path to the safetensors file | Not set. The patch is off. |
| `EXL3_CVEC_MODE` | `project` or `add` | `project` |
| `EXL3_CVEC_SCALE` | The scale `s` | `1.0` |
| `EXL3_CVEC_LAYERS` | The layer range `a-b`. The range includes `a` and `b`. | All layers in the file |

The layer numbers are the same as in llama.cpp. The tensor `direction.k` applies after decoder layer `k`. The first layer is layer 0. llama.cpp does not keep a direction for layer 0.

#### Procedure

1. Set the variables in the environment of the process that loads the model. This process can be TabbyAPI, an example script or an eval script.
2. Start the process.
3. Make sure that the log shows a line of this type:

   ```
    -- Control vector: mode = project, scale = 1.0, layers 4..44 (41 directions) from ...
   ```

4. If the log does not show this line, the vector is not active. Make sure that `EXL3_CVEC` is set and that the patch is installed.

Example for Qwen3.8-Flash-Next with TabbyAPI:

```sh
EXL3_CVEC=/path/to/refusal-projection.safetensors EXL3_CVEC_MODE=project EXL3_CVEC_LAYERS=4-44 python main.py
```

To use the stock model again, remove `EXL3_CVEC` from the environment and start the process again.

#### How the patch works

The patch changes `TransformerBlock.forward` in `exllamav3/modules/transformer.py`. At the end of each decoder block, the patch applies the vector to the residual stream. This is the same point as `build_cvec` in llama.cpp.

- Models with hyper-connections keep the residual as parallel streams. The patch applies the vector to each stream.
- The patch does not apply the vector to the MTP head. The steered model does the verification of all draft tokens. Thus, the output is the output of the steered model.
- During decode, `GatedResidual` gives the next site a weighted copy of the streams. When the vector changes the streams, this copy becomes incorrect. The patch removes the copy. The next site then calculates the weights again.
- The projection changes the streams in place. It does not use more VRAM during prefill.

#### Test results

Model: Qwen3.8-Flash-Next, 4.05 bpw EXL3. Vector: refusal projection, layers 4 to 44, scale 1.0.

| Test | Stock model | With the vector |
|---|---|---|
| Refusals, 6 refusal test prompts | 6 | 0 |
| Answers, 2 harmless control prompts | 2 | 2 |
| Mean KL divergence from stock, 8,192 tokens of text | 0.017 (noise) | 0.077 |

The noise value is the KL divergence between two runs of the stock model. Dynamic expert placement causes this difference. A unit test also compares the decode path with a reference calculation. The two results are the same.

#### Performance

The test uses the hardware in [Hardware and constraints](#hardware-and-constraints), with TabbyAPI. The settings are greedy decode and thinking off, with MTP and hybrid drafting. Each run uses a new server. There are 2 runs of each condition.

| Workload | Stock (2 runs) | With the vector (2 runs) | Difference |
|---|---|---|---|
| Cold 14.9K-token prompt, prefill | 1,760 / 1,701 T/s | 1,703 / 1,695 T/s | −1.8%. The two stock runs differ by 3.4%. |
| Decode, first run (warm-up) | 62.3 / 61.6 T/s | 60.1 / 60.8 T/s | −2.4% (approximately 1 T/s) |
| Decode, mean of runs 2 to 4 | 78.7 / 77.0 T/s | 78.6 / 78.3 T/s | None (+0.7%) |
| Whole-file edit, second run | 106.9 / 103.3 T/s | 102.7 / 102.1 T/s | −2.6%. The two stock runs differ by 3.4%. |
| Accepted MTP draft tokens (decode / edit) | 183–188 / 824–829 | 183–188 / 829 | The same |
| Edit grade | Exact, 4 of 4 | Exact, 4 of 4 | The same |

No noticeable performance difference occurs. All differences are approximately equal to the difference between the two stock runs. If a cost occurs, it is 2–3% or less. Two runs of each condition cannot show a smaller difference.

Compare this result with the llama.cpp implementation of Cudecnik:

| Item | Cudecnik, llama.cpp (source README) | This port, exllamav3 |
|---|---|---|
| Decode speed | 93% of stock (approximately 7% less) | No noticeable difference |
| Extra VRAM | Not given in the README. The default projection makes a temporary tensor of the full residual size at each layer: approximately 42 MB for a 1,024-token batch. | Approximately 420 KB of vectors on each GPU. The projection changes the streams in place. |

On this system, the 3090 has no free VRAM at 215,040 tokens of context. The server with the vector loads at 215,040 tokens with the same split. Thus, the extra VRAM is not important. This test did not measure the VRAM directly.

The two implementations add approximately the same small GPU operations for each layer. On this system, each decode step also waits for the CPU experts, and MTP with hybrid drafting gives more than one token for each forward pass. Thus, the cost of the operations is a smaller part of the step time. This explanation is an estimate. It is not a profiler measurement. On a system with all experts in VRAM, the cost can be larger.

The test prompts are neutral. On refusal test prompts, MTP acceptance can decrease, because the MTP head is not steered. This test did not measure that effect.

#### Limits

- The vector applies to all requests that the process runs. You cannot set a vector for each request.
- Tensor-parallel mode is not tested.
- Only Qwen3.8-Flash-Next is tested. The patch applies to blocks with keys of the type `...layers.N`. Other models can work, but they are not tested.
- A control vector is specific to one model. Do not use a vector from a different model.

> [!NOTE]
> A refusal vector removes a safety behavior of the model. You are responsible for the output of the model.

<p align="center">
  <img src="doc/qb_kld.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

## Installation

Start by making sure you have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed (CUDA 12.4 or later) since the Torch dependency is not automatically handled by `pip`. Then pick a method below:

### Prebuilt wheel · recommended

Pick a wheel from the [releases page](https://github.com/turboderp-org/exllamav3/releases), then e.g.:

```sh
pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.8.0-cp313-cp313-linux_x86_64.whl
```

### Install from PyPI

```sh
pip install exllamav3
```
Note that the PyPI package does not contain a prebuilt extension and requires the CUDA toolkit and build prerequisites (i.e. VS Build Tools on Windows, gcc on Linux, `python-dev` headers etc.).

### Build from source

<details>
<summary>Source installation with uv or pip</summary>


`exllamav3` declares a minimum `torch` version (>= 2.6.0) and CUDA version (>= 12.4), but beyond that the user is free to select a version of `torch` that is compatible with their environment.

`torch` can be installed in three ways (from least to most effort):
1. **with `uv`, setting only `--extra cuXXX`** installs `torch` automatically with the specified CUDA version, `torch` version is selected by `uv` from compatible versions in the specific index associated with the chosen CUDA version (options 1 and 2)
2. **with `uv`, creating a thin project that depends on `exllamav3[cuXXX]` and pins a specific `torch` version** — like (1) but `torch` is pinned in the thin project's `pyproject.toml`, see [pinning a specific PyTorch version (optional)](#pinning-a-specific-pytorch-version-optional) for details
3. Manually with `uv pip` or `pip` (options 3 and 4)

The flavor extras (`--extra`) are `cu124`, `cu126`, `cu128`, `cu129`, `cu130`, and `cu132` — pick the one matching your installed CUDA build. Both `uv sync` and `pip install .` build the package in an isolated environment where your `torch` is not visible, so they install the extension sources and compile them at first import (JIT, a few minutes once per torch version). For a precompiled install run `pip install --no-build-isolation .` in an environment that already has `torch`, or use the release wheels. Selecting a flavor installs the matching CUDA build of `torch`.

**Option 1 — Working in the cloned repo directly (`uv sync`):**

```sh
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3
# (Optional) switch to dev branch for latest in-progress features
git checkout dev

uv venv
uv sync --extra cu130
# add --extra examples and/or --extra eval for those extra dependencies
```

**Option 2 — Using `exllamav3` as a dependency from another project (`uv add`):**

```sh
# `uv add` works inside an existing project (a directory with a pyproject.toml).
# `uv init` creates one if you're starting a new project, if integrating into
# an existing project skip `uv init`.
uv init my-project
cd my-project

# local checkout
uv add 'path/to/exllamav3[cu130]'               # non-editable
uv add 'path/to/exllamav3[cu130]' --editable    # editable

# straight from GitHub
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]'                 # default branch
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]' --branch dev    # specific branch
```

**Option 3 — Bring your own `torch` and let `uv` pick the backend automatically:**

```sh
uv venv            # or: uv venv --python-preference only-managed
source .venv/bin/activate
uv pip install torch --torch-backend=auto
uv pip install .
```

`--torch-backend=auto` inspects your system and installs the matching PyTorch CUDA build; see [Automatic backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection).

**Option 4 — With `pip`:**

On Windows, you also need the `triton-windows` package (declared as a dependency in `pyproject.toml`); the attention, cache and recurrent kernels are Triton and ExLlamaV3 does not import without it.

```sh
# install a CUDA-enabled torch first so it matches your setup, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install .
```

</details>

<details>
<summary>Pinning a specific PyTorch version (optional)</summary>

#### Pinning a specific PyTorch version (optional)

The flavor extra picks the *index*, but by default torch resolves to the latest version on that
index that satisfies `>=2.6.0`. To pin a specific torch version while developing on `exllamav3`,
create a **"thin" project** that consumes your local checkout as an editable install and declares
the exact `torch` version itself. This keeps the pin out of the `exllamav3` pyproject, so
you can change the torch version freely without touching the repo.

```
my-exllamav3-dev/          # thin project (uv init)
├── pyproject.toml
└── src/                  # package sources (auto-generated)
```

In `pyproject.toml`:

```toml
[project]
name = "my-exllamav3-dev"
version = "0.1.0"
description = "Dev environment for exllamav3"
requires-python = ">=3.10.11"
dependencies = [
    "exllamav3[cu130]",   # select correct CUDA version
    "torch==2.13.0",      # pin the exact torch version you need
]

[tool.uv.sources]
exllamav3 = { path = "../exllamav3", editable = true }
```

Adjust `../exllamav3` to point at your local checkout, then a plain `uv sync` sets up an
environment with the correct PyTorch index (routed via the `cuXXX` extra),
the pinned version of `torch` from that index (as long as it exists), and an editable install of `exllamav3` so code
changes apply immediately. Switch CUDA flavors by changing the extra (`exllamav3[cu124]`,
`exllamav3[cu128]`, …) and/or the torch pin in the thin project.

Or, if you're installing torch manually with `uv pip install torch` (e.g. as in Option 3 above),
specify the version directly, e.g. `uv pip install "torch==2.11.0" --torch-backend=auto`.

</details>

After installing with one of the options above, you should be able to run the conversion, eval and 
example scripts from the main repo directory, e.g., `uv run python convert.py -i ...` or, for manual
installations once the venv is active, `python convert.py -i ...`

**Build environment variables**

- `MAX_JOBS`: by default ninja may launch too many processes and run out of system memory for 
compilation. Set this to a reasonable value like 4 in that case.
- `EXLLAMA_NOCOMPILE`: set to install the library without compiling the C++/CUDA extension. Torch
will build/load it at runtime instead.
- `EXLLAMA_EXT_LINEINFO`, `EXLLAMA_EXT_COMPRESS`: see [doc/env_vars.md](doc/env_vars.md).

## Examples

A number of example scripts are provided to showcase the features of the backend and generator. 
For instance, a versatile CLI chatbot:

<p align="center">
  <img src="doc/chatpy.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

```sh
python examples/chat.py -m <input_dir> -mode <prompt_mode>

# Wealth of options
python examples/chat.py -h
```

## Architecture support

| Model family                                      | HF architecture                                                               | Multimodal | Notes                 |
|---------------------------------------------------|-------------------------------------------------------------------------------|:----------:|-----------------------|
| **AFM**                                           | `ArceeForCausalLM`                                                            |            |                       |
| **AfMoE**                                         | `AfmoeForCausalLM`                                                            |            |                       |
| **Apertus**                                       | `ApertursForCausalLM`                                                         |            |                       |
| **Command-R** etc.                                | `CohereForCausalLM`                                                           |            |                       |
| **Command-A**, **Command-R+** etc.                | `Cohere2ForCausalLM`                                                          |            |                       |
| **DeciLM**, **Nemotron**                          | `DeciLMForCausalLM`                                                           |            |                       |
| **Deepseek V3**                                   | `DeepseekV3ForCausalLM`                                                       |            |                       |
| **Deepseek V4**                                   | `DeepseekV4ForCausalLM`                                                       |     ✓     |                       |
| **dots.llm1**                                     | `Dots1ForCausalLM`                                                            |            |                       |
| **ERNIE 4.5**                                     | `Ernie4_5_ForCausalLM`<br>`Ernie4_5_MoeForCausalLM`                           |            |                       |
| **EXAONE 4.0**                                    | `Exaone4ForCausalLM`                                                          |            |                       |
| **Gemma 2**                                       | `Gemma2ForCausalLM`                                                           |            |                       |
| **Gemma 3**                                       | `Gemma3ForCausalLM`<br>`Gemma3ForConditionalGeneration`                       |     ✓     |                       |
| **Gemma 4**                                       | `Gemma4ForConditionalGeneration`<br>`Gemma4UnifiedForConditionalGeneration`   |     ✓     | E2B/E4B unsupported   |
| **GLM 4**, **GLM 4.6**, etc.                      | `Glm4ForCausalLM`<br>`Glm4MoeForCausalLM`                                     |            |                       |
| **GLM 4.1V**, **GLM 4.5V**                        | `Glm4vForConditionalGeneration`<br>`Glm4vMoeForConditionalGeneration`         |     ✓     |                       |
| **GLM 4.7 Flash**                                 | `Glm4MoeLiteForCausalLM`                                                      |            |                       |
| **GLM 5.2**                                       | `GlmMoeDsaForCausalLM`                                                        |            |                       |
| **GLM 5.3-Flash**                                 | `Glm5NextForConditionalGeneration`                                            |     ✓     |                       |
| **GPT-OSS**                                       | `GptOssForCausalLM`                                                           |            |                       |
| **HyperCLOVAX**                                   | `HyperCLOVAXForCausalLM`<br>`HCXVisionV2ForCausalLM`                          |     ✓     |                       |
| **Hy3**                                           | `HYV3ForCausalLM`                                                             |            |                       |
| **IQuest-Coder**                                  | `IQuestCoderForCausalLM`                                                      |            |                       |
| **Kimi Linear**                                   | `KimiLinearForCausalLM`                                                       |            |                       |
| **Ling 3.x**                                      | `BailingMoeV3ForCausalLM`                                                     |            |                       |
| **Laguna 2.1**                                    | `LagunaForCausalLM`                                                           |            |                       |
| **LFM 2.5**, **LFM2.5-VL**                        | `Lfm2ForCausalLM`<br>`Lfm2MoeForCausalLM`<br>`Lfm2VlForConditionalGeneration` |     ✓     |                       |
| **LFM 3**, **LFM3-VL**                            | `Lfm3ForCausalLM`<br>`Lfm3VlForConditionalGeneration`                          |     ✓     |                       |
| **Llama 1/2/3**,**3.1-Nemotron** etc.             | `LlamaForCausalLM`                                                            |            |                       |
| **MiMo-RL**                                       | `MiMoForCausalLM`                                                             |            |                       |
| **MiMo-V2.6-Flash**                               | `MiMoV2ForCausalLM`                                                           |     ✓     | no audio              |
| **MiniMax-M2**                                    | `MiniMaxM2ForCausalLM`                                                        |            |                       |
| **Mistral**, **Ministral 3**, **Mistral-4** etc.  | `MistralForCausalLM`<br>`Mistral3ForConditionalGeneration`                    |     ✓     |                       |
| **Mixtral**                                       | `MixtralForCausalLM`                                                          |            |                       |
| **NemotronH, Nemotron-3 Nano/Super**              | `NemotronHForCausalLM`                                                        |            |                       |
| **Nemotron 3.5 Bolt**, **Nemotron 3.5 Super VL**  | `NemotronH_Omni_Reasoning_V3`                                                 |     ✓     | Image and video input |
| **Olmo 3.1**                                      | `Olmo3ForCausalLM`                                                            |            |                       |
| **Olmo-Hybrid**                                   | `OlmoHybridForCausalLM`                                                       |            |                       |
| **Phi3**, **Phi4**                                | `Phi3ForCausalLM`                                                             |            |                       |
| **Qwen 2**, **Qwen 2.5**, **Qwen 2.5 VL**         | `Qwen2ForCausalLM`<br>`Qwen2_5_VLForConditionalGeneration`                    |     ✓     |                       |
| **Qwen 3**                                        | `Qwen3ForCausalLM`<br>`Qwen3MoeForCausalLM`                                   |            |                       |
| **Qwen 3-Next**                                   | `Qwen3NextForCausalLM`                                                        |            |                       |
| **Qwen 3-VL**                                     | `Qwen3VLForConditionalGeneration`                                             |     ✓     |                       |
| **Qwen 3-VL MoE**                                 | `Qwen3VLMoeForConditionalGeneration`                                          |     ✓     |                       |
| **Qwen 3.5**                                      | `Qwen3_5ForConditionalGeneration`                                             |     ✓     |                       |
| **Qwen 3.5 MoE**                                  | `Qwen3_5MoeForConditionalGeneration`                                          |     ✓     |                       |
| **Qwen 3.8-Flash-Next**                           | `Qwen4ExpForConditionalGeneration`                                            |     ✓     |                       |
| **SenseNova Loop** (text only)                    | `SenseNovaLoopForConditionalGeneration`                                       |     ✓     |                       |
| **Seed-OSS**                                      | `SeedOssForCausalLM`                                                          |            |                       |
| **SmolLM**                                        | `SmolLM3ForCausalLM`                                                          |            |                       |
| **SolarOpen**                                     | `SolarOpenForCausalLM`                                                        |            |                       |
| **Step 3.5 Flash**                                | `Step3p5ForCausalLM`                                                          |            |                       |
| **Step 3.7 Flash**                                | `Step3p7ForConditionalGeneration`                                             |     ✓     |                       |

Always adding more, stay tuned.

## Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume an interrupted quant job
python convert.py -w <working_dir> -r

# More options
python convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors 
until the converted model can be compiled. It should have enough free space to store an entire copy 
of the output model.

See the [conversion guide](doc/convert.md) for more information, or the 
[self-calibration guide](doc/optimize.md). 

## EXL3 quantization

EXL3 quantization is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. It aims to make
SOTA quantization available to users on consumer hardware. The conversion process is designed to be
simple and efficient and requires only an input model (in HF format) and a target bitrate. By
computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a 
model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger
ones (70B+) on a single high-end consumer GPU (see the [conversion guide](doc/convert.md)).

For more information, see the [**QTIP**](https://arxiv.org/abs/2406.11235) and [**QuIP#**](https://arxiv.org/abs/2402.04396) papers, as well as this 
[excellent writeup](https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip) on **QTIP** from together.ai.


## Community

You are always welcome to join the [ExLlama discord server](https://discord.gg/NSFwVuCjRq) ←🎮


### 🤗 Models on Hugging Face

Browse the [EXL3 model collection](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a) for quantized models. Also shout out to the following lovely
people:

- [ArtusDev](https://huggingface.co/ArtusDev)
- [MikeRoz](https://huggingface.co/MikeRoz)
- [MetaphoricalCode](https://huggingface.co/MetaphoricalCode)
- [Ready.Art](https://huggingface.co/ReadyArt)
- [isogen](https://huggingface.co/isogen/models)


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous
supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (chunked linear-attention prefill kernels, vendored under `exllamav3/vendor/fla`)

<p align="center">
  <img src="doc/cat.png" width="40" alt="">
</p>


