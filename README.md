
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
| Control vectors | Applies llama.cpp control vectors to the residual stream. Refer to [Control vectors](#control-vectors). |
| Pipelined prefill with CPU experts | Lets the two GPUs work on different prefill chunks when the CPU expert split is on. This is a port of [a commit by Auryn Macmillan](https://github.com/auryn-macmillan/exllamav3/commit/b38de78bd0376fcdc4f92344ed405800ac8374b6). The port also adds a lock to the n-gram row table. Prefill of prompts longer than one chunk is approximately 10% faster. `EXL3_PIPELINED_PREFILL=0` turns it off. |
| Late promote hook | Moves the promote-on-miss step off the critical path of decode and makes its budget a setting. Decode is approximately 17% faster on this system. Refer to [Late promote hook](#late-promote-hook). |
| Expert lookahead (experimental, off by default) | Predicts the experts of the next split layer and copies them from RAM to VRAM before the layer needs them. It needs a source build of the extension. On this system it makes decode slower. Refer to [Expert lookahead](#expert-lookahead). |
| Inkling-2-Mini (work in progress) | Adds the looped Inkling-2 text model: load from the source checkpoint or from an EXL3 conversion of its experts, cache, generator, CUDA-graph decode and decode kernels. It needs a source build of the extension for full speed. Refer to [Inkling-2-Mini](#inkling-2-mini). |

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

> [!WARNING]
> This is work in progress. Text input only. The vision tower is not implemented. The draft model (the MTP head) works in the Python API; TabbyAPI does not load it yet.

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
| Cache, generator, tabbyAPI | Works. Each loop pass has its own cache layers. FP16 cache only. |
| Context length | Tested with a 161,543-token prompt on the 3090 + 4090. |
| Decode as CUDA graph replays, with decode kernels | Works for one sequence with EXL3 experts. |
| Multi-token graph steps (verify drafted tokens) | Works for a maximum of 8 tokens in one forward pass. Nothing uses them yet. |
| Prefill | Plain PyTorch (cuBLAS). |
| MTP head (`mtp.safetensors`, DSpark draft model) | Works in the Python API as a draft model (`architecture/inkling2_dspark.py`). Greedy, 300 tokens, low reasoning effort: 146 T/s against 72 for code, 96 against 72 for a list, 78 against 72 for prose, 32 against 69 for a 66-token answer. 17–47% of the drafts are accepted. Not quantized. Not set up in TabbyAPI. |
| Vision | Not implemented. |
| Tensor parallelism | Not supported. |

#### Requirements

- **A source build of the extension for full speed.** The patch adds CUDA kernels (`exllamav3/exllamav3_ext/inkling2.cu`) and the Inkling-2 activation in the expert kernels. The release wheels do not contain them. With a wheel, the model gives correct results on a slow path, and the log shows a line that starts with ` !! Inkling2:`.
- **The native CUDA allocator.** CUDA graph capture does not work with `cudaMallocAsync`. In tabbyAPI, set `cuda_malloc_async: false`. With the async allocator the model decodes without graphs and shows a `!!` line.
- **The FP16 cache.** Quantized cache modes are not supported.
- **A GPU split with free memory.** Attention on the global layers needs working memory that grows with the context. On this system, tabbyAPI uses `autosplit_reserve: [3072, 1024]`.

Build the extension with:

```
# in the root of this repository
python setup.py build_ext --build-lib ../exl3_build/lib --build-temp ../exl3_build/tmp
```

Then put `exl3_build/lib` first on `PYTHONPATH`.

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

> [!NOTE]
> The top-k expert routing of this model is discontinuous. Two runs that differ only by rounding give logits that differ by 2 to 3%. The "noise floor" line shows this effect. Do not compare outputs token by token without it.

#### Settings

| Variable | Value | Default |
|---|---|---|
| `EXL3_I2_GRAPH` | `0` runs each decode step operation by operation, without CUDA graphs. | `1` |
| `EXL3_I2_GRAPH_WIDTH` | The maximum number of tokens of one sequence in one forward pass that runs as a graph. | `8` |
| `EXL3_I2_KERNELS` | `0` uses the PyTorch forms in the graph and not the decode kernels. | `1` |
| `EXL3_I2_MOE` | `dense` runs the EXL3 experts in a slow reference loop. Use it to compare with `kernel`. | `kernel` |
| `EXL3_I2_ADT` | The activation data type: `fp16`, `bf16` or `fp32`. The decode kernels need `fp16`. | `fp16` |
| `EXL3_I2_NAN_CHECK` | `1` shows a line when a forward pass gives logits that are not finite. | `0` |

#### Test results

All results are for the EXL3 conversion on the RTX 4090 + RTX 3090, one sequence.

Decode (`util/inkling2/graph_check.py`):

| Context (tokens) | Operation by operation | CUDA graph with kernels |
|---|---|---|
| 600 | 26 T/s | 72 T/s |
| 2,000 | 26 T/s | 71 T/s |
| 9,000 | 26 T/s | 67 T/s |
| 30,000 | 25 T/s | 59 T/s |

Forward passes with more than one token of one sequence (`util/inkling2/graph_spec_check.py`):

| Tokens | Operation by operation | Graph |
|---|---|---|
| 1 | 40.4 ms | 14.5 ms |
| 2 | 61.3 ms | 18.5 ms |
| 4 | 61.6 ms | 23.1 ms |
| 8 | 61.8 ms | 30.0 ms |

Through tabbyAPI:

| Test | Result |
|---|---|
| Short chats and tool calls | Approximately 65 T/s |
| 9K-token prompt | Prefill 1,751 T/s, decode 61 T/s |
| 44K-token prompt | Prefill 1,068 T/s |
| 161,543-token prompt (older build, before the graph and kernel work) | Prefill 183 T/s, decode 12.9 T/s, correct answer to a question about a sentence at 10% depth |

#### How the patch works

- `exllamav3/modules/inkling2.py` has the model in plain PyTorch. It is the reference, and it runs prefill.
- `exllamav3/architecture/inkling2_loop.py` has the modules, the cache state and the graph code. The loop uses the layer map of exllamav3: the looped layers occur three times in the forward order, and each occurrence has its own cache layers.
- The global layers keep K/V in the paged cache. The sliding layers keep the last window in a buffer for each sequence. All layers keep the last inputs of their convolutions in a ring.
- A decode step is captured as one CUDA graph for each GPU. All cache indexes are computed on the GPU from two integers, so one captured graph is correct for all positions.
- `exllamav3/exllamav3_ext/inkling2.cu` has five kernels that replace groups of small PyTorch operations in the graph: convolution, block attention residual, message gate, Q/K/V post-processing and attention.

The checks in `util/inkling2/` compare the port with the transformers reference (`ref_dump.py`, `port_check.py`), cached decode with an uncached pass (`cache_check.py`) and graph decode with operation-by-operation decode (`graph_check.py`, `graph_seq_check.py`, `graph_spec_check.py`).

#### Limits

- Graph decode is for one sequence. Batches, prefill and forward passes wider than `EXL3_I2_GRAPH_WIDTH` run operation by operation.
- Prefill at long context is slow. At 40K tokens of context, 84% of the prefill time is attention on the global layers.
- The checkpoint of the cache state is approximately 162 MiB, because it holds the attention window of all sliding layer applications.
- A few experts that the model uses rarely saw very little calibration data (a minimum of 1 token in layer 22).
- On one 44K-token test prompt, graph decode and operation-by-operation decode gave different answers. All step-by-step comparisons agree, so this is the routing effect in the note above, but it is not proven for all prompts. `EXL3_I2_GRAPH=0` is the switch to compare.

#### TODO

1. **MTP draft model: make it pay off everywhere.** It always drafts 7 tokens, and a rejected draft costs a wide verification pass (30 ms against 14.5 ms). Add a confidence cut (the draft softmax probability) so the block is cut or skipped when the drafter is unsure; put the draft block and the Markov chain in a CUDA graph; move the two windowed draft layers out of the paged cache; set it up in TabbyAPI (`util/inkling2/make_dspark_dir.py` makes the model directory).
2. **Prefill attention.** A kernel that uses tensor cores for the global layers. A scalar kernel was tested and was 3 to 5 times slower than cuBLAS.
3. **Decode.** Tune the block attention residual and gate kernels, and replace the remaining small PyTorch operations. The expert kernel (approximately 4.4 ms of 14 ms for each token) and memory bandwidth set the limit after that.
4. **Graph steps wider than 8 tokens**, for the n-gram draft verification of tabbyAPI (65 tokens).
5. **Batched decode** as graphs.
6. **Vision tower.**
7. **Quantized cache modes** for the global layers.
8. **Native NVFP4 expert kernels**, to run the source checkpoint without a conversion.
9. **Conversion with the stock converter**, which needs support for looped models, and more calibration data for the rare experts.

### Control vectors

#### Overview

A control vector changes the residual stream of the model during inference. It does not change the model weights. This patch applies control vectors in the llama.cpp format. It is a port of the `--control-vector` function of llama.cpp. It also includes the projection mode of [Cudecnik/Qwen3.8-Flash-Next-refusal-projection](https://huggingface.co/Cudecnik/Qwen3.8-Flash-Next-refusal-projection).

The patch has two modes:

- **Projection (`project`).** The patch removes one direction from the residual stream: `h = h - s·(h·v)·v`. `v` is the unit vector for the layer. `s` is the scale.
- **Addition (`add`).** The patch adds the vector to the residual stream: `h = h + s·v`. This is the standard llama.cpp mode.

The patch is off by default. It has no effect until you set `EXL3_CVEC`.

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


