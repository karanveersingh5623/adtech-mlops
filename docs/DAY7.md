# Day 7 — Model Optimization I: Quantization and ONNX

Guided exercise for [Day 7 of the roadmap](ROADMAP.md#day-7--model-optimization-i-quantization-and-onnx). Day 8 continues with [pruning and distillation](DAY8.md).

**Status:** 🗓️ planned. Nothing on this page is implemented yet. It describes concepts, the experiment, and how to evaluate it. **It contains no solution code**: contributors write their own implementation.

**Time box:** about 1.5 hours of concepts, 2 hours of experiments, and 30 minutes of write-up. If time runs out, keep the FP32 baseline, the ONNX export, and **one** quantization variant, and drop the optional extras.

**Prerequisites:**

- The Day 3 chronological split and features.
- The Day 6 benchmark approach.
- [Core ML system concepts](ARCHITECTURE.md#8-core-ml-system-concepts), especially *model size vs data volume* and *scaling vs optimization*.

**New dependencies, added when the day starts:**

- A CPU build of PyTorch.
- `onnx` and `onnxruntime`. The exact extra packages the ONNX exporter needs depend on the PyTorch version, so check its documentation.

Pin every version in `requirements.txt` and justify each one in the pull request.

---

## Part 1 — Concepts

### 1. Number representations

| Format | Bits | Layout | Notes |
|---|---|---|---|
| **FP32** | 32 | 1 sign, 8 exponent, 23 mantissa | Default training and inference format; about 7 significant decimal digits |
| **FP16** | 16 | 1 sign, 5 exponent, 10 mantissa | About 3 significant digits; maximum value about 65,504, so overflow is possible |
| **BF16** | 16 | 1 sign, 8 exponent, 7 mantissa | Same range as FP32 with less precision; common in training on modern accelerators |
| **INT8 / UINT8** | 8 | Signed integer in [−128, 127] (INT8), or unsigned in [0, 255] (UINT8) | Needs a *scale* and *zero point* to represent real numbers |

Halving the bits roughly halves the weight storage and the memory traffic. Whether it also reduces **latency** depends on whether the hardware and runtime have fast kernels for that format.

### 2. Floating-point reduction vs integer quantization

- **FP32 → FP16/BF16** is still floating point. Values are rounded to fewer bits, but every value keeps its own exponent, so no extra parameters are needed. Many CPUs do not compute FP16 faster than FP32. GPUs and other accelerators usually do.
- **FP32 → INT8** is *quantization*. Real values are mapped onto a small grid of integers with an affine transform. Integer kernels can be much faster on hardware that supports them, but rounding and clipping errors are larger and have to be measured.

### 3. Scale and zero point (affine quantization)

For a tensor whose values lie in [*x*min, *x*max], mapped to integers [*q*min, *q*max]:

- scale *s* = (*x*max − *x*min) / (*q*max − *q*min)
- zero point *z* = round(*q*min − *x*min / *s*), clamped to [*q*min, *q*max]
- quantize: *q* = clamp(round(*x* / *s*) + *z*, *q*min, *q*max)
- dequantize: *x̂* = *s* · (*q* − *z*)

**Worked example** (UINT8, [0, 255], weights in [−1.0, 1.5]):

- *s* = 2.5 / 255 ≈ 0.009804, and *z* = round(0 − (−1.0 / 0.009804)) = round(102.0) = 102.
- *x* = 0.30 → *q* = round(30.6) + 102 = 31 + 102 = 133 → *x̂* = 0.009804 × 31 ≈ 0.3039. The error is about 0.004, under half a step (*s*/2).
- *x* = 0.0 → *q* = 102 → *x̂* = 0.0 exactly. A zero point lets real zero be represented exactly, which matters for padding and ReLU outputs.
- A single outlier at 10.0 would stretch the range to about 11. The scale would grow more than four times, and every other value would lose precision. This is why *range calibration* matters.

Related choices:

- **Symmetric vs asymmetric.** Symmetric quantization fixes *z* = 0 and uses [−*a*, *a*]. Asymmetric quantization uses the actual min and max, as in the example above.
- **Per-tensor vs per-channel.** Per-tensor quantization uses one *s* and *z* for the whole tensor. Per-channel quantization uses one pair per output channel or row, which usually gives better accuracy for weights.

### 4. Dynamic vs static quantization

| | Dynamic | Static |
|---|---|---|
| Weights | Quantized ahead of time | Quantized ahead of time |
| Activations | Range computed **at run time**, per batch | Range **calibrated in advance** on representative data |
| Needs calibration data | No | Yes |
| Typical fit | Models dominated by linear layers (MLPs, LSTMs, transformers) on CPU | Convolutional networks; deployments where the whole path runs in integer arithmetic |
| Overhead | Computing ranges and quantizing activations at run time | Requires calibration data that represents production traffic |

### 5. Post-training quantization (PTQ) vs quantization-aware training (QAT)

- **PTQ** quantizes an already trained model. It is fast and needs no retraining, and it is usually good enough at INT8 for well-behaved models.
- **QAT** inserts *fake quantization*, which rounds values in the forward pass, during training or fine-tuning, so that the model learns weights that tolerate rounding. It costs extra training and recovers accuracy when PTQ loses too much. It matters most at very low bit widths.

### 6. ONNX and ONNX Runtime

- **ONNX** is an open format that describes a model as a graph of standard **operators**, versioned by **opset**. Exporting traces the model into that graph.
- **ONNX Runtime** executes ONNX graphs through **execution providers**, such as CPU or CUDA. Each execution provider applies its own graph optimizations and kernels.
- **Compatibility:**
  - Some framework operations have no ONNX operator in a given opset, so the export fails or needs a rewrite.
  - Quantized operators are supported differently by different execution providers and hardware.
  - Dynamic dimensions, such as batch size, must be declared at export time.

  Always check that the exported model runs and gives the same results.
- **Why export at all?** To get a runtime independent of the training framework, graph-level optimizations, portability across languages and hardware, and access to the runtime's quantization tools.

### 7. Numerical parity and quality

- **Export parity:** the FP32 ONNX model should match the original within float rounding. Compare the maximum absolute difference in predicted probabilities with a stated tolerance; the order of 10⁻⁶–10⁻⁵ is typical.
- **Quantized quality:** INT8 outputs will *not* match exactly. Judge them by task quality instead:
  - log loss, ROC-AUC, PR-AUC
  - calibration
  - **ranking agreement**, such as the share of requests where the top-1 candidate is unchanged

### 8. Why quantization may *not* reduce latency

- For tiny models, the time is dominated by Python, framework, and data-conversion overhead, not by arithmetic.
- The CPU may lack fast INT8 or FP16 instructions, or the runtime may not use them.
- Quantize and dequantize steps add work at the boundaries of each layer.
- Memory-bound and compute-bound workloads benefit differently.
- On GPUs, small batches can be dominated by data-transfer and kernel-launch overhead.

**Do not promise a speed-up. Measure it, and report the result even when there is none.**

---

## Part 2 — Experiment

### Step 1. Small neural-network baseline (FP32)

- Train a small MLP on the Day 3 chronological split with the same features as the logistic-regression model. Use the 30 → 64 → 32 → 1 network from the [architecture notes](ARCHITECTURE.md#82-model-size-vs-data-volume), or your Day 5 feature set. The input size depends on your encoding: 30 assumes one-hot `device`, `ad_category`, and `hour`.
- Fix all random seeds. Record the number of epochs and the training time.
- Compare against the Day 3 logistic regression on the test period: log loss, ROC-AUC, PR-AUC. *The MLP does not need to win.* On this synthetic data, a simple model may be just as good. Explain the result you get.

### Step 2. Export to ONNX

- Export the FP32 model with a dynamic batch dimension.
- Run it with ONNX Runtime and check export parity on the test set (concept 7).
- ➕ Optional: also export the Day 3 scikit-learn pipeline with `skl2onnx` and compare.

### Step 3. Quantize

- Apply **dynamic INT8 post-training quantization** to the exported ONNX model with ONNX Runtime's quantization tools. This is the recommended default, because the FP32 and INT8 variants then go through the same runtime.
- If you try PyTorch's own dynamic quantization instead, check these points against the documentation for your pinned version:
  - Whether its quantization APIs are current or deprecated.
  - Which quantized engine your CPU needs. ARM machines, including Apple-silicon Macs, may need a different engine from x86.
  - That a model quantized in PyTorch generally cannot be exported to ONNX afterwards.

  Record which path you used. Do not mix the two paths in one comparison row.
- ➕ Optional variants: FP16, a static quantization with a calibration set drawn from the *training* period, or a wider network (for example 512 → 512) to see when quantization starts to matter.

### Step 4. Benchmark: FP32 vs optimized

Measure every variant the same way:

| Measure | Method |
|---|---|
| Artifact size | Bytes on disk of the saved model file |
| Latency | Batch size 1 *and* a batch the size of a typical candidate list. Warm up first, then run many repetitions and report median and p95 |
| Throughput | Predictions per second at the larger batch size |
| Memory, where practical | Peak resident memory of a **separate process per variant**, minus an import-only baseline process. Importing PyTorch or ONNX Runtime alone costs hundreds of MB, which hides a few KB of weights. `resource.getrusage(...).ru_maxrss` reports bytes on macOS but kilobytes on Linux. `tracemalloc` does not see native allocations |
| Prediction quality | Log loss, ROC-AUC, PR-AUC, max absolute probability difference vs FP32, top-1 ranking agreement |

Record the hardware, OS, library versions, thread settings, and whether timings include Python overhead.

### Step 5. Write `reports/optimization.md`

Use a results table like this. It stays empty until you have measured the numbers:

| Variant | Size (KB) | p50 latency, batch 1 (µs) | p95 latency, batch 1 (µs) | Throughput, batch N | Log loss | ROC-AUC | Max \|Δp\| vs FP32 |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | | | | | | | — |
| ONNX FP32 | | | | | | | |
| ONNX Runtime INT8 (dynamic) | | | | | | | |

Then write a short conclusion: what changed, what did not, and why, using the concepts above.

---

## Acceptance criteria

- [ ] The FP32 neural baseline is trained reproducibly and compared with logistic regression.
- [ ] The ONNX export runs in ONNX Runtime, and parity is verified within a stated tolerance.
- [ ] At least one quantized variant is benchmarked against FP32 on size, latency, and quality, plus memory where practical.
- [ ] The results come from actual runs, with the environment recorded.
- [ ] Every claimed speed-up or slow-down is explained with a concrete cause.
- [ ] Tests cover the parity check, so that a broken export fails a test.

## Questions to answer before Day 8

1. Why does the INT8 model of a 4,097-parameter network save only about 12 KB, and why might that not matter for latency?
2. What would happen to quantization error if one weight were 100× larger than all the others? How do per-channel scales help?
3. When would you choose QAT over PTQ?
4. Your INT8 model is 3× smaller but not faster. Give two plausible reasons and describe how you would check each one.
5. Why is adding replicas not a substitute for this day's work, and why is this day's work not a substitute for adding replicas?
