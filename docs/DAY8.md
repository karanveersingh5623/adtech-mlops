# Day 8 — Model Optimization II: Pruning and Knowledge Distillation

Guided exercise for [Day 8 of the roadmap](ROADMAP.md#day-8--model-optimization-ii-pruning-and-knowledge-distillation). Builds on the [Day 7](DAY7.md) neural baseline and its benchmark method.

**Status:** 🗓️ planned. Nothing on this page is implemented yet. **It contains no solution code**: contributors write their own implementation.

**Time box:** about 30 minutes of shared concepts, then about 1.5 hours each for pruning and distillation, then 45 minutes for the benchmark summary and write-up.

- **Split the hands-on work.** One contributor takes pruning and the other takes distillation, then each reviews and explains the other's work. Swap the roles you took on Day 7.
- **Keep it bounded.** Use small models, a few epochs, and few settings. Each training run should take seconds to a few minutes on a laptop CPU.
- **Optional extras.** Items marked ➕ are optional.

**Dependencies:** the PyTorch and ONNX packages added on Day 7. No new dependencies are needed.

---

## Part 1 — Pruning

### Concepts

- **Magnitude-based pruning.** Remove the weights with the smallest absolute value, on the assumption that they contribute least to the output.
  - *Layer-wise* pruning removes the same fraction from each layer.
  - *Global* pruning ranks all weights together.
- **Sparsity** = the number of zero weights divided by the total number of weights. Report it per layer and overall.
- **Granularity:**
  - **Weight-level (unstructured):** individual weights are set to zero, anywhere in the matrix.
  - **Neuron-level (structured):** a whole hidden unit is removed. In a `Linear` layer, that means one output row of its weight matrix, its bias, and the matching input column of the *next* layer.
  - **Channel-level (structured):** a whole convolution filter or channel is removed. This is the convolutional equivalent of neuron-level pruning.
- **Fine-tuning after pruning.** Pruning usually lowers quality. A few epochs of further training, with the pruned weights held at zero, often recover most of it. Pruning in several rounds, each followed by fine-tuning, usually reaches higher sparsity *at the same quality* than pruning once.
- **The quality vs compression trade-off.** Plot or tabulate quality against sparsity. Quality typically stays flat up to some sparsity level and then drops sharply. Finding that point is the goal of the experiment.

### Why zero-valued weights do not automatically speed anything up

- A **dense** matrix multiplication multiplies every element, including the zeros, so the arithmetic is unchanged.
- The model is still **stored densely**: a zero takes 4 bytes in FP32, just like any other value. The file only gets smaller if you use a sparse storage format, or if you compress it, because runs of zeros compress well.
- Sparse kernels need **high sparsity** and suitable hardware to beat dense kernels. Some accelerators support only specific patterns, for example two zeros in every block of four weights.
- **Structured pruning removes whole rows and columns**, so the matrices really are smaller. This *can* give real size and latency reductions on ordinary dense hardware, but only once you **rebuild the model with smaller layers**. Zeroing a neuron does not change any matrix shape. For tiny models, framework overhead may still hide the latency gain.
- **Framework detail to check yourself.** Some pruning utilities, including PyTorch's `torch.nn.utils.prune`, work by keeping the original weights *and* a mask. Until the pruning is made permanent, the saved model can be **larger** than before. Read the documentation, then measure it.

### Hands-on

1. **Baseline.** Use the Day 7 FP32 MLP. If it is too small to prune meaningfully, use a wider variant, such as 30 → 256 → 128 → 1, and record its quality first.
2. **Unstructured magnitude pruning.**
   - Prune the linear-layer weights at two sparsity levels, for example 50% and 90%. ➕ Add an intermediate level such as 80%.
   - For each level, measure the per-layer and overall sparsity.
   - Measure quality before and after a short fine-tune.
   - Compare predictions with the unpruned model: the maximum absolute probability difference and top-1 ranking agreement.
3. **Make the pruning permanent and measure.**
   - Saved file size: raw, and compressed with `gzip`.
   - Latency, using the same method as Day 7.
   - Explain why the raw size and the latency barely change.
4. **Structured (neuron-level) pruning.**
   - Rank the hidden neurons of one layer by the norm of their weights, and remove a fraction of them, for example 50%.
   - Mask each removed neuron's weight row **and its bias**. Some pruning utilities mask only the weight. A neuron with a surviving bias still outputs activation(bias), so it has not really been removed.
   - **Build a smaller model** by copying the kept rows into the shrunk layer and the matching columns into the next layer.
   - Before fine-tuning, verify that the smaller model matches the masked model within a tight tolerance, for example a maximum absolute difference ≤ 10⁻⁶. Bitwise equality is not guaranteed, because different matrix shapes change the order of floating-point summation.
   - Then fine-tune it, and measure its size, latency, and quality.

**Hints:**

- To check your rebuilt model, compare it with the masked model on the same inputs. If the outputs differ, a column in the next layer was probably dropped or kept by mistake.
- Count parameters before and after. Removing *k* neurons from a layer with *n*in inputs, whose outputs feed a layer with *n*out outputs, removes *k*·*n*in + *k* + *k*·*n*out parameters.

---

## Part 2 — Knowledge distillation

### Concepts

- **Teacher and student.**
  - A larger, more accurate *teacher* is trained first.
  - A smaller, cheaper *student* is then trained to imitate the teacher's outputs, in addition to (or instead of) learning from the true labels.
- **Soft targets.**
  - The teacher's predicted probabilities carry more information than 0/1 labels. They show *how* confident the teacher is, and which examples are borderline.
  - The student learns from this richer signal.
- **Temperature *T*.**
  - Logits are divided by *T* before the softmax, or the sigmoid in this binary task: *p* = σ(*z* / *T*).
  - *T* = 1 is the ordinary prediction.
  - **Multi-class case:** *T* > 1 softens the softmax and reveals how the teacher ranks the *wrong* classes relative to each other. This is the classic "dark knowledge".
  - **Binary case:** a single sigmoid output already carries all of the teacher's information about the example, so temperature reveals nothing new. Instead, it changes **how the student is penalized**. As *T* grows, the soft loss behaves more and more like matching the teacher's *logits*. That gives more weight to examples whose probabilities are close to 0 or 1, which describes most low-CTR rows, where differences in probability are tiny but differences in logits are not.
- **Training objective (binary version).**
  - Loss = α · *T*² · BCE(σ(*z*student / *T*), σ(*z*teacher / *T*)) + (1 − α) · BCE(σ(*z*student), *y*)
  - The first term matches the teacher's softened output, and the second term fits the true label.
  - The *T*² factor keeps the soft term's gradients comparable in size as *T* changes. This formulation follows Hinton, Vinyals and Dean, *Distilling the Knowledge in a Neural Network* (2015), adapted from softmax to sigmoid.
- **Capacity and generalization.**
  - A student that is too small cannot represent the function, however good the teacher is.
  - A distilled student often generalizes better than the same small model trained on labels alone, but **this is not guaranteed**.
  - On simple synthetic data, where the labels are already easy to learn, the benefit may be close to zero. That is a valid, reportable result.

### Hands-on

1. **Teacher.** Train a wider MLP, for example 30 → 256 → 256 → 1, or use your Day 5 features, which gives the teacher more to learn. Record its quality, size, and latency.
2. **Student trained on labels only.** Train a small MLP, for example 30 → 8 → 1, on the true labels alone. This is the fair comparison for distillation.
3. **Distilled student.**
   - Train the *same* student architecture, with the same seed and the same epochs, using the distillation objective above.
   - Try a small grid only: *T* ∈ {1, 4} with α = 0.5. ➕ Extend it to *T* = 2 and α = 0.9.
   - Choose the setting on the **validation** period and report it on the **test** period.
4. **Compare** the teacher, the label-only student, and the distilled student on quality (log loss, ROC-AUC, PR-AUC, calibration), parameter count, file size, and latency.

**Hint:** precompute the teacher's logits once, for the training set, and store them. The student's training loop then never needs to run the teacher.

---

## Part 3 — Benchmark summary and guidance

Extend `reports/optimization.md` from Day 7 with one table that covers every variant from both days, measured the same way: size, latency, throughput, quality, and sparsity where relevant. **Every cell must come from an actual run.**

Then complete this guidance table in your own words, backed by your measurements:

| Technique | Reduces | Typically helps when | Typically does not help when |
|---|---|---|---|
| Quantization (INT8) | Weight storage, memory traffic; compute on supporting hardware | | |
| Unstructured pruning | Non-zero weights; compressed size | | |
| Structured pruning | Layer dimensions, so real size and compute | | |
| Distillation | Model capacity needed for a given quality | | |
| ONNX export | Framework overhead; enables runtime optimizations | | |
| More replicas (not a model optimization) | Not the compute per inference; adds total throughput and can cut queueing delay (tail latency) under saturation | | |

---

## Acceptance criteria

- [ ] Unstructured pruning is applied at several sparsity levels. Sparsity, quality, and prediction changes are measured before and after fine-tuning.
- [ ] Measured size and latency show and explain why dense zeros do not speed up dense inference.
- [ ] A structurally pruned model is rebuilt with smaller layers. A test verifies that it matches the masked model, and its size and latency are measured.
- [ ] The teacher, the label-only student, and the distilled student are trained with matched seeds and epochs, and compared on quality, size, and latency.
- [ ] The benchmark summary and guidance table are complete, with conclusions supported by the numbers.

## Questions to answer before Day 9

1. At 90% unstructured sparsity, why might the model file and its latency be unchanged? What would you need to change to benefit?
2. Why does removing one hidden neuron require changes to two weight matrices?
3. What extra information does a soft target carry compared with a 0/1 label? In this binary task, how does raising *T* change what the student is penalized for, and how would that differ with many classes?
4. If the distilled student is no better than the label-only student, what are three possible explanations?
5. For a ranking service with a strict p99 budget, which technique from Days 7–8 would you try first, and why?
