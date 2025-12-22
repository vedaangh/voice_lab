# Interesting Notes

## LLaMA-style Transformer Design Choices

### RMSNorm (vs LayerNorm)
LayerNorm normalizes by mean and variance with learned scale/shift. RMSNorm only normalizes by root-mean-square - no mean centering, no bias. ~10-15% faster, empirically works just as well.

### Pre-norm (vs Post-norm)
Post-norm: `x = LayerNorm(x + Attention(x))` - norm after residual addition.
Pre-norm: `x = x + Attention(RMSNorm(x))` - norm before sublayer, raw x in residual.

Pre-norm keeps the residual path "clean" (just additions), which stabilizes training in deep networks and helps gradient flow.

### SiLU/Swish Activation (vs ReLU/GELU)
SiLU: `x * sigmoid(x)` - smooth, non-monotonic, self-gated. Avoids ReLU's "dead neuron" problem (zero gradient for negative inputs). Self-gating property helps gradient flow.

### Gated FFN (vs Standard FFN)
Standard: `W_down · ReLU(W_up · x)`
Gated: `W_down · (SiLU(W_gate · x) * (W_up · x))`

The gate projection learns which features to let through via element-wise multiplication. Adds parameters but improves expressiveness - acts as a learned feature selector.

### Rotary Position Embeddings (RoPE)
Rotates query/key vectors based on position at every attention layer. Benefits:
- Encodes relative position (attention depends on distance between positions)
- Extrapolates to longer sequences than training
- No extra parameters
- Position info doesn't "wash out" in deep networks (applied per-layer)

## Residual Connections
`output = x + Layer(x)` instead of `output = Layer(x)`. Provides a "highway" for gradients to flow directly backward in deep networks. Lets layers learn *changes* to input rather than full transformation.

## CTC Loss and Alignments

### The Problem
Our decoder outputs a sequence of length T (e.g., 250 after upsampling), but the target unit sequence Y_U has length L (e.g., 30 deduplicated units). We need to align these without knowing the exact frame-to-unit correspondence.

### Output Projection
We project the decoder hidden states to K+1 classes:
- K = 1000 unit centroids (indices 0-999)
- 1 blank token (index 1000)

This gives logits of shape `[batch, T, 1001]`, which after softmax is a distribution over units + blank at each timestep.

### The Collapsing Function β
CTC defines a collapsing function β that maps an alignment A (length T) to an output sequence (length ≤ T):
1. Remove consecutive duplicates
2. Remove all blank tokens

Example:
```
A = [3, 3, blank, blank, 5, 5, 5, blank, 2, 2]
β(A) = [3, 5, 2]
```

### Inverse Collapsing β⁻¹
Given target Y_U = [3, 5, 2], the set β⁻¹(Y_U) contains ALL valid alignments that collapse to it. This set grows **exponentially** with sequence length because:
- Blanks can be inserted anywhere (between units, at start/end)
- Each unit can repeat any number of times
- The only constraint: units must appear in order, separated by at least one blank or being different

For target length L and output length T, the number of valid alignments is roughly O(C(T, L) × 2^(T-L)) - exponential in T.

### CTC Loss Computation
The loss marginalizes over ALL valid alignments:

```
L_CTC = -log P(Y_U | O) = -log Σ_{A ∈ β⁻¹(Y_U)} P(A | O)
```

Where P(A | O) = ∏_t P(a_t | o_t) is the product of per-frame probabilities.

### Forward-Backward Algorithm
We don't enumerate alignments explicitly (exponential). Instead, CTC uses dynamic programming:
- Forward pass: α(t, s) = probability of emitting first s symbols of Y_U by time t
- Backward pass: β(t, s) = probability of emitting remaining symbols from position s starting at time t
- Total probability: Σ_s α(T, s) × β(T, s)

This computes the sum over all alignments in O(T × L) time.

### PyTorch Implementation
```python
import torch.nn.functional as F

# logits: [batch, T, K+1] (decoder output projected to 1001 classes)
# targets: [batch, L] (unit indices 0-999)
# input_lengths: [batch] (all T, or actual lengths if variable)
# target_lengths: [batch] (actual L for each sample)

log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, batch, K+1]
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=1000)
```

