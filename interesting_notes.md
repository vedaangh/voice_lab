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

