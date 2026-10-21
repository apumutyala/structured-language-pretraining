# Fixed-input RECORD retention diagnostic

More training could improve this model's task behavior. The fixed contraction
ceiling does constrain one route for information: the influence of an incoming
recurrent state across a fixed future input sequence. It does not establish
that this task cannot be learned or that the external interface is inherently
inferior. A separate training-budget comparison must answer whether additional
updates actually help. This diagnostic uses only the saved 400-update weights;
it neither observes nor evaluates the separate longer-budget experiment.

## Exact fixed-input bound

The implementations in `invariant_ssm.py` and `parallel_ssm.py` use the same
diagonal affine recurrence in real arithmetic. At valid token t, write

```text
a_t = rho * sigmoid(W_a x_t + b_a)
c_t = tanh(W_c x_t + b_c)
h_t = a_t * h_(t-1) + (1-a_t) * c_t
```

Multiplication is coordinatewise. The gate and candidate depend on the current
input, not the previous hidden state. For a padding position, the effective
coefficient is exactly 1 and the additive term is 0. Define `d_t=a_t` for valid
positions and `d_t=1` for padding. Holding the inputs and parameters fixed gives

```text
dh_t / dh_(t-1) = Diag(d_t)
J_(T:s) = dh_T / dh_s = Diag(product over t=s+1,...,T of d_t)
```

This is an equality, rather than a bound. For every induced l_p matrix norm,
`1 <= p <= infinity`, including spectral norm at p=2,

```text
||J_(T:s)||_p = max_j product_t d_(t,j) <= rho^N
```

Here `N` is the number of **valid** positions after s, not padded sequence
length. The minimum singular value is the minimum coordinate product. For
finite real logits and `N>0`, each valid coefficient lies strictly between 0
and rho, so the inequality is strict. The non-strict ceiling is the useful
uniform statement, including the identity case `N=0`. There is no uniform
positive lower bound: gates can approach zero. This result holds for arbitrary
finite incoming hidden states; the separate invariant-cube guarantee requires
an incoming state in `[-1,1]^H`.

Equivalently, two different incoming states driven by the same future inputs
satisfy `delta_h_T = J_(T:s) delta_h_s`. The parallel prefix scan reassociates
the affine products and sums but does not alter this real-arithmetic equation.
It does change floating-point association. These equations are not a formal
PyTorch, CUDA, autograd, or IEEE numerical refinement proof.

At the architectural `rho=0.99`, the best possible uniform ceiling is about
0.89534 after 11 valid tokens, 7.59e-6 after 1,173, and 7.00e-6 after 1,181.
Training the gate cannot increase a real coefficient beyond rho. Training can
change which coordinates approach that ceiling, the injected candidate values,
input embeddings, typed projection, and the output readout.

## Bounded CPU measurement

Before measuring coefficients, the selection was fixed to seed `2026102141`,
both saved 400-update arms, initial observations at IID manifest indices
`0,1,2`. Selection uses manifest order, with no outcome filter, oracle action,
rollout selection, or choice among seeds. The tool freezes the selection and
observation hash in its output before loading either checkpoint. It verifies
checkpoint, manifest, identity/config and source digests; loads with
`map_location="cpu", weights_only=True`; and performs no training, optimizer
or GPU work.

The internal input is the 11-byte query with a learned typed-state projection
added to every token. The external input is the complete canonical JSON
observation, a newline, and the same repeated 11-byte question. Every ASCII
byte is one valid token. These individual observations have no padding.

FP32 checkpoint inputs and gate projections supply the measured coefficients.
For each hidden coordinate the tool accumulates `sum_t log(a_(t,j))` in FP64;
this avoids losing the product merely because direct multiplication underflows.
Natural-log minimum/maximum refer to the smallest/largest coordinate products.

| Arm | Initial world | Valid tokens | Minimum log retention | Maximum log retention | Largest coordinate product | rho^N ceiling |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| internal | iid.0 | 11 | -9.59280 | -0.15716 | 0.85457 | 0.89534 |
| internal | iid.1 | 11 | -8.83945 | -0.16503 | 0.84787 | 0.89534 |
| internal | iid.2 | 11 | -8.54340 | -0.15707 | 0.85465 | 0.89534 |
| external | iid.0 | 1,173 | -783.55934 | -15.35744 | 2.13968e-7 | 7.58723e-6 |
| external | iid.1 | 1,179 | -786.71618 | -15.45701 | 1.93690e-7 | 7.14323e-6 |
| external | iid.2 | 1,181 | -788.62561 | -15.50119 | 1.85319e-7 | 7.00108e-6 |

The external JSON components are 1,161, 1,167 and 1,169 bytes respectively;
the newline and repeated question account for the remaining 12 tokens.
The FP32 representation of rho is `0.9900000095367432`; the tool records both
that value and architectural 0.99. It computes the theoretical ceilings from
0.99. This small difference is a reminder that the exact real inequality is
not an asserted floating-point theorem. Measured products are comfortably
below both ceilings here. The smallest external coordinate products underflow
when exponentiated even in FP64; their finite logs are retained in the result.

The result is specifically sensitivity to a state **before the observation**.
It is not the retention of every fact anywhere in that observation. For the
external arm, the maximum log products over only the final 11-byte question
suffix are -0.14952, -0.14711 and -0.14571: incoming-state norms about
0.8611–0.8644. The tool also reports fixed 1-, 128- and 512-token suffix
horizons where applicable. A fact's influence begins when its bytes affect a
gate/candidate, and subsequent relevant bytes can inject information again.
The entire JSON is reread on each operation, and the internal arm also injects
the current workspace anew on every query token. Neither arm depends solely
on carrying the previous operation's hidden state across the whole input.

The arm-specific checkpoints have different learned parameters and inputs.
Their measured coefficient differences do not isolate sequence length as a
cause of task failure. This is six descriptive measurements from one seed,
not a performance test, confidence interval or architecture comparison.

## What the bound does not say about gradients and learning

The fixed-input state Jacobian is only one term in end-to-end differentiation.
For a parameter theta, suppressing padding notation,

```text
dh_T/dtheta = J_(T:0) dh_0/dtheta
             + sum_t J_(T:t) [ (h_(t-1)-c_t) * da_t/dtheta
                               + (1-a_t) * dc_t/dtheta ]
```

Local derivatives include parameter-dependent inputs where appropriate. Recent
injections have short suffixes; the sum is not multiplied uniformly by
`rho^T`. The same distinction applies to input sensitivity. In the invariant
cube, a local induced-norm bound at a valid position is

```text
||dh_t/dx_t||_p <= (rho/2) ||W_a||_p + ||W_c||_p
```

because `|h_(t-1)-c_t|<=2`, `sigmoid'<=1/4`, and `|tanh'|<=1`.
The architecture supplies no fixed norm cap on these learned matrices. Nor
does this state-retention result bound embedding/projection gradients,
SwiGLU/residual transforms, learned output-head weight norms, logits or loss
gradients. The readout uses parameter-free RMS with positive epsilon, but
normalization alone supplies no architecture-wide small Jacobian after
unconstrained learned linear layers. A fixed finite checkpoint has local
readout derivatives; they were not measured or bounded here. Consequently,
`d logits/dh_0 = (d logits/dh_T) J_(T:0)` does not inherit a stand-alone
`rho^T` whole-model bound.

When changed model actions lead to different later observations, the fixed-input
Jacobian also does not describe that complete closed-loop mapping.

Long fixed-input state paths can attenuate both state differences and their
direct backpropagated signal. That makes length/placement a concrete diagnostic
and experiment target. It does not prove insufficient task information,
vanishing gradients for every parameter, optimization failure for every run,
that more updates cannot help, or that an alternate model is necessary.
Additional training may learn better input injection, parsing, finite-task
features and readout despite this ceiling. A controlled budget extension
measures that possibility; this diagnostic alone cannot answer it.

## Reproduction and evidence

Run using the existing CPU environment:

```text
.venv/Scripts/python.exe tools/measure_record_retention.py
```

The default receipt is [record_retention_20261021.json](../results/record_retention_20261021.json).
The tool refuses to overwrite an existing receipt; use `--output` for a fresh
reproduction file. The recorded run used CPU PyTorch `2.14.1+cpu`, FP32
coefficients, FP64 log accumulation and two CPU threads; measurement/loading
took about 1.26 seconds. Checkpoint and source hashes appear in the receipt;
no absolute local paths or observation text are included. Ordinary, empty
(identity), and zero-coefficient product summaries were separately checked
against hand-computed values. No optimizer, GPU or existing source/proof was
modified, and no model-quality or semantic-alignment claim follows.
