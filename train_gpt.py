import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

# mutate_args=() tells pytorch telling PyTorch’s dispatcher and autograd system 
# that this custom operator does not modify any of the tensors you pass in.
@torch.library.custom_op("nanogpt::mm", mutates_args=()) # Registers a new PyTorch operator named "nanogpt::mm".
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile # Wraps so that PyTorch will trace and optimize it into a fused kernel.
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous() 
        # ensures both inputs have contiguous memory layouts, which the low-level kernel requires.
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        # Divide each element by its scale (x_s or w_s).
        # Cast into the 8-bit floating format float8_e4m3fn (4-bit exponent, 3-bit mantissa).
        # This is essentially quantization, we have x ≈ q × s, where x is full precision, q is small
        # low precision quantized number, there is a scaling factor s, we can later require x by multiplying back.
        # there are multiple components, so x_s, w_s are best fits
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32), # Creates a brand‐new tensor of scalar x_s.
            scale_b=x.new_tensor(w_s, dtype=torch.float32), # Creates a brand‐new tensor of scalar w_s.
            use_fast_accum=True,
        )
        # Multiplies x_f8 by the transpose of w_f8, specifies out_dtype=torch.bfloat16 to cast the result into bfloat16.
        # scale_a and scale_b restore the original magnitudes by re-multiplying by x_s and w_s.
        return out, x_f8, w_f8

    return impl(x, w)
    # return the matrix product after quantization, and two quantized multiplier in order

# pure-Python version of your operator that PyTorch can call 
# whenever your high-performance, compiled FP8 kernel isn’t available or can’t be used. 
@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

# This is necessay, otherwise Autograd won’t know how to back-propagate through your custom op
# Gives: RuntimeError: derivative for ‘nanogpt::mm’ not implemented
@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    # g is the incoming gradient ∂L/∂out shape [N, M]
    # returns a tuple (∂L/∂x, ∂L/∂w)
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        # Ensures grad resides in a single, row-major block of memory.
        # The low-level _scaled_mm kernel requires contiguity for correctness and performance.
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32) # same memory format and striding as grad
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)

        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2) # (5-bit exponent, 2-bit mantissa)

        ############################################################################################################
        # Explanation:
        #       The formula: out = x @ W.T
        #           indicates: ∂L/∂x = ∂L/∂out x W
        #           indicates: ∂L/∂W = (∂L/∂out).T x x
        ############################################################################################################
        grad_x = torch._scaled_mm(
            grad_f8,    # [N×M] in FP8, N=batchsize x seq_len, M is the number of output features,
            w_f8.T.contiguous().T,  # [M×D] in FP8, twice-transposed for contiguity, D is the number of input features
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )   # result becomes shape [N, D]
        # Under the hood this computes roughly
        # (∂L/∂out ÷ grad_s) x (W ÷ w_s)^T
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(), # (D × N)
            grad_f8.T.contiguous().T, # (N × M)
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T # Flip to M, D, matching the orientation of original W
        return grad_x, grad_w
        # hint, just check the shape.

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    # it is executed by PyTorch, during the backward pass, 
    # once it reaches the nanogpt::mm node in the autograd graph.
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None # three scale-factor inputs get None.
    # we set materialization=False, so that 
    # PyTorch won’t waste GPU or CPU memory creating unused zeros for those None slots

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    ############################################################################################################
    # Explanation:
    #       ctx: 
    #           a fresh FunctionCtx object that you can use to stash anything you’ll need in the backward pass.
    #       inputs:
    #           the exact Python‐level arguments you originally passed to torch.ops.nanogpt.mm(…)
    #       outputs:
    #           the tuple your forward returned—i.e. (out, x_f8, w_f8)
    ############################################################################################################

    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False) # Not to pre allocate
    # “I promise my backward implementation will only return real gradient tensors for the inputs that 
    # actually need them, and will return None for the rest. You don’t need to pre-allocate 
    # zero-filled tensors for those.""

mm_op.register_autograd(backward, setup_context=setup_context)
# Register so that PyTorch knows that, after running the forward, 
# it should call your setup_context(ctx, inputs, output) to save whatever is needed for backward.
# Later, when it sees this op in the graph during backprop, it should call your backward

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    
    #   Linear / attention projection	(out_features, in_features)	The common 2-D case.
    #   Conv kernel flattened to a matrix before the call	(C_out, C_in × k_h × k_w) Muon flattens 4-D conv weight 
    #   earlier so they still look 2-D here.

    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

        params: list[Tensor] = [*params]
        # creates a new list by unpacking whatever iterable was passed into the function. 
        # the one being passed here is hidden_matrix_params
        param_groups = []
        for size in {p.numel() for p in params}: # numel() returns the product of a tensor'shape
            # notice that this is a set, so every distinct element only appears once
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            ############################################################################################################
            #   Explanation: Allocates a 2-D GPU tensor shaped [world_size, size]
            #       world_size: one row per process/GPU in the distributed job.
            #       size: exactly the number of elements in every parameter that will go into this group.
            #
            #   Process rank R writes its flattened, orthogonalised update into row b[R].
            #   An dist.all_gather_into_tensor call fills all rows so every rank receives the full update matrix.
            ############################################################################################################

            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=b, 
                update_buffer_views=[b[i] for i in range(world_size)]
            )
            ############################################################################################################
            #   Explanation: Allocates a 2-D GPU tensor shaped [world_size, size]
            #       params: 
            #           All those tensors share a common element-count, so they can reuse the same communication buffer
            #           without padding.
            #       update_buffer:
            #           Shared storage for cross-rank gathering of updates.
            #       update_buffer_views:
            #           creates a simple Python list of one-dimensional views
            ############################################################################################################
            
            param_groups.append(group)
            ############################################################################################################
            #   Result: param_groups
            #   [
            #       {
            #         'params': [Wqkv, Wmlp_fc, …],      # all tensors with 589 824 elements
            #         'update_buffer': tensor([...]),    # [world_size, 589 824]
            #         'update_buffer_views': [...]
            #       },
            #       {
            #         'params': [Wmlp_proj, …],          # all tensors with 2 359 296 elements
            #         'update_buffer': tensor([...]),    # [world_size, 2 359 296]
            #         'update_buffer_views': [...]
            #       },
            #       ...
            #   ]
            ############################################################################################################

            # The above buffers created to follow standard optimizer structure and NCCL communication.

        super().__init__(param_groups, defaults)
        # Calling super().__init__(…) lets Muon reuse all that machinery instead of re-implementing it.


    @torch.no_grad()
    def step(self):
        # Looping over groups of paramters of different shapes
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]

            handle = None
            # The asynchronous NCCL request returned by all_gather_into_tensor
            # You later call handle.wait() to block only when you’re ready to use the gathered data.
            params_world = None

            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()  # Block until all gather finished
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(   # in place addition
                        g_world.view_as(p_world),   # reshape flat row coming out of Newton Schultz
                        alpha = - group["lr"] * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5
                        # p_world.size(-2): d_out; p_world.size(-1) d_in, but no less than 1
                    )
                    
            # Looping over parameters of a certain shape
            for base_i in range(len(params))[::self.world_size]:
                # Slice that sequence with a step of self.world_size, for example, if worldsize is 4
                # then we have 0, 4, 8, 12

                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank] # Each rank get a different parameter
                    g = p.grad
                    assert g is not None

                    state = self.state[p] # state is inherited from Optimizer, stores per parameter state tensor
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    # lazily allocates a momentum buffer the first time we see this parameter.
                    # buf is that momentum tensor (same shape & dtype as g).

                    buf.lerp_(g, 1 - group["momentum"])
                    # exponential moving average (EMA) update: buf ← (momentum)·buf + (1−momentum)·g.

                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    # if Nesterov momentum is enabled, compute
                    # at this point gradient is modified as SGDM update

                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                    # gradient orthogonalization
                else:
                    g = update_buffer_views[self.rank]
                    # if this rank had no tensor in the (incomplete) bucket, we just reuse its row view as g
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng

                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                ################################################################################################
                # Explanation: 
                #   1. Copy g to update_buffer[rank]
                #   2. Sends its row update_buffer[rank] to all peers
                #   3. Receives row k from peer k into update_buffer[k]
                #   4. Returns immediately with a Work handle object stored in handle
                ##################################################################################################

                params_world = params[base_i : base_i + self.world_size]
                # Slices the parameter list to capture the exact set of tensors (“bucket”) whose updates are now in 
                # transit. What bucket means: For world_size = 4 and base_i = 8, 
                # the slice is params[8 : 12] → [ W8, W9, W10, W11 ]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))
    # root mean square norm, element wise operation, normalize along the last dimension (d = model dimension)

class CastedLinear(nn.Linear):
    # it inherits the nn.Linear class
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False) # no bias vector
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s
        # x_s, w_s, grad_s: Control how the inputs, weights, and gradients 
        # are quantized into FP8 and then de‐scaled back.

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            # only if in the training we want to use FP8
            _x = x.flatten(0, -2) # starting from 0 and end in the second last (inclusiv)
            out: Tensor = torch.ops.nanogpt.mm( _x, # 2-D input matrix of shape (N, in_features)
                                                self.weight, # 2-D weight matrix of shape (out_features, in_features)
                                                x_s=self.x_s,
                                                w_s=self.w_s,
                                                grad_s=self.grad_s
                                                )[0]
            return out.reshape(*x.shape[:-1], -1) # reshape it
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    # inject time-step awareness into Q & K
    # RoPE (Rotary Positional Embedding):
    # In RoPE, we treate 1D vectors of length head_dim=d as d/2 complex numbers (real, img, real img)
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        # the goal here is to build 2 GPU look up tables
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        # 1D tensor of frequenceies, dim // 4, (half of the pairs rotated and the other half unrotated) 
        # [1.0, ..., (1/1024)^{1/(N-1)}, ..., 1/1024].
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]) # pad with zero
        # angular_freq.new_zero creates a fresh tensor of zeros that inherits 
        # all the “meta” properties of the tensor it is called on, such as dtype and device
        t = torch.arange(max_seq_len, dtype=torch.float32) # token index
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        # compute outer product,
        # theta_{t, k} = t [t] x ω [k], size [L, dim //2]
        self.cos = nn.Buffer(theta.cos(), persistent=False) # Store cos θ in a non-parameter buffer. 
        # persistent=False → the tensor won’t be written to the state-dict
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        # x1 gets all even-index channels, and x2 gets all old-index channels\
        # x1: [B, T, H, d//2], x2: [B, T, H, d//2]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        # Rotation:
        # [cosθ, sinθ   [ RE
        #  -sinθ, cosθ]   IMG ]
        return torch.cat((y1, y2), 3).type_as(x_BTHD) # concatenate wrt the 3rd axis.

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        ############################################################################################################
        #   Explanation:
        #       dim:                model embedding dimension
        #       num_heads:          number of attention heads
        #       max_seq_length:     maximum sequence length for rotary positional encoding
        #       head_dim:           size of each attention head (defaults to 128).
        ############################################################################################################
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        hdim = num_heads * head_dim # Q, K, V should be of shape [Batch_size, Sequence Length, hdim=num_heads×head_dim]
        std = 0.5 * (dim ** -0.5) # 0.5 is a tunable factor
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # If x ~ U(-a, a), uniform distribution, then Var(x) = a^2 / 3, Std(x) = a / sqrt(3)
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        ############################################################################################################
        #   Explanation:
        #       3：                 corresponds to Q, K, V
        #       hdim:               total dimension of projection output
        #                           decounpling head dim from dim
        #       dim:                input embedding dimension
        #       Result:             we are preparing 3 weight matrices W_q, W_k, W_v of [hdim, dim] and we are
        #                           stacking them together
        #       .uniform_:          fills with random values from a uniform distribution
        #                           initializes the Q, K, V weights with a zero-centered, small, uniformly bounded 
        #                           distribution, which helps stablize gradients, balance activation norms
        #
        #   Benefits:
        #           1. reduce memory fragmentation
        #           2. compute QKV in a single matrix multiplication for efficiency
        ############################################################################################################


        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # immediately overwrites every entry with 0, without adding any ops to the autograd tape

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        # “value-embedding” tensor *or* None
        B, T = x.size(0), x.size(1) # batch size, sequence length
        # it is using T as the sequence length, in our notation we use L.
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        # notice that each gpu have a sequence, and the design is optimized for B=1

        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        # Generate Q, K, V in a single fused matrix-multiplication and
        # reshape them into the usual (B, T, H, d) layout for multi-head attention
        # x: (1, T, E);  self.qkv_w : (3, H·d, E); 
        ############################################################################################################
        #   Explanation:
        #       self.qkv_w.flatten(end_dim=1): merge the first two dims, resulting in (3*H*d, E)

        #       .type_as(x): cast to x's type: bf16 when training, fp32 in eval

        #       F.linear: fused projection, merges the original 3 generalized matrix-matrix multiplication
        #       into 1. In our case, x: (1, T, E), W: (3Hd, E) and no bias, so we do x W^T
        #       This allows us to get a result of (1, T, 3Hd), and for every token in the sequence, we get 
        #       [ Q_tokens , K_tokens , V_tokens ].
        
        #       .view(B, T, 3*H, d): immediately reshape it into # (B, T, 3H, d)
        #       .chunk(3, dim=-2): chunk the second last dimesion  # → q, k, v each (B, T, H, d)
        ############################################################################################################

        q, k = norm(q), norm(k) # QK norm @Grad62304977
        # the attention score is computed using normalized q, k with fixed magnitude, 
        #  the authors can use a single learnt or fixed scale (scale=0.12 later in the Flex-Attention call) instead of the canonical 1 / √d

        
        # (B, T, H, d); (B, T, H, d)
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=0.12).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side 
        # (B, T, E)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng ---> Mimicing a UNet structure
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.])) # trainable interpolation parameters

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

# Equivalent to ceil[v / n] x n
# * in signature ensures that n must be passed by names n=128
def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)
    # range(n, int(v) + 1 + n, n) gives n, 2n, 3n, ... up to at least v
    # next() Retrieves the first value the generator yields


############################################################################################################
#   Explanation:
#       Subclassing torch.nn.Module:
#           By inheriting from nn.Module, all layers and parameters (e.g., nn.Linear, nn.Embedding) 
#           registered in the constructor are automatically tracked, stored in model.parameters(), 
#           properly moved to GPU with .cuda() or .to(device), saved/loaded via torch.save() and torch.load().
############################################################################################################
class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        # Initializes the parent class nn.Module.
        self.embed = nn.Embedding(vocab_size, model_dim)
        # (Implementation): Embedding layer is done using efficient look up. The input is a bunch of id, and the  
        # embedding layer is like a matrix of size [vocab_size, model_dim], where model_dim is also the dimension of 
        # embedding. The output is the corresponding row of the matrix to the token id.

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        # The idea is to inject multiple "value embeddings" into the transformer layers, which 
        # enrich the input with semantic content. They are used selectively in certain layers to help with skip 
        # connections and richer representations.
        # nn.ModuleList([...]) is important, because it ensures each embedding layer is treated as a proper nn.Module
        # Registered: .to(device), .cuda(), .parameters(), .state_dict(), .load_state_dict()
        # If using a normal Python list, PyTorch would ignore these modules during training and saving.

        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers // 2))
        # This makes the skip weights trainable

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128 # the whole 48k tokens sequence is reshaped into blocks of 128 tokens, B = 384
        docs = (input_seq == 50256).cumsum(0) # document boundaries returns true, cumsum gives mannual indices

        def document_causal(b, h, q_idx, kv_idx):
            # q_idx, kv_idx are absolute token indices, plus batch, head slots (unused)
            causal_mask = q_idx >= kv_idx # Only look at casual blocks (not in future)
            document_mask = docs[q_idx] == docs[kv_idx] # both tokens belong to the same document (docs[...] equal).
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            # Input: B x B matrix, boolean
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32) # cheaper than int64
            # sum(dim=-1) reduces along the last axis (columns), 
            # num_blocks[q] tells how many KV blocks row q must later look at.
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            # Stage 1: argsort(dim=-1, descending=False, stable=True) returns indices that would sort each row 
            # ascending (stable) along the last dimension, Shape (B, B), default dtype int64
            # For each row q you now have a permutation of 0…B-1. All the True columns are clustered at the right end
            # Stage 2: flip(-1): Reverses the elements within each row (last dimension). 
            # After reversing, the indices of True columns move to the left side of the row, in descending column
            # order. This is handy because subsequent kernels will read just the first num_blocks[q] entries of
            # each row — those now correspond exactly to the allowed KV blocks for that query row.
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
            # nums_blocks[None, None] adds two singleton dimensions at the front (unsqueeze twice), shape (1, 1, B)
            # indices: shape (1, 1, B, B)
            ############################################################################################################
            #   Explanation:
            #       num_blocks[q] tells the Flex-Attention kernel how many KV tiles (columns) should be fetched 
            #       for query block q. 
            
            #       indices[q, 0 : num_blocks[q]] tells which tiles (their column indices), already ordered so 
            #       the kernel can stream them without another sort.

            #       The extra leading (1,1, …) dims allow a single mask tensor to be broadcast 
            #       across batch and head dimensions without copying.
            ############################################################################################################

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0 # ensure that we do it without padding
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE # number of blocks
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda") # block index list shape (B,)
        # block_idx[:, None] turns it into (B, 1), we compare (B, 1) and (1, B) broadcasted resuly is (B, B)
        # By default row is query block and column is key block
        causal_blockmask_any = block_idx[:, None] >= block_idx # Lower triangular
        #         q≥k   k=0  1  2  3        Meaning
        #         ───────────────────────────────
        #         q=0   T  F  F  F   ← block 0 may look at 0 (its own past)
        #         q=1   T  T  F  F   ← block 1 may look at blocks 0 & 1
        #         q=2   T  T  T  F
        #         q=3   T  T  T  T
        causal_blockmask_all = block_idx[:, None] > block_idx
        #         q>k   k=0  1  2  3        Meaning
        #         ───────────────────────────────
        #         q=0   F  F  F  F   ← block 0 has no strictly earlier block
        #         q=1   T  F  F  F   ← every token of block 0 is < every token of block 1
        #         q=2   T  T  F  F
        #         q=3   T  T  T  F
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        # .view(-1, BLOCK_SIZE) reshape it into [B, 128], without copying (it just adjusts strides)
        # [:, 0] takes the index of the first token in each block, as a result it is shape (B,)
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        ############################################################################################################
        #   Explanation:
        #       contiguous():
        #           If the tensor is already stored in one tight, row-major block of memory (tensor.is_contiguous() == 
        #           True), contiguous() is a no-op: it returns the same storage, no data move.
        #
        #           Transpose, slice, or flip operations create views whose elements live at “skipped” 
        #           addresses (non-unit strides). Calling .contiguous() allocates fresh memory and copies 
        #           the data into that new, gap-free layout, so later kernels can treat the tensor as a regular 
        #           dense array.
        ############################################################################################################

        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        # docs_low[:, None] becomes [B, 1]
        # The earliest document seen in block q is no later than the latest document in block k
        # The latest document in block q is no earlier than the earliest document in block k
        # Overall the result is there exists at least one token pair (q-token, k-token) 
        # that belongs to the same document. Result: shape (B, B)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        # every token of both blocks lies in the same single document Result: shape (B, B)

        # Combine causal and document rules
        blockmask_any = causal_blockmask_any & document_blockmask_any
        # shape (B, B): blockmask_any[i,k]==1 means “some subset of block k is visible to queries in block i.”*
        blockmask_all = causal_blockmask_all & document_blockmask_all
        # shape (B, B): blockmask_all[i,k]==1 means “all 128 tokens of block k are visible to block i.”

        # The previous BxB = 147654 (150 k)
        # ~: to invert True and False, basically, we want to exclude the those blocks whose tokens are all visable
        # in a word, there is some legal attention (any), but not the whole tile (all is False)
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        # full_kv_num_blocks[q] – how many completely legal KV blocks (whole 128 × 128 tile safe)
        # full_kv_indices[q, ·] – their column indices
        # partial_kv_num_blocks[q] – how many partially legal KV blocks (need a causal/doc prefix mask)
        # partial_kv_indices[q, ·] – their column indices
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(
                    partial_kv_num_blocks,
                    torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)
                ), # (1, 1, B) allowed count
                partial_kv_indices, # (1, 1, B, B) allowed indices
                torch.clamp_max(
                    full_kv_num_blocks, 
                    window_size_blocks - 1
                ), # (1, 1, B) allowed count, reserved 1 for itself
                full_kv_indices, # (1, 1, B, B) allowed indices
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
                # When you pass a mask_mod callable:
                #   1. Full tiles (those in full_kv_*) are unaffected; the kernel uses the whole 128 × 128 KV tile.
                #   2. Partial tiles (those in partial_kv_*) need a prefix mask, 
                #      because only some tokens of the tile are legal.
                #           For each query token in the block, the kernel evaluates
                #           keep = mask_mod(b, h, q_abs_idx, kv_abs_idx) to decide 
                #           whether to include the (q, k) pair in the soft-max accumulation.
                #
                #       Using a callable instead of a pre-computed dense mask keeps the BlockMask tiny
                #        (O(B·W) metadata) while still giving the CUDA code an exact per-token rule.
            )
        ############################################################################################################
        #   Explanation:
        #       1st torch.clamp_max: 
        #           partial_kv_num_blocks: (1, 1, B)
        #               How many partially visible KV-blocks each 
        #               query-block could use if there were no window limit.
        #           full_kv_num_blocks: (1, 1, B)
        #               How many fully visible KV-blocks are already available to each query-block.
        #           window_size_blocks: scalar (e.g. 1792 tokens ÷ 128)
        #           
        #           window_size_blocks - full_kv_num_blocks: (1, 1, B)
        #               equivalent to remaining[q] = window_size_blocks - full_kv_num_blocks[q]
        #               remaining budget for “partial” tiles, can go below 0 if the  already filled by full tiles
        #           torch.clamp_min(…, 1): (1, 1, B)
        #               returns max(1, x), forces remaining[q] to be 1 since we need to keep query's own block
        #           torch.clamp_max(partial_kv_num_blocks, remaining): (1, 1, B)
        #               returns min(x, y) elementwisely.
        ############################################################################################################

        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)
        # Long mask: span = W blocks
        # Short mask: span = W / 2 blocks
        # Pattern L-S-S-S-L-S-S-L-S-S-S-L
        #
        # About the mask
        # Rows kept:            all queries (nothing is pruned on that axis).
        # Columns kept:         only K/V tokens that are
        #                       > within the sliding window of size W blocks,
        #                       > not in the future
        #                       > in the same document
        # Everything else is treated as -∞, i.e. receives zero probability after the soft-max.

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1
        # assuring that input_seq has shape (T, ), where T is the sequence length
        # notice we are using a micro batch of size 1 on 1 gpu
        # the overall "batch" size equals the number of GPU


        ve = [value_embed(input_seq) for value_embed in self.value_embeds]  # Recall that this is a look up table
        # the output would be [T, model_dim] for each element in the list


        # 012 ... 012 structure for injecting token value embeddings into selected transformer blocks
        # as part of a U-Net-like skip connection structure within the GPT model.
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)
        ############################################################################################################
        #   Explanation:
        #       [ve[0], ve[1], ve[2]]            # for first 3 blocks (blocks 0, 1, 2)
        #       [None] * (12 - 6) = [None]*6     # skip 6 middle blocks (blocks 3–8)
        #       [ve[0], ve[1], ve[2]]            # for last 3 blocks (blocks 9, 10, 11)
        ############################################################################################################

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        # long_bm, short_bm: Blockmask object
        ############################################################################################################
        #   Explanation:
        #       Blockmask object: four integer tensors plus two scalars: very small, GPU-resident struct.
        #       # 1. “partial” KV tiles (need inside-tile prefix mask)
        #           partial_kv_num_blocks : int32[1,1,B]   # how many partial blocks per query-block
        #           partial_kv_indices    : int32[1,1,B, ≤W]# their column indices (padded)

        #       # 2. “full” KV tiles (whole 128×128 tile is legal)
        #           full_kv_num_blocks    : int32[1,1,B]   # how many full blocks per query-block
        #           full_kv_indices       : int32[1,1,B, ≤W]# their column indices (padded)

        #       BLOCK_SIZE = 128                       # tokens per block (compile-time const)
        #       mask_mod   = <function document_causal># callback for inside-tile masking
        ############################################################################################################
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        # L-S-S-S-L-S-S-L-S-S-S-L, short block masks indicate fewer K/V tiles per query, and lower FLOPS, memory.
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977
        # self.embed turns input seq from (L,) to (L, d)
        # [None] is equivalent to .unsqueeze(0), creating the batch dim with batchsize=1: (1, L, d)
        # norm(): normalize using RMS norm

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights) # how many such skips you are going to need
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
                # 6 linked with output of 5, 7 linked with output of 4, ..., pop remove and returns the last tensor

            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            # x0 is the original token embedding,  provides a global, per-token skip available
            # in every layer, reminiscent of UNet’s “constant resolution” pathway.
            if i < n:
                skip_connections.append(x)

        x = norm(x) # (1, L, E) (E is the model dimension, hdim * num_heads)
        # running hidden state can still accumulate magnitude drift through the residual paths
        # A final normalisation rescales every token vector to unit‐RMS, 
        # giving the next layer a consistent input distribution.
        logits = self.lm_head(x).float() # (1, L, V) in bfloat 16, V stands for vocabulary size.
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        # x.size(-1) gives model dimension E, and then we take the square root
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')
        # logits original shape (1, L, V), we flatten the 1,T dim ==> each row is vocabulary logit for 1 token
        # target_seq: vocabulary token ID, should be (L, )
        # reduction=sum adds the loss, so the gradient ~ L
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    # shard refers to something like fineweb_train_000001.bin
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    # Read the 256-int32 “header” at the very start of the file
    # torch.from_file maps the first 256 32-bit integers straight off disk into a tensor called `header`.  
    # `header[0]` is a “magic number” to verify you’re looking at the right file format; `header[1]` is a version.  

    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)

    # "rb", no buffering mode
    with file.open("rb", buffering=0) as f:
        # Allocate a pinned-memory uint16 tensor to hold all tokens
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        # pin_memory=True means this buffer can be DMA-transferred to the GPU without an extra copy
        f.seek(256 * 4)
        # Skip past the 256×4-byte header, 

        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        # reads raw bytes straight into the NumPy array backing your tensor
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
        # uint16
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    # List all matching files on disk, sorted to keep a stable order.
    # Wrap each path in a Path(...) for easy file operations.
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size 
    # simply how many tokens each individual 
    # GPU (or process) handles on every step.
    # Each GPU's share
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    # Turn your file list into an iterator, so you can cycle through shards.

    tokens, pos = _load_data_shard(next(file_iter)), 0
    # 1-D torch.Tensor of shape [num_tokens] containing your token ID
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0 # directly drop the rest
            # we need batch_size + 1 tokens each time (so we can form inputs of 
            # length batch_size and targets shifted by one).
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        # GPU 0 reads from pos + 0 * L, GPU 1 from pos + 1 * L, then after slicing, we
        # then take the first local_batch_size + 1
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        # we do not train on sliding window, since it would be redundant

        yield inputs, targets
        # turns the surrounding function into a generator
        # returns (inputs, targets) back to whatever code 
        # called next(train_loader) (or iterated over train_loader).
        # After yielding, the function’s internal state (all local variables, the current pos, etc.) is frozen.
        # Control returns to the caller.
        # The next time you call next(train_loader), execution resumes 
        # immediately after the yield, picking up the while True: loop for the next batch.

# -----------------------------------------------------------------------------
# int main

if __name__ == "__main__":

    ############################################################################################################
    #   Explanation:
    #       @dataclass:
    #           A decorator automatically generates boilerplate code for a class, specifically to make it behave
    #           like a data container. Python will automatically generate: __init__() constructor,
    #           __repr__() for nice printing, __eq__() for equality comparisonsm and ordering if requested.
    ############################################################################################################
    @dataclass 
    class Hyperparameters:
        # data
        train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
        val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
        val_tokens = 10485760 
        # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
        train_seq_len = 48 * 1024 # FlexAttention sequence length
        val_seq_len = 4 * 64 * 1024 # FlexAttention sequence length for validation
        # optimization
        num_iterations = 1770 # number of iterations to run
        cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
        # cooling down is another way to say learning rate decay
        # architecture
        vocab_size = 50257
        # evaluation and logging
        val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
        save_checkpoint = False
    args = Hyperparameters()


    ############################################################################################################
    #   Explanation:
    #       When training with torchrun, it launches multiple processes, and each process executes the same 
    #       Python script
    ############################################################################################################
    # torchrun sets these env variables
    rank = int(os.environ["RANK"]) # unique ID of the current process
    world_size = int(os.environ["WORLD_SIZE"]) # total number of processes in training
    # uncomment if we are using 8 GPUs
    # assert world_size == 8 # this code is designed for 8 x H100
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    # This sets the "default" GPU device for all subsequent PyTorch CUDA operations in the current process.
    
    dist.init_process_group(backend="nccl", device_id=device)
    ############################################################################################################
    #   Explanation:
    #       "nccl": 
    #           "nccl" is highly optimized for NVIDIA GPUs, and is the standard for multi-GPU training.
    #
    #       device_id=device:
    #           Tells PyTorch which GPU device this process should use.
    #           This argument is sometimes optional and PyTorch can infer it from torch.cuda.set_device().
    ############################################################################################################
    dist.barrier()
    # Forces all processes to wait here until every other process reaches the same point.
    master_process = (rank == 0) # this process will do logging, checkpointing etc.
    # Sets a flag: master_process will be True only for the process with rank == 0.

    # begin logging
    logfile = None
    if master_process: # Only the master handles the logging
        run_id = uuid.uuid4() # generates a randomly-generated UUID, ensures log files don’t overwrite each other
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)

    def print0(s, console=False): # if console=True, also prints to terminal
        if master_process:
            with open(logfile, "a") as f: # append mode, every call adds a line
                if console:
                    print(s)
                print(s, file=f)

    # begin by printing this file (the Python code)
    print0(code)
    print0("=" * 100)
    # log information about the hardware/software environment this is running on
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

    def nvidia_smi():
        # Uses subprocess to run the command and capture its output as a string.
        import subprocess  # avoid top level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi())
    print0("=" * 100)

    ########################################
    #    Construct model and optimizer     #
    ########################################

    model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
                        max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    # convert the parameters of embedding layers to bfloat16 (memory efficiency and faster compute on H100)
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    # copies the tensor from the process with rank 0 to all other processes in the distributed group
    # need to use .detach(), to prevent it from interfering with autograd
    ############################################################################################################
    #   An overview of the model structure:
    #     GPT(...)                  # the model root
    #     ├── Embedding(...)
    #     ├── ModuleList([Embedding, Embedding, Embedding])
    #     │   ├── Embedding(...)
    #     │   ├── Embedding(...)
    #     │   └── Embedding(...)
    #     ├── ModuleList([... 12 × Block ...])
    #     │   ├── Block(...)
    #     │   │   ├── CausalSelfAttention(...)
    #     │   │   │   ├── CastedLinear(qkv)
    #     │   │   │   └── CastedLinear(proj)
    #     │   │   └── MLP(...)
    #     │   │       ├── CastedLinear(fc)
    #     │   │       └── CastedLinear(proj)
    #     │   └── … (other Blocks)
    #     └── CastedLinear(lm_head)
    ############################################################################################################

    # dividing the params into several lists
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    # gradient orthogonalisation requires p.ndim >= 2, and we do not want to touch the embedding matrix
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    # token embedding, value embeddings
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # init the optimizer(s)
    adam_params = [
        dict(params=head_params, lr=0.22), # language model head with Adam
        dict(params=embed_params, lr=0.6), # embedding use Adam
        dict(params=scalar_params, lr=0.04) # layer norm weights, bias, skip scalars, use Adam
    ]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)

    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
            # record initial lr rate

    # learning rate schedule: stable then decay
    def get_lr(step: int):
        x = step / args.num_iterations # progress in training
        assert 0 <= x < 1
        if x < 1 - args.cooldown_frac:
            return 1.0
        else:
            w = (1 - x) / args.cooldown_frac
            return w * 1.0 + (1 - w) * 0.1 # producing a linear decay

    # attention window size schedule: linearly increase
    @lru_cache(1)
    def get_window_size_blocks_helper(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    ############################################################################################################
    #   Explanation:
    #       lru:
    #           Wraps the function with a memo-table that keeps the return value for the most recent call. 
    #           Subsequent calls with the same argument skip execution and return the cached object immediately.
    #       maxsize=1: 
    #           Only one entry is stored at a time, so memory never grows beyond a single tensor. When a new window_size
    #           is requested, the old tensor is evicted and replaced.
    ############################################################################################################
    
    def get_window_size_blocks(step: int):
        x = step / args.num_iterations # progress in training
        assert 0 <= x <= 1
        # Linearly increase the block-wise sliding window size over training 128 -> 1792
        # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
        window_size = next_multiple_of_n(1728 * x, n=128) # ceil[1728 * x / 128] x 128
        # as x increases: window_size gradually becomes 128, 258 ..., 1792
        return get_window_size_blocks_helper(window_size) # Convert token count to block count

    # Flash attention and Flex attention
    model: nn.Module = torch.compile(model, dynamic=False)
    # The result is a wrapped module that behaves identically to the original but runs its forward 
    # and backward passes through those fused kernels.
    # dynamic=False: Assume input tensor shapes do not change across iterations; compile a single static graph

    ########################################
    #            Warmup kernels            #
    ########################################

    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    warmup_steps = 10
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
    
    # make the just-compiled model and the two optimizers generate (and cache) their
    # CUDA kernels before real training starts
    for _ in range(warmup_steps):
        inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
        # [0, args.vocab_size), 
        model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    

    # Reload the snapshots 
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state

    ########################################
    #      Overlap Communication Setup     #
    ########################################

    ############################################################################################################
    #   Explanation:
    #       We “bucket” parameters here so that when it comes time 
    #       to all-reduce (i.e. average) gradients across GPUs:
    #           1. Overlap communication with computation. By grouping many small tensors into
    #            a few medium‐sized buckets (∼25 MB each), we issue fewer large NCCL calls instead 
    #            of hundreds of tiny ones. That cuts latency and lets us start the 
    #            all-reduce for one bucket while still computing gradients for the next.
    ############################################################################################################

    # Create parameter buckets for better overlap
    def create_buckets(params, bucket_size_mb=25):
        """Group parameters into buckets of approximately bucket_size_mb MB each"""
        buckets = []
        current_bucket = []
        current_size = 0

        # Sort parameters by size (largest first) for better bucketing, descending order
        sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)

        for param in sorted_params:
            param_size_mb = param.numel() * param.element_size() / (1024 * 1024)
            # size in MB, the original product is size in byte

            if current_size + param_size_mb > bucket_size_mb and current_bucket:
                # over 25 MB, close out and start a new bucket
                buckets.append(current_bucket)
                current_bucket = [param]
                current_size = param_size_mb
            else:
                # else just append
                current_bucket.append(param)
                current_size += param_size_mb

        if current_bucket:
            # if there are any partially filled bucket
            buckets.append(current_bucket)

        return buckets

    # Create buckets for all parameters
    all_params = [p for p in model.parameters() if p.requires_grad] 
    # every parameter tensor in the model that participates in training
    param_buckets = create_buckets(all_params)
    # Once we have param_buckets, later we register a gradient‐ready hook on each parameter; 
    # when all gradients in a bucket are ready, we fire a single all-reduce on that bucket, 
    # rather than one per parameter.

    print0(f"Created {len(param_buckets)} gradient buckets")
    for i, bucket in enumerate(param_buckets):
        total_size = sum(p.numel() * p.element_size() for p in bucket) / (1024 * 1024)
        print0(f"Bucket {i}: {len(bucket)} params, {total_size:.1f} MB")
    # Recording bucket size

    # Bucket state tracking
    bucket_ready_count = [0] * len(param_buckets)
    # When bucket_ready_count[b] reaches the number of parameters in bucket b, all gradients are ready
    bucket_handles = [None] * len(param_buckets)
    # used to store the returned handle of all_reduce
    param_to_bucket = {}
    # maps each parameter tensor object to the index of the bucket

    # Map each parameter to its bucket index
    for bucket_idx, bucket in enumerate(param_buckets):
        for param in bucket:
            param_to_bucket[param] = bucket_idx

    def _gradient_hook(param: Tensor):
        """Called when a parameter's gradient is ready"""
        # “post‐accumulate” hook on every trainable parameter
        # so it runs each time PyTorch finishes computing a .grad for one parameter.
        # track when all gradients in a given bucket are ready, and 
        # then kick off a single asynchronous all-reduce on that entire bucket
        if param.grad is None:
            return

        bucket_idx = param_to_bucket[param]
        bucket_ready_count[bucket_idx] += 1

        # Check if all parameters in this bucket are ready
        if bucket_ready_count[bucket_idx] == len(param_buckets[bucket_idx]):
            # All-reduce this bucket
            bucket_grads = [p.grad for p in param_buckets[bucket_idx]] # list of gradients

            # For multi-tensor operations, we can reduce them together
            if len(bucket_grads) == 1:
                handle = dist.all_reduce(bucket_grads[0], op=dist.ReduceOp.AVG, async_op=True)
            else:
                # Use multi-tensor all-reduce for efficiency
                handle = dist.all_reduce_coalesced(bucket_grads, op=dist.ReduceOp.AVG, async_op=True)
                # packs them into a single NCCL call for efficiency
            # In both cases, async_op=True means the call is non-blocking; 
            # it returns immediately with a “work handle.”

            bucket_handles[bucket_idx] = handle

    # Register hooks for all parameters
    print0("Registering bucketed gradient hooks...")
    for param in all_params:
        param.register_post_accumulate_grad_hook(_gradient_hook)
    # Under the hood, as soon as PyTorch finishes accumulating that parameter’s gradient tensor 
    # (i.e. sets or adds into param.grad), it invokes _gradient_hook(param).

    def wait_for_gradients():
        """Wait for all gradient reductions to complete and reset bucket state"""
        for handle in bucket_handles:
            if handle is not None:
                handle.wait()
        # Ensure all asynchronous gradient reductions have finished

        # Reset state for next iteration
        for i in range(len(bucket_ready_count)):
            bucket_ready_count[i] = 0
            bucket_handles[i] = None

        
    ########################################
    #        Training and validation       #
    ########################################

    train_loader = distributed_data_generator(
        args.train_files, 
        world_size * args.train_seq_len, 
        rank,
        world_size
    )
    # setting up your infinite training-data iterator, sharded across GPUs
    # args.train_files is the glob pattern for your on-disk shard files.
    # batch_size = world_size * args.train_seq_len is the global batch size in tokens.
    # rank and world_size tell each process which slice of that global batch it should actually load.
    # Every time you call next(train_loader), you get a pair (inputs, targets) of length 
    # local_batch_size = batch_size // world_size = args.train_seq_len; 
    # each process sees a non-overlapping window of the data, so together they cover the full global batch.

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    # ensures that any preceding GPU work (e.g. your warmup kernel launches) has completed before we start timing.
    t0 = time.perf_counter() # records the current (wall-clock) time in seconds with high precision.
    # begin training
    train_steps = args.num_iterations
    for step in range(train_steps + 1):
        last_step = (step == train_steps)

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # measure validation loss periodically and also include the training time up to that point.
            
            torch.cuda.synchronize() # stop the clock
            training_time_ms += 1000 * (time.perf_counter() - t0) # add the elapsed time (in ms) to training_time_ms.
            model.eval()

            val_batch_size = world_size * args.val_seq_len # total tokens per validation batch across all GPUs.
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size 
            # how many batches we need to process to cover exactly val_tokens in total

            val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, get_window_size_blocks(step))
            val_loss /= val_steps
            del val_loader
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(step)).backward()
        # As each grad is ready, your _gradient_hook fires and 
        # kicks off bucketed all‐reduce communications under the hood.
        #for param in model.parameters():
        #    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        wait_for_gradients() # does the same thing as commented two lines above, but faster
        

        # set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers
        for opt in optimizers:
            opt.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    # Logs the peak and currently reserved GPU memory (in MiB).
    dist.destroy_process_group() # cleanly shuts down your NCCL communication context.
