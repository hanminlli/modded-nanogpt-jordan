## all_reduce():

Imagine 4 GPUs, each holding a 2-D gradient matrix Gr, r= 1,2,3,4 (same shape).
```
    dist.all_reduce(G, op=dist.ReduceOp.SUM)   # or .AVG
```

After the call:
```
    GPU0: G0 ← G0+G1+G2+G3
    GPU1: G1 ← G0+G1+G2+G3
    GPU2: G2 ← G0+G1+G2+G3
    GPU3: G3 ← G0+G1+G2+G3
```

If we choose op=AVG, then each rank-row now holds the same mean gradient.



## all_gather():

Each GPU starts with a different tensor slice:
```
    # rank 0: x = [A]
    # rank 1: x = [B]
    # rank 2: x = [C]
    # rank 3: x = [D]
    xs = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(xs, x)
```
After the call every rank’s xs list contains [A, B, C, D].


## Faster variant: all_gather_into_tensor():
If every slice has the same length you can pre-allocate a 2-D buffer and let the backend fill rows:
```
    buf = torch.empty(world_size, slice_len, device="cuda")
    dist.all_gather_into_tensor(buf, x)   # x is your local 1-D slice
    # buf[row] now holds the slice from that row’s rank
```
That’s the API Muon uses — it avoids Python lists and small copies.


## reduce_scatter(): — the “mirror-image” of all_gather():
| Rank | Local 1-D tensor `xᵣ` (length = 4) |
| ---- | ---------------------------------- |
| 0    | `[A₀, A₁, A₂, A₃]`                 |
| 1    | `[B₀, B₁, B₂, B₃]`                 |
| 2    | `[C₀, C₁, C₂, C₃]`                 |
| 3    | `[D₀, D₁, D₂, D₃]`                 |

Call
```
    # one output slice per rank
    out = torch.empty(4, device="cuda")
    dist.reduce_scatter(out, x, op=dist.ReduceOp.SUM)
```

1. Reduce step: 
```
    [A₀+B₀+C₀+D₀,  A₁+B₁+C₁+D₁,  A₂+B₂+C₂+D₂,  A₃+B₃+C₃+D₃]
```

2. Scatter step:
Split the reduced vector into four equal slices (one element each here) and give rank r slice r:
| Rank gets | Value         |
| --------- | ------------- |
| 0         | `A₀+B₀+C₀+D₀` |
| 1         | `A₁+B₁+C₁+D₁` |
| 2         | `A₂+B₂+C₂+D₂` |
| 3         | `A₃+B₃+C₃+D₃` |

Now every GPU holds only 1/4 of the reduction, but together those pieces reconstruct the full result.

Typical usage: 
    Sharded optimizers / parameter servers – each GPU keeps the gradient slice for the parameters it owns.
    Pipeline-parallel layers – produce local loss vectors, sum them, and hand each stage just its shard.
    Efficient metrics – compute global sum and immediately discard the parts you don’t need.

## Other core collectives in PyTorch / NCCL ((pronounced “nickel”) = NVIDIA Collective Communications Library)
| Collective | “One-liner description     |
| broadcast  | Rank 0 *sends*, everyone else *receives* a tensor so that all ranks hold the same copy.     |
| scatter    | Rank 0 holds a list/large tensor and **sends slice *r*** to rank *r*.  (Send-only—no reduction.)     |
| gather     | Opposite of scatter: every rank sends a tensor to rank 0, which assembles them into a list/big tensor.  |
| reduce     | Like `all_reduce` but **only the destination rank** (often 0) receives the reduced result; others drop it. |
| **`all_to_all`** | Generalised “shuffle”: every rank sends a distinct slice to every other rank and receives one from each.  Useful for tensor-parallelism where each stage needs different shards of each other’s output. |


## all_to_all():
Assume every GPU starts with four 1-D slices—one it intends for itself plus three it intends for the others:
| Rank  | Local tensor before the call (`xᵣ`) | We’ll treat it as **row-major:** `[send→0, send→1, send→2, send→3]` |
| ----- | ----------------------------------- | ------------------------------------------------------------------- |
| **0** | `[A₀, A₁, A₂, A₃]`                  | GPU 0 plans to send *A₁* to rank 1, *A₂* to rank 2, *A₃* to rank 3. |
| **1** | `[B₀, B₁, B₂, B₃]`                  | …                                                                   |
| **2** | `[C₀, C₁, C₂, C₃]`                  | …                                                                   |
| **3** | `[D₀, D₁, D₂, D₃]`                  | …                                                                   |

We call
```
    # contiguous 2-D tensor: rows = outgoing slices
    x  = torch.stack([row0, row1, row2, row3])      # shape [world_size, slice_len]
    out = torch.empty_like(x)                       # same shape
    dist.all_to_all_single(out, x)                  # low-level API
    # or
    # dist.all_to_all(out_chunks, in_chunks)        # list-of-tensors API
```
What happens under the hood
Partition phase     – the backend interprets row j of rank i as “data destined for rank j”.
Exchange phase      – it ships row j to rank j over NVLink / IB.
Assemble phase      – each rank stitches the received rows together in the order of sender ranks.
| Rank  | Tensor **after** the call (`outᵣ`) | Meaning                                              |
| ----- | ---------------------------------- | ---------------------------------------------------- |
| **0** | `[A₀, B₀, C₀, D₀]`                 | 1st element came from itself, others from ranks 1-3. |
| **1** | `[A₁, B₁, C₁, D₁]`                 | Each element in position *k* came from rank *k*.     |
| **2** | `[A₂, B₂, C₂, D₂]`                 | …                                                    |
| **3** | `[A₃, B₃, C₃, D₃]`                 | …                                                    |
