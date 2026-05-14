import torch, time
from flash_attn import flash_attn_func

torch.manual_seed(0)


def ref_attn(q, k, v, causal=False, softmax_scale=None):
    q2 = q.transpose(1, 2).float()
    k2 = k.transpose(1, 2).float()
    v2 = v.transpose(1, 2).float()
    scale = softmax_scale if softmax_scale is not None else 1.0 / (q.shape[-1] ** 0.5)
    s = (q2 @ k2.transpose(-2, -1)) * scale
    if causal:
        S = q.shape[1]
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = (p @ v2).transpose(1, 2)
    return o.to(q.dtype)


print(f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

# accuracy
for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    for causal in [False, True]:
        for D in [64, 128]:
            B, S, H = 2, 1024, 8
            q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
            k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
            v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
            out = flash_attn_func(q, k, v, causal=causal)
            ref = ref_attn(q, k, v, causal=causal)
            diff = (out.float() - ref.float()).abs()
            rel = diff.max().item() / max(ref.float().abs().max().item(), 1e-6)
            print(
                f"dtype={dtype_name} causal={causal} D={D} shape={tuple(out.shape)} "
                f"max_abs_diff={diff.max().item():.4e} max_rel={rel:.4e}"
            )

# backward
q = torch.randn(2, 1024, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 1024, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(2, 1024, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
out = flash_attn_func(q, k, v, causal=False)
g = torch.randn_like(out)
out.backward(g)
print(
    "backward finite dq/dk/dv =",
    torch.isfinite(q.grad).all().item(),
    torch.isfinite(k.grad).all().item(),
    torch.isfinite(v.grad).all().item(),
)

# latency
B, S, H, D = 2, 4096, 8, 128
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
for _ in range(10):
    flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
N = 50
t0 = time.perf_counter()
for _ in range(N):
    flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"fwd causal hdim=128 seqlen=4096 fp16 latency_ms_per_call={(t1 - t0) / N * 1000:.3f}")
