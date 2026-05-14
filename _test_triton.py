import torch
from flash_attn.flash_attn_triton import flash_attn_func

torch.manual_seed(0)
print(f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")


def ref_attn(q, k, v, causal=False, softmax_scale=None):
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float()
    vf = v.transpose(1, 2).float()
    s = (qf @ kf.transpose(-1, -2)) * (softmax_scale if softmax_scale is not None else (1.0 / (q.shape[-1] ** 0.5)))
    if causal:
        sq, sk = qf.shape[-2], kf.shape[-2]
        mask = torch.ones(sq, sk, device=q.device, dtype=torch.bool).tril(sk - sq)
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = (p @ vf).transpose(1, 2).contiguous()
    return o.to(q.dtype)


# accuracy
for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    for causal in (False, True):
        for D in (64, 128):
            B, H, S = 2, 4, 256
            q = torch.randn(B, S, H, D, device="cuda", dtype=dtype) * 0.5
            k = torch.randn(B, S, H, D, device="cuda", dtype=dtype) * 0.5
            v = torch.randn(B, S, H, D, device="cuda", dtype=dtype) * 0.5
            o_ref = ref_attn(q, k, v, causal=causal)
            o = flash_attn_func(q, k, v, None, causal)
            diff = (o.float() - o_ref.float()).abs()
            rel = (diff / (o_ref.float().abs() + 1e-6)).max().item()
            print(f"triton dtype={dtype_name} causal={causal} D={D} max_abs={diff.max().item():.4e} max_rel={rel:.4e}")

# backward (smaller shape to fit RTX 5060 Ti shared memory budget for triton autotune)
B, H, S, D = 1, 2, 64, 32
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
o = flash_attn_func(q, k, v, None, False)
o.sum().backward()
print(
    "triton backward finite dq/dk/dv =",
    torch.isfinite(q.grad).all().item(),
    torch.isfinite(k.grad).all().item(),
    torch.isfinite(v.grad).all().item(),
)

# backward vs reference
q2 = q.detach().clone().requires_grad_(True)
k2 = k.detach().clone().requires_grad_(True)
v2 = v.detach().clone().requires_grad_(True)
o_ref = ref_attn(q2, k2, v2, causal=False)
o_ref.sum().backward()
for name, a, b in (("dq", q.grad, q2.grad), ("dk", k.grad, k2.grad), ("dv", v.grad, v2.grad)):
    diff = (a.float() - b.float()).abs()
    rel = (diff / (b.float().abs() + 1e-6)).max().item()
    print(f"triton grad {name} max_abs={diff.max().item():.4e} max_rel={rel:.4e}")
