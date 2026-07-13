import torch
from flash_attn import flash_attn_func

try:
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    # Forward pass test
    q = torch.randn(2, 128, 8, 128, dtype=torch.float16, device='cuda', requires_grad=True)
    k = torch.randn(2, 128, 8, 128, dtype=torch.float16, device='cuda', requires_grad=True)
    v = torch.randn(2, 128, 8, 128, dtype=torch.float16, device='cuda', requires_grad=True)
    
    print('Testing FA2 Forward pass...')
    out = flash_attn_func(q, k, v, causal=True)
    print(f'Forward pass success! Output shape: {out.shape}')
    
    # Backward pass test
    print('Testing FA2 Backward pass...')
    loss = out.sum()
    loss.backward()
    print(f'Backward pass success! q.grad shape: {q.grad.shape}')
    
    print('\nAll FA2 functionality tests passed successfully!')
except Exception as e:
    print(f'\nError occurred during testing: {e}')
