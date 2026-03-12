# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from attr import dataclass
import pytest
import torch
import torch.nn.functional as F

from tests.pytorch.utils import reset_rng_states
from transformer_engine.pytorch.triton.mhc import (
    mHCElementwiseOp,
    mHCPostResOp,
    mHCPreOp,
    mHCProjectionOp,
    mHCSinkhornOp,
)

seed = 1234
reset_rng_states()

# Enable TF32 for matmul to ensure consistency between the fused and reference implementations
torch.backends.cuda.matmul.allow_tf32 = True

@torch.compile
def mHCElementwiseRef(H, alpha, beta, r, n):
    """
    Reference operator for mHC's pre and post calculations

    :param: Hs: (B, T, 2n + n^2), the unprocessed H matrices
    :param: alpha: (3,), three scalar parameters
    :param: beta: (1, 2n + n^2), bias term
    :param: r: (B, T), the denominator for RMSNorm
    :param: n: int, the width of Hyper-Connection

    :return Hs: (B, T, 2n + n^2), the processed H matrices
    """
    
    B, T, _ = H.shape

    H_pre = H[:, :, :n]  # (B, T, n)
    H_post = H[:, :, n:2*n]  # (B, T, n)
    H_res = H[:, :, 2*n:]  # (B, T, n^2)

    beta_pre = beta[0, :n]
    beta_post = beta[0, n:2*n]
    beta_res = beta[0, 2*n:2*n +  n*n]

    alpha_pre, alpha_post, alpha_res = alpha[0], alpha[1], alpha[2]

    H_pre = H_pre * alpha_pre
    H_post = H_post * alpha_post
    H_res = H_res * alpha_res

    H_pre = H_pre / r[:, :, None]
    H_post = H_post / r[:, :, None]
    H_res = H_res / r[:, :, None]

    H_pre = H_pre + beta_pre
    H_post = H_post + beta_post
    H_res = H_res + beta_res

    H_pre = F.sigmoid(H_pre)
    H_post = 2 * F.sigmoid(H_post)

    out = torch.cat([H_pre, H_post, H_res], dim=-1) # (B, T, 2n + n^2)

    return out

@torch.compile
def mHCPostResRef(f, H_post, x, H_res, n):
    """
    Reference operator for applying mHC's post transformation and residual transformation

    f: (B, T, C)
    H_post: (B, T, n)
    x: (B, T, nC)
    H_res: (B, T, n, n)
    """

    B, T, nC = x.shape
    C = nC // n

    f = f.view(B, T, 1, C)
    H_post = H_post.view(B, T, n, 1)
    x = x.view(B, T, n, C)
    H_res = H_res.view(B, T, n, n)

    out = H_post @ f + H_res @ x # (B, T, n, C)
    out = out.view(B, T, nC)

    return out

@torch.compile
def mHCPreRef(x, H_pre, n):
    """
    Reference operator for applying mHC's pre matrix H to a vector x.

    x: (B, T, nC)
    H_pre: (B, T, n)
    """
    H_pre = H_pre.contiguous()

    B, T, nC = x.shape
    C = nC // n
    x = x.view(B, T, n, C)  # (B, T, n, C)
    H_pre = H_pre.view(B, T, 1, n)  # (B, T, 1, n)

    out = (H_pre @ x).view(B, T, C) # (B, T, C)

    return out

@torch.compile
def mHCProjectionRef(x, phi):
    """
    Reference operator for mHC's projection building operation.

    x: (B, T, nC)
    phi: (nC, 2n + n^2), which consists of the following matrices
        - phi_pre: (nC, n)
        - phi_post: (nC, n)
        - phi_res: (nC, n^2)
    n: number of Hyper Connection streams
    C: hidden dimension per stream
    """
    eps = 1e-8

    B, T, nC = x.shape
    Hs = x @ phi  # (B, T, 2n + n^2)
    norm = torch.sum(x * x, dim=2)
    r = norm / (nC ** 0.5)  # (B, T)

    return Hs, r+eps

@torch.compile
def mHCSinkhornRef(x, n=4, iterations=20):
    """
    Sinkhorn-Knopp algorithm to convert a matrix into a doubly stochastic matrix.
    Calculated in log space for numerical stability.
    
    :param x: a tensor of shape (B, T, n, n)
    """
    B, T, _ = x.shape
    device = x.device

    H_res = x[:, :, 2*n:2*n + n*n].clone().view(B, T, n, n) # (B, T, n*n)
    
    log_mu = torch.zeros(B, T, n, device=device)
    log_nu = torch.zeros(B, T, n, device=device)
    
    f = torch.zeros(B, T, n, device=device)
    g = torch.zeros(B, T, n, device=device)
    
    for _ in range(iterations):
        # Update f: logsumexp over the column dimension (3)
        f = log_mu - torch.logsumexp(H_res + g.unsqueeze(2), dim=3)
        # Update g: logsumexp over the row dimension (2)
        g = log_nu - torch.logsumexp(H_res + f.unsqueeze(3), dim=2)
        
    log_P = f.unsqueeze(3) + H_res + g.unsqueeze(2)
    H_res = torch.exp(log_P)

    out = torch.cat([x[:, :, :2*n], H_res.view(B, T, n*n)], dim=2) # Concatenate the unchanged part with the result

    return out

@dataclass
class MHCConfig:
    B: int = 32 # Batch size
    T: int = 2048 # Sequence length
    C: int = 1024 # Hidden dimension
    n: int = 4 # Number of Hyper Connection streams

    allow_n = [4,]

    def __init__(self, B, T, C, n=4):
        assert n in self.allow_n, f"n must be one of {self.allow_n}"
        self.B = B
        self.T = T
        self.C = C
        self.n = n

    @staticmethod
    def desc(cfg):
        return f"B{cfg.B}_T{cfg.T}_C{cfg.C}_n{cfg.n}"

mhc_configs = [
    MHCConfig(8, 32, 32),
    MHCConfig(8, 128, 16 * 64),
    MHCConfig(4, 128, 16 * 64,),
    MHCConfig(2, 2048, 24 * 128),
    MHCConfig(1, 2048, 24 * 128,),
    MHCConfig(8, 1, 16 * 128,),
    MHCConfig(8, 1, 16 * 256,),
    MHCConfig(8, 1, 16 * 192,),
    MHCConfig(8, 128, 16 * 192,),
    MHCConfig(8, 1, 16 * 512,),
    MHCConfig(8, 128, 16 * 512,),
    MHCConfig(8, 1, 16 * 1024,),
    MHCConfig(8, 128, 16 * 1024,),
]

def get_tols(dtype):
    if dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32:
        return dict(atol=1e-4, rtol=1e-4)
    # Allow higher tolerance for tf32 & bf16 due to their higher numerical error, especially in larger matrix multiplications.
    return dict(atol=1e-1, rtol=1e-1)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_projection(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n
    nC = n * C
    N = 2*n + n*n

    tols = get_tols(dtype)
    # TF32 matmul error grows as sqrt(K) due to rounding in partial products.
    # Two TF32 implementations (triton vs cuBLAS) with different tile sizes diverge
    # by ~sqrt(nC) * eps_tf32. Scale atol accordingly relative to the baseline K=4096.
    if dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
        # TF32 error in a K-dim dot product grows as sqrt(K)*eps_tf32 (random walk over K roundings).
        # Two TF32 implementations with different tile orderings diverge by this amount.
        # eps_tf32 = 2^-10 ≈ 1e-3, safety factor ~2 for tail coverage.
        atol = 2e-3 * (nC ** 0.5)
        tols = dict(atol=atol, rtol=tols['rtol'])

    x = torch.randn(B, T, nC, device='cuda', requires_grad=True, dtype=dtype)
    phi_padded_T = torch.randn(32, nC, dtype=dtype, requires_grad=True, device='cuda')
    phi_padded = phi_padded_T.T # Column-major for the fused op
    phi = phi_padded[:, :N]

    x_ref = x.detach().clone().requires_grad_(True)
    phi_ref = phi.detach().clone().requires_grad_(True)

    ref_out_Hs, ref_out_r = mHCProjectionRef(x_ref, phi_ref)
    fused_out_Hs_padded, fused_out_r = mHCProjectionOp.apply(x, phi_padded)
    fused_out_Hs = fused_out_Hs_padded[:, :, :N]

    torch.testing.assert_close(fused_out_Hs, ref_out_Hs, **tols)
    torch.testing.assert_close(fused_out_r, ref_out_r, **tols)

    (ref_out_Hs.sum() + ref_out_r.sum()).backward()
    (fused_out_Hs.sum() + fused_out_r.sum()).backward()

    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    torch.testing.assert_close(phi_padded_T.grad.T[:, :N], phi_ref.grad, **tols)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["fp32"])
def test_mhc_elementwise(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n
    N = 2*n + n*n

    tols = get_tols(dtype)

    H_padded = torch.randn(B, T, 32, device='cuda', requires_grad=True, dtype=dtype)
    H = H_padded[:, :, :N]
    alpha = torch.randn(3, device='cuda', requires_grad=True, dtype=dtype)
    beta = torch.randn(1, 2*n + n*n, device='cuda', requires_grad=True, dtype=dtype)
    r_raw = torch.randn(B, T, device='cuda', dtype=dtype) + 1.0
    r = r_raw.detach().clone().requires_grad_(True)

    H_ref = H.detach().clone().requires_grad_(True)
    alpha_ref = alpha.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    r_ref = r.detach().clone().requires_grad_(True)

    ref_out = mHCElementwiseRef(H_ref[:, :, :N], alpha_ref, beta_ref, r_ref, n)
    fused_out_padded = mHCElementwiseOp.apply(H_padded, alpha, beta, r, n)
    fused_out = fused_out_padded[:, :, :N]

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(H_padded.grad[:, :, :N], H_ref.grad, **tols)
    torch.testing.assert_close(alpha.grad, alpha_ref.grad, **tols)
    torch.testing.assert_close(beta.grad, beta_ref.grad, **tols)
    torch.testing.assert_close(r.grad, r_ref.grad, **tols)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["fp32"])
def test_mhc_sinkhorn_knopp(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n

    tols = get_tols(dtype)

    x = torch.randn(B, T, 2*n + n*n, device='cuda', requires_grad=True, dtype=dtype)

    x_ref = x.detach().clone().requires_grad_(True)

    ref_out = mHCSinkhornRef(x_ref, n)
    fused_out = mHCSinkhornOp.apply(x, n)

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad, **tols)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_pre(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n
    nC = n * C

    tols = get_tols(dtype)

    x = torch.randn(B, T, nC, device='cuda', requires_grad=True, dtype=dtype)
    H_pre = torch.randn(B, T, n, device='cuda', requires_grad=True, dtype=dtype)

    x_ref = x.detach().clone().requires_grad_(True)
    H_pre_ref = H_pre.detach().clone().requires_grad_(True)

    ref_out = mHCPreRef(x_ref, H_pre_ref, n)
    fused_out = mHCPreOp.apply(x, H_pre, n)

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    torch.testing.assert_close(H_pre.grad, H_pre_ref.grad, **tols)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_post_res(cfg: MHCConfig, dtype):
    B, T, C, n = cfg.B, cfg.T, cfg.C, cfg.n
    nC = n * C

    tols = get_tols(dtype)

    f = torch.randn(B, T, C, device='cuda', requires_grad=True, dtype=dtype)
    H_post = torch.randn(B, T, n, device='cuda', requires_grad=True, dtype=dtype)
    x = torch.randn(B, T, nC, device='cuda', requires_grad=True, dtype=dtype)
    H_res = torch.randn(B, T, n*n, device='cuda', requires_grad=True, dtype=dtype)

    f_ref = f.detach().clone().requires_grad_(True)
    H_post_ref = H_post.detach().clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    H_res_ref = H_res.detach().clone().requires_grad_(True)

    ref_out = mHCPostResRef(f_ref, H_post_ref, x_ref, H_res_ref, n)
    fused_out = mHCPostResOp.apply(f, H_post, x, H_res, n)

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(f.grad, f_ref.grad, **tols)
    torch.testing.assert_close(H_post.grad, H_post_ref.grad, **tols)
    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    torch.testing.assert_close(H_res.grad, H_res_ref.grad, **tols)
