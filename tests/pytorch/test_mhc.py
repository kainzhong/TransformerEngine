# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from attr import dataclass
import pytest
import torch
import torch.nn.functional as F

from utils import reset_rng_states
from transformer_engine.pytorch.triton.mhc import (
    mHCScaleFusedOp,
    mHCExpandCombineOp,
    mHCAggregateOp,
    mHCProjectionOp,
    mHCSinkhornOp,
)

seed = 1234
reset_rng_states()

# Enable TF32 for matmul to ensure consistency between the fused and reference implementations
torch.backends.cuda.matmul.allow_tf32 = False


@torch.compile
def mHCProjectionRef(x, phi):
    """
    Reference operator for mHC's projection building operation.

    x: (M, nC) where M = s * b
    phi: (2n + n^2, nC), which consists of the following matrices
        - phi_pre: (n, nC)
        - phi_post: (n, nC)
        - phi_res: (n^2, nC)
    n: number of Hyper Connection streams
    C: hidden dimension per stream
    """
    x_dtype = x.dtype
    x = x.to(torch.float32)
    phi = phi.to(torch.float32)

    Hs = x @ phi.T  # (M, 2n + n^2)

    x_fp32 = x.to(torch.float32)  # Use fp32 for better numerical stability in variance calculation
    ms = (x_fp32 * x_fp32).mean(dim=1)

    return Hs.to(x_dtype), ms


@torch.compile
def mHCScaleRef(H, alpha, beta, ms, n):
    """
    Reference operator for mHC's pre and post calculations

    :param: H: (M, 2n + n^2), the unprocessed H matrices where M = s * b
    :param: alpha: (3,), three scalar parameters
    :param: beta: (1, 2n + n^2), bias term
    :param: r: (M,), the denominator for RMSNorm
    :param: n: int, the width of Hyper-Connection

    :return Hs: (M, 2n + n^2), the processed H matrices
    """

    M, _ = H.shape
    H_dtype = H.dtype
    H = H.to(torch.float32)
    alpha = alpha.to(torch.float32)
    beta = beta.to(torch.float32)
    eps = torch.finfo(torch.float32).eps
    rms = torch.sqrt(ms + eps)  # (M,)
    rms = rms.to(torch.float32)

    H_pre = H[:, :n]  # (M, n)
    H_post = H[:, n : 2 * n]  # (M, n)
    H_res = H[:, 2 * n :]  # (M, n^2)

    beta_pre = beta[0, :n]
    beta_post = beta[0, n : 2 * n]
    beta_res = beta[0, 2 * n : 2 * n + n * n]

    alpha_pre, alpha_post, alpha_res = alpha[0], alpha[1], alpha[2]

    H_pre = H_pre * alpha_pre
    H_post = H_post * alpha_post
    H_res = H_res * alpha_res

    H_pre = H_pre / rms[:, None]
    H_post = H_post / rms[:, None]
    H_res = H_res / rms[:, None]

    H_pre = H_pre + beta_pre
    H_post = H_post + beta_post
    H_res = H_res + beta_res

    H_pre = F.sigmoid(H_pre)
    H_post = 2 * F.sigmoid(H_post)

    out = torch.cat([H_pre, H_post, H_res], dim=-1)  # (M, 2n + n^2)

    return out.to(H_dtype)


@torch.compile
def mHCSinkhornRef(H_res, n=4, iterations=20):
    """
    Sinkhorn-Knopp algorithm to convert a matrix into a doubly stochastic matrix.
    Calculated in log space for numerical stability.

    :param H_res: a tensor of shape (s, b, n, n)
    :return: a tensor of shape (s, b, n, n)
    """
    s, b = H_res.shape[:2]
    device = H_res.device
    dtype = H_res.dtype

    H_res_f = H_res.to(
        torch.float32
    ).clone()  # Use float32 for better numerical stability during Sinkhorn iterations

    log_mu = torch.zeros(s, b, n, device=device, dtype=torch.float32)
    log_nu = torch.zeros(s, b, n, device=device, dtype=torch.float32)

    f = torch.zeros(s, b, n, device=device, dtype=torch.float32)
    g = torch.zeros(s, b, n, device=device, dtype=torch.float32)

    for _ in range(iterations):
        # Update f: logsumexp over the column dimension (3)
        f = log_mu - torch.logsumexp(H_res_f + g.unsqueeze(2), dim=3)
        # Update g: logsumexp over the row dimension (2)
        g = log_nu - torch.logsumexp(H_res_f + f.unsqueeze(3), dim=2)

    log_P = f.unsqueeze(3) + H_res_f + g.unsqueeze(2)
    H_res_out = torch.exp(log_P).to(dtype)  # Convert back to original dtype

    return H_res_out


@torch.compile
def mHCAggregateRef(x, H_pre, n):
    """
    Reference operator for applying mHC's pre matrix H to a vector x.

    x: (s, b, C, n)
    H_pre: (s, b, n)
    """
    H_pre = H_pre.contiguous()

    s, b, C, n = x.shape
    H_pre = H_pre.view(s, b, n, 1)

    out = (x @ H_pre).view(s, b, C)

    return out

@torch.compile
def mHCExpandCombineRef(f, bias, H_post, x, H_res, n):
    """
    Reference operator for applying mHC's post transformation and residual transformation

    f: (s, b, C)
    bias: (C,) or None
    H_post: (s, b, n)
    x: (s, b, C, n)
    H_res: (s, b, n, n)
    """

    s, b, C, n = x.shape

    if bias is not None:
        f = f + bias

    f = f.view(s, b, C, 1)
    H_post = H_post.view(s, b, 1, n)

    out = f @ H_post + x @ H_res  # (s, b, C, n)

    return out

@dataclass
class MHCConfig:
    s: int = 2048  # Sequence length
    b: int = 32  # Batch size
    C: int = 1024  # Hidden dimension
    n: int = 4  # Number of Hyper Connection streams

    allow_n = [
        4,
    ]

    def __init__(self, b, s, C, n=4):
        assert n in self.allow_n, f"n must be one of {self.allow_n}"
        self.b = b
        self.s = s
        self.C = C
        self.n = n

    @staticmethod
    def desc(cfg):
        return f"b{cfg.b}_s{cfg.s}_C{cfg.C}_n{cfg.n}"


mhc_configs = [
    MHCConfig(8, 32, 32),
    MHCConfig(8, 128, 16 * 64),
    MHCConfig(
        4,
        128,
        16 * 64,
    ),
    MHCConfig(2, 2048, 24 * 128),
    MHCConfig(
        1,
        2048,
        24 * 128,
    ),
    MHCConfig(
        13,
        1,
        16 * 128,
    ),
    MHCConfig(
        7,
        1,
        16 * 256,
    ),
    MHCConfig(
        8,
        1,
        16 * 192,
    ),
    MHCConfig(
        8,
        128,
        16 * 192,
    ),
    MHCConfig(
        8,
        1,
        16 * 500,
    ),
    MHCConfig(
        8,
        128,
        16 * 512,
    ),
    MHCConfig(
        8,
        1,
        16 * 376,
    ),
    MHCConfig(
        8,
        128,
        16 * 1024,
    ),
]


def get_tols(dtype):
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)
    else:
        tols = dict(atol=5e-3, rtol=5e-3)
    return tols


@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_projection(cfg: MHCConfig, dtype):
    s, b, C, n = cfg.s, cfg.b, cfg.C, cfg.n
    nC = n * C
    N = 2 * n + n * n

    tols = get_tols(dtype)
    use_tf32 = False

    x = torch.randn(s * b, nC, device="cuda", requires_grad=True, dtype=dtype)
    phi = torch.randn(N, nC, dtype=dtype, requires_grad=True, device="cuda")

    x_ref = x.detach().clone().requires_grad_(True)
    phi_ref = phi.detach().clone().requires_grad_(True)

    ref_out_Hs, ref_out_ms = mHCProjectionRef(x_ref, phi_ref)
    fused_out_Hs_padded, fused_out_ms = mHCProjectionOp.apply(x, phi, use_tf32)
    fused_out_Hs = fused_out_Hs_padded[:, :N]

    torch.testing.assert_close(fused_out_Hs, ref_out_Hs, **tols)
    torch.testing.assert_close(fused_out_ms, ref_out_ms, **tols)
    (ref_out_Hs.sum() + ref_out_ms.sum()).backward()
    (fused_out_Hs.sum() + fused_out_ms.sum()).backward()

    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    torch.testing.assert_close(phi.grad, phi_ref.grad, **tols)


@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["fp32"])
def test_mhc_elementwise(cfg: MHCConfig, dtype):
    s, b, C, n = cfg.s, cfg.b, cfg.C, cfg.n
    N = 2 * n + n * n

    tols = get_tols(dtype)

    H_padded = torch.randn(s * b, 32, device="cuda", requires_grad=True, dtype=dtype)
    H = H_padded[:, :N]
    alpha = torch.randn(3, device="cuda", requires_grad=True, dtype=dtype)
    beta = torch.randn(1, 2 * n + n * n, device="cuda", requires_grad=True, dtype=dtype)
    ms_raw = torch.randn(s * b, device="cuda", dtype=dtype).abs() + 1.0
    ms = ms_raw.detach().clone().requires_grad_(True)

    H_ref = H.detach().clone().requires_grad_(True)
    alpha_ref = alpha.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    ms_ref = ms.detach().clone().requires_grad_(True)

    ref_out = mHCScaleRef(H_ref[:, :N], alpha_ref, beta_ref, ms_ref, n)
    fused_out_padded = mHCScaleFusedOp.apply(H_padded, alpha, beta, ms, n)
    fused_out = fused_out_padded[:, :N]

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(H_padded.grad[:, :N], H_ref.grad, **tols)
    torch.testing.assert_close(alpha.grad, alpha_ref.grad, **tols)
    torch.testing.assert_close(beta.grad, beta_ref.grad, **tols)
    torch.testing.assert_close(ms.grad, ms_ref.grad, **tols)


@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_combined(cfg: MHCConfig, dtype):
    s, b, C, n = cfg.s, cfg.b, cfg.C, cfg.n
    N = 2 * n + n * n
    nC = n * C

    tols = get_tols(dtype)

    tols = get_tols(dtype)
    use_tf32 = False

    x = torch.randn(s * b, nC, device="cuda", requires_grad=True, dtype=dtype)
    phi = torch.randn(N, nC, dtype=dtype, requires_grad=True, device="cuda")

    alpha = torch.randn(3, device="cuda", requires_grad=True, dtype=dtype)
    beta = torch.randn(1, 2 * n + n * n, device="cuda", requires_grad=True, dtype=dtype)

    x_ref = x.detach().clone().requires_grad_(True)
    phi_ref = phi.detach().clone().requires_grad_(True)

    alpha_ref = alpha.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)

    ref_out_H, ref_out_r = mHCProjectionRef(x_ref, phi_ref)
    fused_out_H_padded, fused_out_r = mHCProjectionOp.apply(x, phi, use_tf32)

    ref_out = mHCScaleRef(ref_out_H[:, :N], alpha_ref, beta_ref, ref_out_r, n)
    fused_out_padded = mHCScaleFusedOp.apply(fused_out_H_padded, alpha, beta, fused_out_r, n)
    fused_out = fused_out_padded[:, :N]

    def mhc_combined(x_ref, phi_ref, alpha_ref, beta_ref):
        dtype = x_ref.dtype
        x_ref = x_ref.to(torch.float32)
        phi_ref = phi_ref.to(torch.float32)
        alpha_ref = alpha_ref.to(torch.float32)
        beta_ref = beta_ref.to(torch.float32)

        x_rmsnorm = F.rms_norm(x_ref, normalized_shape=(nC,))
        H = x_rmsnorm @ phi_ref.T
        H_pre = H[:, :n]
        H_post = H[:, n : 2 * n]
        H_res = H[:, 2 * n :]

        out_pre = H_pre * alpha_ref[0] + beta_ref[:, :n]
        out_post = H_post * alpha_ref[1] + beta_ref[:, n : 2 * n]
        out_res = H_res * alpha_ref[2] + beta_ref[:, 2 * n :]

        out_pre = out_pre.sigmoid()
        out_post = 2 * out_post.sigmoid()
        out_res = out_res

        return out_pre.to(dtype), out_post.to(dtype), out_res.to(dtype)

    H_pre_combined, H_post_combined, _ = mhc_combined(x_ref, phi_ref, alpha_ref, beta_ref)

    torch.testing.assert_close(H_pre_combined, ref_out[:, :n], **tols)
    torch.testing.assert_close(H_post_combined, ref_out[:, n : 2 * n], **tols)

    torch.testing.assert_close(H_pre_combined, fused_out[:, :n], **tols)
    torch.testing.assert_close(H_post_combined, fused_out[:, n : 2 * n], **tols)


@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "recompute"])
def test_mhc_sinkhorn_knopp(cfg: MHCConfig, dtype, recompute):
    s, b, C, n = cfg.s, cfg.b, cfg.C, cfg.n

    tols = get_tols(dtype)

    x = torch.randn(s, b, n, n, device="cuda", requires_grad=True, dtype=dtype)
    x_ref = x.detach().clone().requires_grad_(True)

    ref_out = mHCSinkhornRef(x_ref, n)
    fused_out = mHCSinkhornOp.apply(x, n, recompute)

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad, **tols)

@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_mhc_aggregate(cfg: MHCConfig, dtype):
    s, b, C, n = cfg.s, cfg.b, cfg.C, cfg.n

    tols = get_tols(dtype)

    x = torch.randn(s, b, C, n, device="cuda", requires_grad=True, dtype=dtype)
    H_pre = torch.randn(s, b, n, device="cuda", requires_grad=True, dtype=dtype)

    x_ref = x.detach().clone().requires_grad_(True)
    H_pre_ref = H_pre.detach().clone().requires_grad_(True)

    ref_out = mHCAggregateRef(x_ref, H_pre_ref, n)
    fused_out = mHCAggregateOp.apply(x, H_pre, n, False)

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    torch.testing.assert_close(H_pre.grad, H_pre_ref.grad, **tols)


@pytest.mark.parametrize("cfg", mhc_configs, ids=MHCConfig.desc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("with_bias", [True, False], ids=["with_bias", "no_bias"])
def test_mhc_expand_combine(cfg: MHCConfig, dtype, with_bias):
    s, b, C, n = cfg.s, cfg.b, cfg.C, cfg.n

    tols = get_tols(dtype)

    f = torch.randn(s, b, C, device="cuda", requires_grad=True, dtype=dtype)
    bias = None
    if with_bias:
        bias_raw = torch.randn(C, device="cuda", requires_grad=True, dtype=dtype) * 0.1
        bias = bias_raw.detach().clone().requires_grad_(True)
    H_post = torch.randn(s, b, n, device="cuda", requires_grad=True, dtype=dtype)
    x = torch.randn(s, b, C, n, device="cuda", requires_grad=True, dtype=dtype)
    H_res = torch.randn(s, b, n, n, device="cuda", requires_grad=True, dtype=dtype)

    f_ref = f.detach().clone().requires_grad_(True)
    bias_ref = None  if bias is None else bias.detach().clone().requires_grad_(True)
    H_post_ref = H_post.detach().clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    H_res_ref = H_res.detach().clone().requires_grad_(True)

    ref_out = mHCExpandCombineRef(f_ref, bias_ref, H_post_ref, x_ref, H_res_ref, n)
    fused_out = mHCExpandCombineOp.apply(f, bias, H_post, x, H_res, n, False)

    torch.testing.assert_close(fused_out, ref_out, **tols)

    ref_out.sum().backward()
    fused_out.sum().backward()

    torch.testing.assert_close(f.grad, f_ref.grad, **tols)
    torch.testing.assert_close(H_post.grad, H_post_ref.grad, **tols)
    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    torch.testing.assert_close(H_res.grad, H_res_ref.grad, **tols)
