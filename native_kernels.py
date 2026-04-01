import torch
import torch.nn.functional as F

@torch.compile
def mHCProjectionRef(x, phi):
    """
    Reference operator for mHC's projection building operation.

    x: (M, nC) where M = B * T
    phi: (2n + n^2, nC), which consists of the following matrices
        - phi_pre: (n, nC)
        - phi_post: (n, nC)
        - phi_res: (n^2, nC)
    n: number of Hyper Connection streams
    C: hidden dimension per stream
    """
    eps = torch.finfo(torch.float32).eps
    x_dtype = x.dtype
    x = x.to(torch.float32)
    phi = phi.to(torch.float32)

    Hs = x @ phi.T  # (M, 2n + n^2)

    x_fp32 = x.to(torch.float32)  # Use fp32 for better numerical stability in variance calculation
    var = (x_fp32 * x_fp32).mean(dim=1)
    r = torch.sqrt(var + eps)  # (M,)

    return Hs.to(x_dtype), r.to(x_dtype)


@torch.compile
def mHCElementwiseRef(H, alpha, beta, r, n):
    """
    Reference operator for mHC's pre and post calculations

    :param: H: (M, 2n + n^2), the unprocessed H matrices where M = B * T
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
    r = r.to(torch.float32)

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

    H_pre = H_pre / r[:, None]
    H_post = H_post / r[:, None]
    H_res = H_res / r[:, None]

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

    :param H_res: a tensor of shape (B, T, n, n)
    :return: a tensor of shape (B, T, n, n)
    """
    B, T = H_res.shape[:2]
    device = H_res.device
    dtype = H_res.dtype

    H_res_f = H_res.to(
        torch.float32
    ).clone()  # Use float32 for better numerical stability during Sinkhorn iterations

    log_mu = torch.zeros(B, T, n, device=device, dtype=torch.float32)
    log_nu = torch.zeros(B, T, n, device=device, dtype=torch.float32)

    f = torch.zeros(B, T, n, device=device, dtype=torch.float32)
    g = torch.zeros(B, T, n, device=device, dtype=torch.float32)

    for _ in range(iterations):
        # Update f: logsumexp over the column dimension (3)
        f = log_mu - torch.logsumexp(H_res_f + g.unsqueeze(2), dim=3)
        # Update g: logsumexp over the row dimension (2)
        g = log_nu - torch.logsumexp(H_res_f + f.unsqueeze(3), dim=2)

    log_P = f.unsqueeze(3) + H_res_f + g.unsqueeze(2)
    H_res_out = torch.exp(log_P).to(dtype)  # Convert back to original dtype

    return H_res_out


@torch.compile
def mHCPreRef(x, H_pre, n):
    """
    Reference operator for applying mHC's pre matrix H to a vector x.

    x: (B, T, n, C)
    H_pre: (B, T, n)
    """
    H_pre = H_pre.contiguous()

    B, T, n, C = x.shape
    H_pre = H_pre.view(B, T, 1, n)  # (B, T, 1, n)

    out = (H_pre @ x).view(B, T, C)  # (B, T, C)

    return out


@torch.compile
def mHCPostResRef(f, H_post, x, H_res, n):
    """
    Reference operator for applying mHC's post transformation and residual transformation

    f: (B, T, C)
    H_post: (B, T, n)
    x: (B, T, n, C)
    H_res: (B, T, n, n)
    """

    B, T, n, C = x.shape

    f = f.view(B, T, 1, C)
    H_post = H_post.view(B, T, n, 1)

    out = H_post @ f + H_res @ x  # (B, T, n, C)

    return out

