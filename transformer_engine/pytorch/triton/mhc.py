# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from flask import g
import torch
import triton
import triton.language as tl

from transformer_engine.common.triton.mhc import (
    _mhc_scale_fwd_fused,
    _mhc_scale_bwd_fused,
    _mhc_expand_combine_with_bias_fwd,
    _mhc_expand_combine_with_bias_bwd,
    _mhc_expand_combine_fwd,
    _mhc_expand_combine_bwd,
    _mhc_aggregate_fwd,
    _mhc_aggregate_bwd,
    _mhc_projection_fwd_fused,
    _mhc_projection_bwd_fused,
    _mhc_sinkhorn_fwd_fused,
    _mhc_sinkhorn_fwd_fused_recompute,
    _mhc_sinkhorn_bwd_fused,
    _mhc_sinkhorn_bwd_fused_recompute,
)


class mHCProjectionOp(torch.autograd.Function):
    """
    Fused projection operation to compute H matrices and mean square for RMSNorm (see eq. 14-15, seciton 4.3.1 of the DeepSeek mHC paper)
    :param x: input tensor of shape (M, K), where M=s*b is the batch size and K=nC is the hidden dimension after expansion.
    :param phi: projection matrix of shape (N, K), where N=n+n+n*n

    H = x @ phi^T: (M, K) @ (K, N) -> (M, N), which is padded to (M, 32) for better memory access pattern in the next kernels.
    ms = mean(x^2, dim=-1): (M,)

    :return: H of shape (M, 32), where only the first N elements in the last dimension are valid
    :return: ms of shape (M,), which is the mean square used for RMSNorm in the next kernel

    Note: the current implementation only supports n=4
    """

    @staticmethod
    def forward(ctx, x, phi, use_tf32=True):
        x = x.contiguous()

        ctx.use_tf32 = use_tf32
        ctx.dtype = x.dtype

        M, K = x.shape
        device = x.device

        N = phi.shape[0]
        assert N == 24, "Currently only n=4 is supported, which means phi should have 24 rows"

        # Pad H to (s, b, 32) for better memory access pattern in the kernel, but only the first N elements in the last dimension are valid
        H = torch.zeros((M, 32), device=device, dtype=torch.float32)
        ms = torch.zeros((M), device=device, dtype=torch.float32) # Mean square for s, used to compute RMSNorm in the next kernel

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(K, META["BLOCK_SIZE_K"]))

        if use_tf32:
            _mhc_projection_fwd_fused[grid](
                x_ptr=x,  # (M, K)
                phi_ptr=phi,  # (N, K)
                h_ptr=H,  # (M, 32)
                ms_ptr=ms,  # (M,)
                M=M,
                N=N,
                K=K,
                stride_xm=K,
                stride_xk=1,
                stride_phin=K,
                stride_phik=1,
                stride_hm=32,
                stride_hn=1,
                stride_ms=1,
                BLOCK_SIZE_N=32,
                precision="tf32",
            )
        else:
            _mhc_projection_fwd_fused[grid](
                x_ptr=x,  # (M, K)
                phi_ptr=phi,  # (N, K)
                h_ptr=H,  # (M, 32)
                ms_ptr=ms,  # (M,)
                M=M,
                N=N,
                K=K,
                stride_xm=K,
                stride_xk=1,
                stride_phin=K,
                stride_phik=1,
                stride_hm=32,
                stride_hn=1,
                stride_ms=1,
                BLOCK_SIZE_N=32,
                precision="ieee",
            )

        ctx.save_for_backward(x, phi, ms)
        ctx.phi_dtype = phi.dtype

        return H.to(ctx.dtype), ms # Keep ms in fp32

    @staticmethod
    def backward(ctx, grad_H, grad_ms):
        x, phi, ms = ctx.saved_tensors
        M, K = x.shape
        device = x.device

        N = phi.shape[0]

        grad_H = grad_H.contiguous().view(M, -1)
        grad_ms = grad_ms.contiguous().view(M,)
        ms = ms.contiguous().view(M,)

        grad_x = torch.zeros((M, K), device=device, dtype=x.dtype)
        grad_phi = (grad_H.T @ x)[:N, :].to(
            ctx.phi_dtype
        )  # (2n + n^2, M) @ (M, nC) = (2n + n^2, nC), note that the last dimension of grad_H is already padded to 32

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(K, META["BLOCK_SIZE_K"]),
        )

        if ctx.use_tf32:
            _mhc_projection_bwd_fused[grid](
                x_ptr=x,
                grad_x_ptr=grad_x,  # (M, K)
                phi_ptr=phi,  # (N, K)
                grad_h_ptr=grad_H,  # (M, 32)
                grad_ms_ptr=grad_ms,  # (M,)
                M=M,
                N=N,
                K=K,
                stride_xm=K,
                stride_xk=1,
                stride_grad_xm=K,
                stride_grad_xk=1,
                stride_phin=K,
                stride_phik=1,
                stride_grad_phin=K,
                stride_grad_phik=1,
                stride_grad_hm=32,
                stride_grad_hn=1,
                stride_grad_ms=1,
                BLOCK_SIZE_N=32,
                precision="tf32",
            )
        else:
            _mhc_projection_bwd_fused[grid](
                x_ptr=x,
                grad_x_ptr=grad_x,  # (M, K)
                phi_ptr=phi,  # (N, K)
                grad_h_ptr=grad_H,  # (M, 32)
                grad_ms_ptr=grad_ms,  # (M,),
                M=M,
                N=N,
                K=K,
                stride_xm=K,
                stride_xk=1,
                stride_grad_xm=K,
                stride_grad_xk=1,
                stride_phin=K,
                stride_phik=1,
                stride_grad_phin=K,
                stride_grad_phik=1,
                stride_grad_hm=32,
                stride_grad_hn=1,
                stride_grad_ms=1,
                BLOCK_SIZE_N=32,
                precision="ieee",
            )

        return grad_x.to(ctx.dtype), grad_phi.to(ctx.dtype), None


class mHCScaleFusedOp(torch.autograd.Function):
    """
    Fused scale operation to compute the scaled H matrices (see eq. 16-18, section 4.3.1 of the DeepSeek mHC paper)
    :param H: input H matrix of shape (M, 32), where M=s*b, and only the first N elements in the last dimension are valid
    :param alpha: scaling factor for H, of shape (3,), where
        alpha[0] is applied to H[:, 0:n] for H_pre
        alpha[1] is applied to H[:, n:2n] for H_post
        alpha[2] is applied to H[:, 2n:2n+n*n] for H_res
    :param beta: bias term for H, of shape (2*n+n*n,), where
        beta[0:n] is applied to H[:, 0:n] for H_pre
        beta[n:2n] is applied to H[:, n:2n] for H_post
        beta[2n:2n+n*n] is applied to H[:, 2n:2n+n*n] for H_res
    :param ms: mean square for each row of H from the projection kernel, of shape (M,), used for RMSNorm scaling
    :param n: number of hyper connections, where only n=4 is supported in the current implementation

    H_pre = H[:, 0:n] * alpha[0] / sqrt(ms) + beta[0:n]
    H_post = H[:, n:2n] * alpha[1] / sqrt(ms) + beta[n:2n]
    H_res = H[:, 2n:2n+n*n] * alpha[2] / sqrt(ms) + beta[2n:2n+n*n]

    H_pre = sigmoid(H_pre)
    H_post = 2*sigmoid(H_post)

    :return: out of shape (M, 32), where only the first N elements in the last dimension are valid
    """

    @staticmethod
    def forward(ctx, H, alpha, beta, ms, n):
        assert n == 4, "Only n=4 is supported in this implementation"

        ctx.dtype = H.dtype
        H = H.to(torch.float32)
        alpha = alpha.to(torch.float32)
        beta = beta.to(torch.float32)
        ms = ms.to(torch.float32)

        M, _ = H.shape

        H = H.contiguous()
        beta = beta.contiguous()
        ms = ms.contiguous()

        out = torch.empty(
            (M, 32), device=H.device, dtype=H.dtype
        )  # Pad the output to 32 in the last dimension

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        _mhc_scale_fwd_fused[grid](
            h_ptr=H,  # (M, N), which is padded to (M, 32)
            b_ptr=beta,  # (N,)
            a_ptr=alpha,  # (N,)
            ms_ptr=ms,  # (M,)
            out_ptr=out,  # (M, N), which is padded to (M, 32)
            M=M,
            n=n,
            stride_hm=32,
            stride_hn=1,
            stride_a=1,
            stride_b=1,
            stride_ms=1,
            stride_out_m=32,
            stride_out_n=1,  # strides for out, which is padded to 32 in the last dimension
            BLOCK_SIZE_N=32,
            eps=torch.finfo(ms.dtype).eps,
        )

        ctx.save_for_backward(H, alpha, ms, out)
        ctx.n = n

        return out.to(ctx.dtype)  # Cast back to the original dtype of H

    @staticmethod
    def backward(ctx, grad_out):
        H, alpha, ms, out = ctx.saved_tensors
        n = ctx.n

        grad_out = grad_out.contiguous()
        grad_out = grad_out.to(torch.float32)

        M, _ = grad_out.shape
        N = 2 * n + n * n

        grad_h = torch.zeros(
            (M, 32), device=grad_out.device, dtype=grad_out.dtype
        )  # Pad the grad_h to 32 in the last dimension
        grad_alpha = torch.zeros((3,), device=grad_out.device, dtype=grad_out.dtype)
        grad_beta_padded = torch.zeros((1, 32), device=grad_out.device, dtype=grad_out.dtype)
        grad_beta = grad_beta_padded[
            :, :N
        ]  # Use only the first N elements for grad_beta, the rest are just padding
        grad_ms = torch.zeros((M,), device=grad_out.device, dtype=grad_out.dtype)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        _mhc_scale_bwd_fused[grid](
            grad_out_ptr=grad_out,
            out_ptr=out,
            grad_h_ptr=grad_h,
            h_ptr=H,
            grad_a_ptr=grad_alpha,
            a_ptr=alpha,
            grad_b_ptr=grad_beta,
            grad_ms_ptr=grad_ms,
            ms_ptr=ms,
            M=M,
            n=n,
            stride_grad_out_m=32,
            stride_grad_out_n=1,
            stride_out_m=32,
            stride_out_n=1,
            stride_grad_hm=32,
            stride_grad_hn=1,
            stride_hm=32,
            stride_hn=1,
            stride_grad_a=1,
            stride_a=1,
            stride_grad_b=1,
            stride_grad_ms=1,
            stride_ms=1,
            BLOCK_SIZE_N=32,
            eps=torch.finfo(ms.dtype).eps,
        )

        return (
            grad_h.to(ctx.dtype),
            grad_alpha.to(ctx.dtype),
            grad_beta.to(ctx.dtype),
            grad_ms.to(ctx.dtype),
            None,
        )


class mHCSinkhornOp(torch.autograd.Function):
    """
    Sinkhorn operation to compute the final H_res matrix (see eq. 19, section 4.3.1 of the DeepSeek mHC paper)
    :param H_res: input H_res matrix of shape (M, n*n)
    :param n: number of hyper connections, where only n=4 is supported in the current implementation
    :param recompute_hist: whether to recompute the intermediate history in the backward pass to save memory
    :param iters: number of Sinkhorn iterations, according to the DeepSeek paper 20 is enough for convergence

    Sinkhorn operation conducts iterative normalization process that alternately rescales rows and columns to sum to 1.
    This kernel performance this operation in the log space for numerical stability.

    :return: out of shape (s, b, n, n), which is the final H_res after Sinkhorn normalization
    """

    @staticmethod
    def forward(ctx, H_res, n=4, recompute_hist=True, iters=20):
        assert n == 4, "Only n=4 is supported in this implementation"

        s, b, _, _ = H_res.shape

        ctx.dtype = H_res.dtype
        H_res = H_res.to(torch.float32)

        H_res = H_res.contiguous().view(s * b, n * n)

        hist_f, hist_g = None, None
        if not recompute_hist:
            # History buffers: (iters+1, s, b, n)
            hist_f = torch.empty((iters + 1, s, b, n), device=H_res.device, dtype=H_res.dtype)
            hist_g = torch.empty((iters + 1, s, b, n), device=H_res.device, dtype=H_res.dtype)
        H_res_out = torch.empty_like(H_res)  # (s*b, n*n)

        grid = lambda meta: (triton.cdiv(s * b * n * n, meta["BLOCK_SIZE"]),)

        if recompute_hist:
            _mhc_sinkhorn_fwd_fused_recompute[grid](
                x_ptr=H_res,
                output_ptr=H_res_out,
                stride_xm=n * n,
                stride_xn=1,
                stride_out_m=n * n,
                stride_out_n=1,
                M=s * b,
                n=n,
                iters=iters,
            )
        else:
            _mhc_sinkhorn_fwd_fused[grid](
                x_ptr=H_res,
                output_ptr=H_res_out,
                hist_f_ptr=hist_f,
                hist_g_ptr=hist_g,
                stride_xm=n * n,
                stride_xn=1,
                stride_out_m=n * n,
                stride_out_n=1,
                M=s * b,
                n=n,
                iters=iters,
            )

        if recompute_hist:
            ctx.save_for_backward(H_res, H_res_out)
        else:
            ctx.save_for_backward(H_res, H_res_out, hist_f, hist_g)
        ctx.recompute_hist = recompute_hist
        ctx.iters = iters
        ctx.n = n

        H_res_out = H_res_out.view(s, b, n, n)
        return H_res_out.to(ctx.dtype)  # Cast back to the original dtype of H

    @staticmethod
    def backward(ctx, grad_out):

        s, b, n, _ = grad_out.shape
        M = s * b

        hist_f, hist_g = None, None
        recompute_hist = ctx.recompute_hist
        iters = ctx.iters
        if recompute_hist:
            H_res, H_res_out = ctx.saved_tensors
            hist_f = torch.empty((iters + 1, s, b, n), device=H_res.device, dtype=H_res.dtype)
            hist_g = torch.empty((iters + 1, s, b, n), device=H_res.device, dtype=H_res.dtype)
        else:
            H_res, H_res_out, hist_f, hist_g = ctx.saved_tensors

        iters = ctx.iters
        n = ctx.n

        grad_res_out = grad_out.clone().contiguous().view(M, n * n)

        grad_res = torch.empty_like(H_res)

        grid = lambda meta: (triton.cdiv(M * n * n, meta["BLOCK_SIZE"]),)

        if recompute_hist:
            _mhc_sinkhorn_bwd_fused_recompute[grid](
                grad_out_ptr=grad_res_out,
                output_ptr=H_res_out,
                grad_x_ptr=grad_res,
                x_ptr=H_res,
                hist_f_ptr=hist_f,
                hist_g_ptr=hist_g,
                stride_grad_out_m=n * n,
                stride_grad_out_n=1,
                stride_out_m=n * n,
                stride_out_n=1,
                stride_grad_xm=n * n,
                stride_grad_xn=1,
                stride_xm=n * n,
                stride_xn=1,
                M=M,
                n=n,
                iters=iters,
            )
        else:
            _mhc_sinkhorn_bwd_fused[grid](
                grad_out_ptr=grad_res_out,
                output_ptr=H_res_out,
                grad_x_ptr=grad_res,
                x_ptr=H_res,
                hist_f_ptr=hist_f,
                hist_g_ptr=hist_g,
                stride_grad_out_m=n * n,
                stride_grad_out_n=1,
                stride_out_m=n * n,
                stride_out_n=1,
                stride_grad_xm=n * n,
                stride_grad_xn=1,
                stride_xm=n * n,
                stride_xn=1,
                M=M,
                n=n,
                iters=iters,
            )

        grad_res = grad_res.view(s, b, n, n)

        return grad_res.to(ctx.dtype), None, None, None


class mHCAggregateOp(torch.autograd.Function):
    """
    Aggregate operation to merge n activation streams to one (see section 4.3.1 of the DeepSeek mHC paper)
    :param x: input activation tensor of shape (s, b, C, n), 
        where s is the sequence length, b is the batch size, C is the hidden dimension per hyper connection, and n is the number of hyper connections. Note that C is equal to the original hidden dimension divided by n.
    :param H_pre: input H_pre matrix of shape (s, b, n)
    :param n: number of hyper connections, where only n=4 is supported in the current implementation
    :param use_tf32: whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision. 
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail

    out = x @ H_pre: (s, b, C, n) @ (s, b, n, 1) -> (s, b, C, 1) -> (s, b, C) after squeezing the last dimension

    :return: out of shape (s, b, C), which is the aggregated output after merging n hyper connections
    """

    @staticmethod
    def forward(ctx, x, H_pre, n, use_tf32=True):
        assert n == 4, "Only n=4 is supported in this implementation"

        x = x.contiguous()
        H_pre = (
            H_pre.contiguous()
        )

        s, b, C, n = x.shape
        nC = n * C
        M = s * b

        out = torch.empty((s, b, C), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
        )

        _mhc_aggregate_fwd[grid](
            x_ptr=x,
            H_pre_ptr=H_pre,
            output_ptr=out,
            M=M,
            C=C,
            n=n,
            stride_xm=nC,
            stride_xCn=1,
            stride_output_m=C,
            stride_output_c=1,
        )

        ctx.save_for_backward(x, H_pre)
        ctx.n = n
        ctx.use_tf32 = use_tf32

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()

        x, H_pre = ctx.saved_tensors
        n = ctx.n

        s, b, C, n = x.shape
        nC = n * C
        assert n == 4, "Only n=4 is supported in this implementation"
        M = s * b

        grad_x = torch.empty_like(x)
        grad_H_pre = torch.zeros(
            (s, b, n), dtype=torch.float32, device=H_pre.device
        )  # We need to use atomic_add for this so we need higher precision

        grid = lambda META: (
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
        )

        if ctx.use_tf32:
            _mhc_aggregate_bwd[grid](
                grad_output_ptr=grad_output,
                H_pre_ptr=H_pre,
                grad_H_pre_ptr=grad_H_pre,
                x_ptr=x,
                grad_x_ptr=grad_x,
                M=M,
                C=C,
                n=n,
                stride_grad_output_m=C,
                stride_grad_output_c=1,
                stride_xm=nC,
                stride_xCn=1,
                stride_grad_xm=nC,
                stride_grad_xCn=1,
                precision="tf32",
            )
        else:
            _mhc_aggregate_bwd[grid](
                grad_output_ptr=grad_output,
                H_pre_ptr=H_pre,
                grad_H_pre_ptr=grad_H_pre,
                x_ptr=x,
                grad_x_ptr=grad_x,
                M=M,
                C=C,
                n=n,
                stride_grad_output_m=C,
                stride_grad_output_c=1,
                stride_xm=nC,
                stride_xCn=1,
                stride_grad_xm=nC,
                stride_grad_xCn=1,
                precision="ieee",
            )

        grad_H_pre = grad_H_pre.to(H_pre.dtype)  # Cast back to the original dtype of H_pre

        return grad_x, grad_H_pre, None, None


class mHCExpandCombineOp(torch.autograd.Function):
    """
    Expand and combine operation for merging n hyper connections (see section 4.3.1 of the DeepSeek mHC paper)
    :param f: input activation tensor of shape (s, b, C), which is the output from the attention / FFN sub-layer in a transformer block
    :param bias: optional bias tensor of shape C from the last linear layer, where f + bias is fused in this kernel for better performance
    :param H_post: input H_post matrix of shape (s, b, n)
    :param x: input activation tensor of shape (s, b, C, n), which is the hyper connection input before the aggregation operation
    :param H_res: input H_res matrix of shape (s, b, n)
    :param n: number of hyper connections
    :param use_tf32: whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision. 
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail
    
    out = (f [+ bias]) @ H_post + x @ H_res: (s, b, C, 1) @ (s, b, 1, n) + (s, b, C, n) @ (s, b, n, n) -> (s, b, C, n)
        
    :return: out of shape (s, b, C, n), which is the expanded and combined output after merging n hyper connections
    """

    @staticmethod
    def forward(ctx, f, bias, H_post, x, H_res, n, use_tf32=True):
        assert n == 4, "Only n=4 is supported in this implementation"

        x = x.contiguous()
        f = f.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        H_post = H_post.contiguous()
        H_res = H_res.contiguous()

        s, b, C, n = x.shape
        Cn = C * n
        M = s * b

        out = torch.empty((s, b, C, n), device=x.device, dtype=x.dtype)

        if bias is None:
            # If no bias then we can use the naive grid where triton will launch blocks in C direction first
            # In this case it's more cache friendly for H
            grid = lambda META: (
                triton.cdiv(C, META["BLOCK_SIZE_C"]),
                triton.cdiv(M, META["BLOCK_SIZE_M"]),
            )
            _mhc_expand_combine_fwd[grid](
                f_ptr=f,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                output_ptr=out,
                M=M,
                C=C,
                n=n,
                stride_fm=C,
                stride_fc=1,
                stride_xm=Cn,
                stride_xCn=1,
                stride_output_m=Cn,
                stride_output_Cn=1,
            )
        else:
            # If bias is present then we need use the grouped order since launching in one direction will 
            # cause cache thrashing for either H or bias
            grid = lambda META: (
                triton.cdiv(C, META["BLOCK_SIZE_C"]),
                triton.cdiv(M, META["BLOCK_SIZE_M"]),
            )
            _mhc_expand_combine_with_bias_fwd[grid](
                f_ptr=f,
                bias_ptr=bias,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                output_ptr=out,
                M=M,
                C=C,
                n=n,
                stride_fm=C,
                stride_fc=1,
                stride_bias=1,
                stride_xm=Cn,
                stride_xCn=1,
                stride_output_m=Cn,
                stride_output_Cn=1,
            )

        ctx.n = n
        ctx.have_bias = bias is not None
        if bias is not None:
            ctx.save_for_backward(f, bias, H_post, x, H_res)
        else:
            ctx.save_for_backward(f, H_post, x, H_res)
        ctx.use_tf32 = use_tf32

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        s, b, C, n = grad_output.shape

        if ctx.have_bias:
            f, bias, H_post, x, H_res = ctx.saved_tensors
        else:
            bias = None
            f, H_post, x, H_res = ctx.saved_tensors
        M = s * b

        grad_f = torch.empty_like(f)
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else None
        grad_H_post = torch.zeros_like(
            H_post, dtype=torch.float32
        )  # We need to use atomic_add for this so we need higher precision
        grad_x = torch.empty_like(x)
        grad_H_res = torch.zeros_like(
            H_res, dtype=torch.float32
        )  # We need to use atomic_add for this so we need higher precision

        grid = lambda META: (
            triton.cdiv(C, META["BLOCK_SIZE_C"]),
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
        )

        if ctx.use_tf32:
            if bias is None:
                _mhc_expand_combine_bwd[grid](
                    grad_output_ptr=grad_output,
                    f_ptr=f,
                    H_post_ptr=H_post,
                    x_ptr=x,
                    H_res_ptr=H_res,
                    grad_H_post_ptr=grad_H_post,
                    grad_f_ptr=grad_f,
                    grad_H_res_ptr=grad_H_res,
                    grad_x_ptr=grad_x,
                    M=M,
                    C=C,
                    n=n,
                    stride_grad_output_m=n * C,
                    stride_grad_output_Cn=1,
                    stride_fm=C,
                    stride_fc=1,
                    stride_xm=n * C,
                    stride_xCn=1,
                    stride_grad_fm=C,
                    stride_grad_fc=1,
                    stride_grad_xm=n * C,
                    stride_grad_xCn=1,
                    precision="tf32",
                )
            else:
                _mhc_expand_combine_with_bias_bwd[grid](
                    grad_output_ptr=grad_output,
                    f_ptr=f,
                    bias_ptr=bias,
                    H_post_ptr=H_post,
                    x_ptr=x,
                    H_res_ptr=H_res,
                    grad_H_post_ptr=grad_H_post,
                    grad_f_ptr=grad_f,
                    grad_bias_ptr=grad_bias,
                    grad_H_res_ptr=grad_H_res,
                    grad_x_ptr=grad_x,
                    M=M,
                    C=C,
                    n=n,
                    stride_grad_output_m=n * C,
                    stride_grad_output_Cn=1,
                    stride_fm=C,
                    stride_fc=1,
                    stride_bias=1,
                    stride_xm=n * C,
                    stride_xCn=1,
                    stride_grad_fm=C,
                    stride_grad_fc=1,
                    stride_grad_bias=1,
                    stride_grad_xm=n * C,
                    stride_grad_xCn=1,
                    precision="tf32",
                )
        else:
            if bias is None:
                _mhc_expand_combine_bwd[grid](
                    grad_output_ptr=grad_output,
                    f_ptr=f,
                    H_post_ptr=H_post,
                    x_ptr=x,
                    H_res_ptr=H_res,
                    grad_H_post_ptr=grad_H_post,
                    grad_f_ptr=grad_f,
                    grad_H_res_ptr=grad_H_res,
                    grad_x_ptr=grad_x,
                    M=M,
                    C=C,
                    n=n,
                    stride_grad_output_m=n * C,
                    stride_grad_output_Cn=1,
                    stride_fm=C,
                    stride_fc=1,
                    stride_xm=n * C,
                    stride_xCn=1,
                    stride_grad_fm=C,
                    stride_grad_fc=1,
                    stride_grad_xm=n * C,
                    stride_grad_xCn=1,
                    precision="ieee",
                )
            else:
                _mhc_expand_combine_with_bias_bwd[grid](
                    grad_output_ptr=grad_output,
                    f_ptr=f,
                    bias_ptr=bias,
                    H_post_ptr=H_post,
                    x_ptr=x,
                    H_res_ptr=H_res,
                    grad_H_post_ptr=grad_H_post,
                    grad_f_ptr=grad_f,
                    grad_bias_ptr=grad_bias,
                    grad_H_res_ptr=grad_H_res,
                    grad_x_ptr=grad_x,
                    M=M,
                    C=C,
                    n=n,
                    stride_grad_output_m=n * C,
                    stride_grad_output_Cn=1,
                    stride_fm=C,
                    stride_fc=1,
                    stride_bias=1,
                    stride_xm=n * C,
                    stride_xCn=1,
                    stride_grad_fm=C,
                    stride_grad_fc=1,
                    stride_grad_bias=1,
                    stride_grad_xm=n * C,
                    stride_grad_xCn=1,
                    precision="ieee",
                )

        grad_H_post = grad_H_post.to(H_post.dtype)  # Cast back to the original dtype of H_post
        grad_H_res = grad_H_res.to(H_res.dtype)  # Cast back to the original dtype of H_res
        if bias is not None:
            grad_bias = grad_bias.to(bias.dtype)

        return grad_f, grad_bias, grad_H_post, grad_x, grad_H_res, None, None
