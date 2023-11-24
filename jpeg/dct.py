import torch


def dct1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Perform a DCT on a tensor.

    Args:
        x (torch.Tensor): The tensor to perform the DCT on with shape :math:`(*, N, *)`.
        dim (int): The dimension to perform the DCT on.

    Returns:
        torch.Tensor: The DCT coefficients with shape :math:`(*, N, *)`.
    """
    shape = x.shape
    dim = dim % len(shape)
    permutation = [i for i in range(len(shape)) if i != dim] + [dim]
    N = shape[dim]
    x = x.permute(permutation)                                                  # * x N
    permuted_shape = x.shape
    x = x.contiguous().reshape(-1, N)                                           # L x N

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)             # L x N
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))                            # L x N x 2

    k = - torch.arange(N, dtype=x.dtype, device=x.device) * torch.pi / (2 * N)  # N
    W_r = torch.cos(k)                                                          # N
    W_i = torch.sin(k)                                                          # N

    V = Vc[..., 0] * W_r - Vc[..., 1] * W_i                                     # L x N

    # normalize
    V[:, 0] /= torch.sqrt(torch.tensor(N, dtype=V.dtype, device=V.device)) * 2
    V[:, 1:] /= torch.sqrt(torch.tensor(N / 2, dtype=V.dtype, device=V.device)) * 2

    V = 2 * V.reshape(permuted_shape)                                           # * x N
    permutation = [permutation.index(i) for i in range(len(shape))]
    V = V.permute(permutation)                                                  # * x N x *
    return V


def dct2d(x: torch.Tensor) -> torch.Tensor:
    """Perform a 2D DCT on a tensor.

    Args:
        x (torch.Tensor): The tensor to perform the DCT on with shape :math:`(*, M, N)`.

    Returns:
        torch.Tensor: The DCT coefficients with shape :math:`(*, M, N)`.
    """
    return dct1d(dct1d(x, dim=-2), dim=-1)


def idct1d(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Perform an inverse DCT on a tensor.

    Args:
        X (torch.Tensor): The tensor to perform the inverse DCT on with shape :math:`(*, N, *)`.
        dim (int): The dimension to perform the inverse DCT on.

    Returns:
        torch.Tensor: The inverse DCT coefficients with shape :math:`(*, N, *)`.
    """
    shape = X.shape
    dim = dim % len(shape)
    permutation = [i for i in range(len(shape)) if i != dim] + [dim]
    N = shape[dim]
    X = X.permute(permutation)                                                  # * x N
    permuted_shape = X.shape
    X_v = X.contiguous().reshape(-1, N) / 2                                     # L x N

    # normalize
    X_v[:, 0] *= torch.sqrt(torch.tensor(N, dtype=X.dtype, device=X.device)) * 2
    X_v[:, 1:] *= torch.sqrt(torch.tensor(N / 2, dtype=X.dtype, device=X.device)) * 2

    k = torch.arange(N, dtype=X.dtype, device=X.device) * torch.pi / (2 * N)    # N
    W_r = torch.cos(k)                                                          # N
    W_i = torch.sin(k)                                                          # N

    V_t_r = X_v
    V_t_i = torch.cat([torch.zeros_like(X_v[:, :1]), -X_v.flip([1])[:, :-1]], dim=1)  # L x N

    V_r = V_t_r * W_r - V_t_i * W_i                                             # L x N
    V_i = V_t_r * W_i + V_t_i * W_r                                             # L x N

    V = torch.stack([V_r, V_i], dim=-1)                                 # L x N x 2

    v = torch.fft.irfft(torch.view_as_complex(V), dim=1, n=N)                   # L x N
    x = torch.zeros_like(v)                                                     # L x N
    x[:, ::2] = v[:, :N - N // 2]
    x[:, 1::2] = v.flip([1])[:, :N // 2]

    x = x.reshape(permuted_shape)                                               # * x N
    permutation = [permutation.index(i) for i in range(len(shape))]
    x = x.permute(permutation)                                                  # * x N x *
    return x


def idct2d(X: torch.Tensor) -> torch.Tensor:
    """Perform a 2D inverse DCT on a tensor.

    Args:
        X (torch.Tensor): The tensor to perform the inverse DCT on with shape :math:`(*, M, N)`.

    Returns:
        torch.Tensor: The inverse DCT coefficients with shape :math:`(*, M, N)`.
    """
    return idct1d(idct1d(X, dim=-1), dim=-2)
