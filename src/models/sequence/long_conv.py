from typing import Callable

import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

from src.models.nn import Activation, DropoutNd, LinearActivation
from src.models.sequence.block_fft import BlockFFT
from src.models.sequence.long_conv_kernel import LongConvKernel


class LongConv(nn.Module):
    def __init__(
        self,
        d_model,
        l_max=1024,
        channels=1,
        bidirectional=False,
        # Arguments for position-wise feedforward components
        activation="gelu",  # activation between conv and FF
        postact="glu",  # activation after FF
        initializer=None,  # initializer on FF
        weight_norm=False,  # weight normalization on FF
        dropout=0.0,
        tie_dropout=False,
        transposed=True,  # axis ordering (B, L, D) or (B, D, L)
        verbose=False,
        block_fft_conv=False,  # replace the FFT conv with Monarch blocks
        block_fft_conv_args={},
        # SSM Kernel arguments
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided

        Position-wise feedforward components:
        --------------------
        activation: activation in between SS and FF
        postact: activation after FF ('id' for no activation, None to remove FF layer)
        initializer: initializer on FF
        weight_norm: weight normalization on FF
        dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

        Other arguments:
        --------------------
        transposed: choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=hidden dimension]
        """

        super().__init__()
        if verbose:
            import src.utils.train

            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing Long Conv (H, L) = ({d_model}, {l_max})")

        self.d_model = d_model
        self.H = d_model
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.block_fft_conv = block_fft_conv
        self.block_fft_conv_args = block_fft_conv_args

        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = LongConvKernel(
            self.H, L=self.L, channels=channels, verbose=verbose, **kernel_args
        )

        if self.block_fft_conv:
            self.block_fft_u = BlockFFT(**self.block_fft_conv_args)
            self.block_fft_k = BlockFFT(**self.block_fft_conv_args)

        # Pointwise
        self.activation = Activation(activation)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model * self.channels,
                self.d_model,
                # self.H*self.channels,
                # self.d_model*(1 if self.gate is None else self.gate),
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )

    def forward(
        self, u, state=None, rate=1.0, lengths=None, **kwargs
    ):  # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed, remnant from state spaces repo

        Returns: same shape as u
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
            else:
                lengths = None
        if lengths is not None:
            assert (
                isinstance(lengths, torch.Tensor)
                and lengths.ndim == 1
                and lengths.size(0) in [1, u.size(0)]
            )
            mask = torch.where(
                torch.arange(L, device=lengths.device) < lengths[:, None, None],
                1.0,
                0.0,
            )
            u = u * mask

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, _ = self.kernel(L=L_kernel, rate=rate, state=state)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))

        if self.block_fft_conv:
            k_f = self.block_fft_k(k.to(torch.complex64), N=L_kernel + L)  # (C H L)
            u_f = self.block_fft_u(u.to(torch.complex64), N=L_kernel + L)  # (B H L)
            y_f = contract("bhl,chl->bchl", u_f, k_f)
            if self.learn_ifft:
                y = self.block_fft_u(y_f, N=L_kernel + L, forward=False).real[..., :L]
            else:
                y = torch.fft.ifft(y_f, n=L_kernel + L, dim=-1).real[
                    ..., :L
                ]  # (B C H L)
        else:
            k_f = torch.fft.rfft(k, n=L_kernel + L)  # (C H L)
            u_f = torch.fft.rfft(u, n=L_kernel + L)  # (B H L)
            y_f = contract("bhl,chl->bchl", u_f, k_f)
            y = torch.fft.irfft(y_f, n=L_kernel + L)[..., :L]  # (B C H L)

        # Compute skip connection
        y = y + contract("bhl,ch->bchl", u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, "... c h l -> ... (c h) l")

        if not self.transposed:
            y = y.transpose(-1, -2)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.output_linear(y)

        return y, None

    @property
    def d_state(self):
        return self.H

    @property
    def d_output(self):
        return self.d_model


def _interleave(a, b):
    """
    Merge 2 tensors on the last dimension by interleaving elements:
    Args:
        a: torch.tensor(..., L)
            a = [1,2,3]
        b: torch.tensor(..., L)
            b = [4,5,6]
    Returns:
        res: torch.tensor(..., 2L)
            res = [1,4,2,5,3,6]
    """
    assert a.shape[-1] == b.shape[-1] or a.shape[-1] == b.shape[-1] + 1
    add = False
    dim = len(b.shape) - 1
    if a.shape[-1] == b.shape[-1] + 1:
        b = torch.cat([b, torch.zeros_like(b[..., 0].unsqueeze(dim=-1))], dim=dim)
        add = True
    dims = list(a.shape)
    dims[-1] = a.shape[-1] + b.shape[-1]
    res = torch.zeros(size=dims, dtype=a.dtype, device=a.device)
    mask = torch.arange(start=0, end=dims[-1], step=2, device=a.device)
    res[..., mask] = a
    res[..., mask + 1] = b
    if add:
        return res[..., :-1]
    return res


def _combine(fn, a_flat, b_flat):
    """
    fn each element of a with each element of b.
    TODO: Can parallelize here
    Args:
        fn: Callable
            Associative binary function to apply on last dimension of a_flat and b_flat
        a_flat: torch.tensor(..., L)
            a = [1,2,3]
        b_flat: torch.tensor(..., L)
            b = [4,5,6]
    Return:
        res: torch.tensor(..., L)
            Reduced a_flat and b_flat
            res = [fn(1,4), fn(2,5), fn(3,6)]
    """
    fn_map = torch.vmap(fn, in_dims=(-1, -1))
    out = fn_map(a_flat, b_flat)
    # move the first dimension to last
    # vmap puts the vectorisation dimensions in front
    dims = list(range(a_flat.dim()))
    dims = [x + 1 for x in dims]
    dims[-1] = 0
    out = out.permute(dims)
    return out


def associative_scan(fn: Callable, elems):
    """
    Implement associative scan for a binary function fn.
    See notion for and explanation of the algorithm
    Args:
        fn: Callable
            Associative function to apply on last dimension of a_flat and b_flat
        elems: torch.tensor(..., L)
            a = [1,2,3]
    Return:
        res: torch.tensor(..., L)
            Prefix scan of the array.
            res[-1] = Identity fn
            res[i] = fn(res[i-1], a[i])
    """
    num_elems = elems.shape[-1]
    if num_elems < 2:
        return elems

    # compute sums of pairs on the branch level of the binary tree
    reduced_elems = _combine(fn, elems[..., 0:-1:2], elems[..., 1::2])
    # compute sum of sums
    odd_elements = associative_scan(fn, reduced_elems)
    # even elements = from_left + node sum (sums in between parent.left and current.left
    # as with recursion we only get subtrees
    if num_elems % 2 == 0:
        even_elems = _combine(fn, odd_elements[..., :-1], elems[..., 2::2])
    else:
        even_elems = _combine(fn, odd_elements, elems[..., 2::2])
    # merge odd and even
    # the first element of the scan is the same as the first element of the orignal
    dim = len(elems.shape) - 1
    even_elems = torch.cat([elems[..., 0].unsqueeze(dim=-1), even_elems], dim=dim)
    return _interleave(even_elems, odd_elements)


class RotSSM(nn.Module):
    def __init__(
        self,
        lru_dim: int = 64,  # devisible by heads
        hidden_dim: int = 128,  # devisible by heads
        nheads: int = 64,  # apply model in parallel
        r_min: float = 0.9,
        r_max: float = 0.999,
        max_phase: float = 6.28,
        bidirectional: bool = False,
        step_rescale: float = 0.0,
        transposed: bool = True,
    ):
        super().__init__()
        self.lru_dim = lru_dim
        self.hidden_dim = hidden_dim
        self.H = nheads
        self.N = self.lru_dim // self.H
        assert self.N % 2 == 0, "N should be even"
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase
        self.bidirectional = bidirectional
        self.step_rescale = step_rescale
        self.transposed = transposed

        self.thetas = nn.Parameter(
            self.theta_init(self.lru_dim // 2, self.max_phase)
        ).reshape(self.H, self.N // 2)
        self.P = nn.Parameter(self.ortho_mat_init(self.lru_dim, self.N)).reshape(
            self.H, self.N, self.N
        )
        self.B = nn.Parameter(self.mat_init(self.lru_dim, self.hidden_dim)).reshape(
            self.H, self.N, self.hidden_dim
        )
        self.C = nn.Parameter(self.mat_init(self.hidden_dim, self.lru_dim))
        if self.bidirectional:
            self.C2 = nn.Parameter(self.mat_init(self.hidden_dim, self.lru_dim))
        self.D = nn.Parameter(torch.randn(self.hidden_dim))
        self.gamma_log = nn.Parameter(
            self.nu_log_init(self.H, r_min=self.r_min, r_max=self.r_max)
        )

    def forward(self, u):
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed, remnant from state spaces repo

        Returns: same shape as u
        """
        print(u.shape)
        if self.transposed:
            batch_sz, hidden_dim, T = u.shape
            u = u.transpose(1, 2)
        else:
            batch_sz, T, hidden_dim = u.shape

        # do not forget the double exponential
        gamma_log = -torch.exp(self.gamma_log)
        gamma = torch.exp(gamma_log)

        trace_per_head = torch.trace(
            torch.einsum("HDd,HAd->HDA", self.B, self.B), axis1=-2, axis2=-1
        )
        norm = torch.sqrt((1 - gamma**2) / trace_per_head)  #  H / H elementwise -> H
        B_norm = torch.einsum("H,HnD->HnD", norm, self.B)
        P = torch.matrix_exp(self.P - self.P.transpose(1, 2))
        # apply P.T to Bx_t
        Us = torch.einsum("HnD,BTD->HTBn", B_norm, u)
        Us = torch.einsum("HnN,HTBn->HTBN", P.transpose(1, 2), Us)
        # mix per head
        mix_head_fn = torch.vmap(self.mix_sequence, in_dims=(0, 0, 0, None), out_dims=0)
        thetas = torch.exp(self.thetas)

        y = mix_head_fn(gamma, thetas, Us, False)  # H T B N
        # multiply P back to \tilde{x}_t
        y = torch.einsum("HNn,HTBN->HTBn", P, y)

        if self.bidirectional:
            backward = mix_head_fn(gamma, thetas, Us, True)  # H T B N
            # multiply P back to \tilde{x}_t
            backward = torch.einsum("HNn,HTBN->HTBn", P, backward)
            y = torch.concatenate([y, backward], axis=-1)
            C = torch.concatenate([self.C, self.C2], axis=-1)

        y = y.transpose(2, 1, 0, 3)  # H T B N -> B T H N
        y = torch.einsum("Dn,BTn->BTD", C, y.reshape(batch_sz, T, -1)) + self.D * u

        if self.transposed:
            y = y.transpose(1, 2)
        return y, None

    def theta_init(self, N, max_phase):
        return torch.log(torch.rand((N, 1)) * max_phase)

    def nu_log_init(self, H, r_min=0, r_max=1):
        """
        r_min, r_max in (0, 1)
        """
        u1 = torch.rand(H)
        # double exponential
        nu_log = torch.log(-torch.log(r_max)) + u1 * (
            torch.log(-torch.log(r_min)) - torch.log(-torch.log(r_max))
        )
        return nu_log

    def mat_init(self, lru_dim, hidden_dim):
        # Glorot initialized Input/Output projection matrices
        B = torch.randn((lru_dim, hidden_dim)) / torch.sqrt(hidden_dim + lru_dim)
        return B

    def ortho_mat_init(self, lru_dim, hidden_dim):
        # Glorot initialized Input/Output projection matrices
        B = torch.randn((lru_dim, hidden_dim)) / torch.sqrt(hidden_dim + lru_dim)
        return B

    @staticmethod
    def mix_sequence(gamma, R, Us, reverse=False):
        """
        N - per head dimension
        Args:
            gammas: jax.Array(T,)
            As: jax.Array(T,N,N)
            Us: jax.array(T,B,N)
        Returns:
            out: jax.array(T,B,N)
        """

        def binf(a, b):
            gamma_i, thetas_i, acc_i = a
            gamma_j, thetas_j, acc_j = b
            # R_j@acc_i + acc_j
            # get [-x2, x1, -x4, x3,...]
            rotate_half_mat_i = torch.stack(
                [-acc_i[..., 1::2], acc_i[..., 0::2]], axis=-1
            )
            shapes = list(rotate_half_mat_i.shape)[:-1]
            shapes[-1] *= 2
            rotate_half_mat_i = rotate_half_mat_i.reshape(shapes)
            # duplicate theta [o1, o1, o2, o2,...]
            shapes = list(thetas_j.shape)
            shapes[-1] *= 2
            theta = (
                thetas_j.unsqueeze(-1)
                .repeat([1 for _ in range(len(shapes) - 1)] + [2])
                .reshape(shapes)
            )
            sin = torch.sin(theta)[..., None, :]  # add mock batch dimension
            cos = torch.cos(theta)[..., None, :]  # add mock batch dimension
            acc = gamma_j[..., None, None] * (cos * acc_i + sin * rotate_half_mat_i)

            return (gamma_i * gamma_j, thetas_i + thetas_j, acc + acc_j)

        T = Us.shape[0]
        gamma_shape = list(gamma.shape)
        R_shape = list(R.shape)
        gammas = gamma.unsqueeze(0).repeat(
            [T] + [1 for _ in range(len(gamma_shape) - 1)]
        )
        R = R.unsqueeze(0).repeat([T] + [1 for _ in range(len(R_shape) - 1)])
        _, _, res = associative_scan(binf, (gammas, R, Us), reverse=reverse)
        return res
