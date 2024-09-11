import torch
from torch import nn

from utils.assoc_scan import associative_scan as parallel_scan


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
        _, _, res = parallel_scan(binf, (gammas, R, Us), reverse=reverse)
        return res
