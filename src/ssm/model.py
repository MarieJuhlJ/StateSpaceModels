from torch import nn
from ssm.hippo import make_DPLR_HiPPO
from ssm.kernel import fourier_kernel_DPLR
import lightning as L
import torch
from ssm.hippo import make_DPLR_HiPPO

class S4Layer(nn.Module):
    def __init__(self, N, H, L):
        super(S4Layer, self).__init__()

        self.N = N
        self.H = H
        self.L = L

        Lambda, P, B, _ = make_DPLR_HiPPO(N)
        
        self.P = nn.Parameter(self.P)
        self.B = nn.Parameter(self.B)
        self.Lambda = nn.Parameter(self.Lambda)

        # Define C as a complex number with real and imaginary parts sampled from a normal distribution with mean 0 and std 0.5**0.5
        self.C = nn.Parameter(torch.randn(N, 2) * 0.5**0.5)
        self.C = self.C[:, 0] + 1j * self.C[:, 1]

        self.step = 1e-3 # Should this be a learnable parameter?

    def forward(self, x):
        kernel_DPLR()







class S4(nn.Module):
    def __init__(self, input_dim, output_dim, H):
        super(S4, self).__init__()

        self.layers = nn.ModuleList([S4Layer(input_dim, output_dim) for _ in range(H)])


