import torch
import torch.nn as nn

class s4_kernel(nn.Module):
    def __init__(self, L: int):
        super(s4_kernel, self).__init__()
        Omega_L = torch.exp((-2j * torch.pi) * (torch.arange(L) / L))
        self.register_buffer("Omega_L", Omega_L)

    def _cauchy(self, v, omegas, Lambda):
        return torch.sum(v[:, None] / (omegas[None, :] - Lambda[:, None]), dim=0)
    
    def __call__(self, Lambda, P, Q, B, C, step):
        aterm = torch.stack([C.conj(), Q.conj()], dim=0)
        bterm = torch.stack([B, P], dim=0)

        g = (2.0 / step) * ((1.0 - self.Omega_L) / (1.0 + self.Omega_L))
        c = 2.0 / (1.0 + self.Omega_L)

        k00 = self._cauchy(aterm[0] * bterm[0], g, Lambda)
        k01 = self._cauchy(aterm[0] * bterm[1], g, Lambda)
        k10 = self._cauchy(aterm[1] * bterm[0], g, Lambda)
        k11 = self._cauchy(aterm[1] * bterm[1], g, Lambda)

        atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

        return atRoots