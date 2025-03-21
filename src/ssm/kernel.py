import torch
import torch.nn as nn

class s4_kernel(nn.Module):
    def __init__(self, L: int):
        super(s4_kernel, self).__init__()
        Omega_L = torch.exp((-2j * torch.pi) * (torch.arange(L) / L))
        c = 2.0 / (1.0 + Omega_L)
        self.register_buffer("Omega_L", Omega_L)
        self.register_buffer("c", c)

    def _cauchy(self, v, omegas, Lambda):
        return torch.sum(v[:, None] / (omegas[None, :] - Lambda[:, None]), dim=0)
    
    def __call__(self, Lambda, P, Q, B, C, step):
        aterm = torch.stack([C.conj(), Q.conj()], dim=0)
        bterm = torch.stack([B, P], dim=0)
        g = (2.0 / step) * ((1.0 - self.Omega_L) / (1.0 + self.Omega_L))

        k00 = self._cauchy(aterm[0] * bterm[0], g, Lambda)
        k01 = self._cauchy(aterm[0] * bterm[1], g, Lambda)
        k10 = self._cauchy(aterm[1] * bterm[0], g, Lambda)
        k11 = self._cauchy(aterm[1] * bterm[1], g, Lambda)

        atRoots = self.c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

        return atRoots
    
class s4dss_kernel(nn.Module):
    def __init__(self, L: int):
        super(s4dss_kernel, self).__init__()
        self.L = L
        self.pos = torch.arange(self.L)

    def _reciprocal(self, x, eps=1e-6):
        return x.conj() / (x.real**2 + x.imag**2 + eps)
    
    def __call__(self, Lambda, w, step):
        dt_Lambda = Lambda * step
        P = self.pos[None, :] * dt_Lambda[:, None]
        Lambda_gt = Lambda.real > 0
        if Lambda_gt.any():
            P_max = dt_Lambda * (self.L - 1) * Lambda_gt
            P = P - P_max[:, None]
        P.exp()
        dt_Lambda_neg = dt_Lambda * (1 - 2*Lambda_gt) #Translate with L-1, summing the opposite way?.
        num = dt_Lambda_neg.exp() - 1
        den = (dt_Lambda_neg * self.L).exp() - 1
        w = w * num * self._reciprocal(den * Lambda)
        WSLinv = w[:, None] * P.exp()
        return WSLinv.sum(0).real

if __name__ == '__main__':
    s4dss = s4dss_kernel(10)
    Lambda = torch.tensor([1.0 + 1j, 2.0 + 1j, -3.0 + 1j])
    w = torch.tensor([4.0, 2.0, 3.0])
    step = 0.1
    kernel = s4dss(Lambda, w, step)
    print(kernel)