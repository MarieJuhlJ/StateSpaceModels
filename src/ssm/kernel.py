import torch
"""
Functions related to computing the kernel, convolutions etc. Like the ones in annotated:

K_gen_DPLR → Truncated generating function when A is DPLR (S4-part)
conv_from_gen → Convert generating function to filter
causal_convolution → Run convolution
discretize_DPLR → Convert SSM to discrete form for RNN.
"""

def cauchy(v, omegas, Lambda):
    out = torch.zeros_like(omegas)
    for i, omega in enumerate(omegas):
        out[i] = torch.sum(v / (omega - Lambda))
    return out
    

def fourier_kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = torch.exp((-2j * torch.pi) * (torch.arange(L) / L))

    aterm = torch.stack([C.conj(), Q.conj()], dim=0)
    bterm = torch.stack([P, B], dim=0)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    return atRoots

if __name__ == '__main__':
    Lambda = torch.tensor([1.0, 2.0, 3.0])
    P = torch.tensor([1.0, 2.0, 3.0])
    Q = torch.tensor([1.0, 2.0, 3.0])
    B = torch.tensor([1.0, 2.0, 3.0])
    C = torch.tensor([1.0, 2.0, 3.0])
    step = torch.tensor([1.0])
    L = 10
    print(fourier_kernel_DPLR(Lambda, P, Q, B, C, step, L))