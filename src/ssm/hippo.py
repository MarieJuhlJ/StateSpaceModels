""" Functions for creating hippo, NPLR hippo and DPLR hippo"""
import numpy as np
from scipy.linalg import eigh
import torch

def make_HiPPO(N):
    P = torch.sqrt(1 + 2 * torch.arange(N))
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(torch.arange(N))
    return -A

def make_NPLR_HiPPO(N):
    # First make a HiPPO matrix
    nhippo = make_HiPPO(N)
    
    # Create P and B matrices
    P = torch.sqrt(torch.arange(N) + 0.5)
    B = torch.sqrt(2 * torch.arange(N) + 1.0)
    return nhippo, P, B

def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    # Construct the normal part of the normal + low rank matrix
    S = A + P[:, None] * P[None, :]

    # Check skew symmetry
    S_diag = S.diag()
    Lambda_real = S_diag.mean() * torch.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)
    V = torch.tensor(V, dtype=torch.cfloat)

    P = V.conj().T @ (P+0j)
    B = V.conj().T @ (B+0j)
    return Lambda_real + 1j * Lambda_imag, P, B, V


def test_nplr(N=8):
    A2, P, B = make_NPLR_HiPPO(N)
    Lambda, Pc, Bc, V = make_DPLR_HiPPO(N)
    Vc = V.conj().T
    P = P[:, None]
    Pc = Pc[:, None]
    Lambda = Lambda.diag()

    A3 = V @ Lambda @ Vc - (P @ P.T)  # Test NPLR
    A4 = V @ (Lambda - Pc @ Pc.conj().T) @ Vc  # Test DPLR
    assert np.allclose(A2, A3, atol=1e-4, rtol=1e-4)
    assert np.allclose(A2, A4, atol=1e-4, rtol=1e-4)
    print("NPLR and DPLR tests passed")

if __name__ == "__main__":
    print(make_DPLR_HiPPO(8))
    
