from torch import nn
from ssm.hippo import make_DPLR_HiPPO
from ssm.kernel import fourier_kernel_DPLR
import lightning
import torch
from ssm.hippo import make_DPLR_HiPPO
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os

class S4Layer(nn.Module):
    """
    S4Layer: A single layer of the S4 model implementing a kernel based on a DPLR HiPPO matrix.
    
    Args:
        N: int: Size of hidden state space
        L: int: Sequence length
    """
    def __init__(self, N: int, L: int):
        super(S4Layer, self).__init__()

        self.N = N
        self.L = L

        Lambda, P, B, _ = make_DPLR_HiPPO(N)
        
        self.P = nn.Parameter(P)
        self.B = nn.Parameter(B)
        self.Lambda = nn.Parameter(Lambda)

        # Define C as a complex number with real and imaginary parts sampled from a normal distribution with mean 0 and std 0.5**0.5
        self.C_params = torch.randn(N, 2) * 0.5**0.5
        self.C_tilde = nn.Parameter(self.C_params[:, 0] + 1j * self.C_params[:, 1])

        self.step_size = 1e-3 # Should this be a learnable parameter?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fourier_kernel = fourier_kernel_DPLR(self.Lambda, self.P, self.P, self.B, self.C_tilde, self.step_size, self.L)
        fft_u = torch.fft.fft(x, self.L)
        y = torch.fft.ifft(fourier_kernel * fft_u, self.L)
        return y.real
     
class S4sequence(nn.Module):
    """
    S4sequence: A sequence block encasing an S4 layer for each feature with normalization, 
    position-wise linear layers and dropout.

    Args:
        N: int: Size of hidden state space
        H: int: Number of features
        L: int: Sequence length
    """
    def __init__(self, N: int, H: int, L:int, glu:bool=True):
        super(S4sequence, self).__init__()
        self.use_glu = glu

        self.norm = nn.LayerNorm((L,H))
        self.s4layers = nn.ModuleList([S4Layer(N, L) for _ in range(H)])
        #self.layers = torch.vmap(S4Layer, in_dims=1 , out_dims=1)(N, H, L)
        self.activation = nn.GELU() # Perhaps replace with ReLU based on "ReLU strikes back" paper (on LLM tasks)
        self.out1 = nn.Linear(H, H)
        if self.use_glu:
            self.out2 = nn.Linear(H, H)
            #self.glu = nn.GLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm(x)

        # Loop through the S4 layers for each of the H features
        outputs =[]
        for i, layer in enumerate(self.s4layers):
            outputs.append(layer(x[:,:,i]).unsqueeze(-1))

        x = torch.cat(outputs, dim=-1)
        x = self.activation(x)
        x = self.dropout(x)

        if self.use_glu:
            #x = self.glu(torch.cat((self.out1(x), self.out2(x)),dim=-1))
            x = self.out1(x)*nn.Sigmoid()(self.out2(x)) # a gated linear unit
        else:
            x = self.out1(x)

        return x + skip

class S4Model(lightning.LightningModule):
    """
    S4Model: The S4 model consisting of an encoder, a stack of S4 sequences and a classifier.
    
    Args:
        N: int: Size of hidden state space
        H: int: Number of features
        L: int: Sequence length
        num_blocks: int: Number of S4 sequence blocks
        cls_out: int: Number of classes
    """
    def __init__(self,  N: int, H:int, L:int, num_blocks:int, cls_out:int):
        super(S4Model, self).__init__()
        
        self.enc = nn.Linear(1,H)
        self.blocks = nn.ModuleList([S4sequence(N, H, L) for _ in range(num_blocks)])
        self.cls = nn.Linear(H, cls_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x.unsqueeze(-1))
        for block in self.blocks:
            x = block(x)
        x = torch.mean(x, dim=1)
        x = self.cls(x)
        x = nn.Softmax(dim=-1)(x)
        return x
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss
    

if __name__=="main":
    N = 64
    H = 128
    L = 784
    num_blocks = 4
    class_out = 10

    x = torch.randn(1, 1, 784)

    model = S4Model(N=N, H=H, L=L, num_blocks=num_blocks, cls_out=class_out)
    
    y = S4Model(x)


