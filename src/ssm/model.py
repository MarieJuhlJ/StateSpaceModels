from torch import nn
from ssm.hippo import make_DPLR_HiPPO
from ssm.kernel import fourier_kernel_DPLR
import pytorch_lightning as L
import torch
from ssm.hippo import make_DPLR_HiPPO
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os

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
        self.C_tilde = nn.Parameter(torch.randn(N, 2) * 0.5**0.5)
        self.C_tilde = self.C_tilde[:, 0] + 1j * self.C_tilde[:, 1]

        self.step_size = 1e-3 # Should this be a learnable parameter?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fourier_kernel = fourier_kernel_DPLR(self.Lambda, self.P, self.P, self.B, self.C_tilde, self.step_size, self.L)
        fft_u = torch.fft.fft(x, self.L)
        y = torch.fft.ifft(fourier_kernel * fft_u, self.L)
        return y.real
     
class S4sequence(nn.Module):
    def __init__(self, N, H, L):
        super(S4sequence, self).__init__()

        self.norm = nn.LayerNorm()
        self.layers = nn.vmap(S4Layer, in_dims=1, out_dims=1)(N, H, L)
        self.linear = nn.Linear(H, H)
        self.activation = nn.GLU()
        self.output = nn.Linear(H, H)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm(x)
        x = self.layers(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.dropout(x)
        return x + skip

class S4Model(L.LightningModule):
    def __init__(self, N: int, H:int, L:int, num_blocks:int, cls_out:int):
        super(S4Model, self).__init__()
        
        self.enc = nn.Linear(1, H)
        self.blocks = nn.ModuleList([S4sequence(N, H, L) for _ in range(num_blocks)])
        self.cls = nn.Linear(H, cls_out)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        for block in self.blocks:
            x = block(x)
        x = torch.mean(x, dim=1)
        x = self.cls(x)
        x = nn.Softmax(x)
        return x
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss(y_hat, y)
        return loss
    

if __name__=="main":
    N = 64
    H = 128
    L = 768
    num_blocks = 4
    class_out = 10

    train_loader = DataLoader(MNIST(os.path.join(os.getcwd(), "data"),train=True, download=True, transform=transforms.ToTensor()))
    test_loader = DataLoader(MNIST(os.path.join(os.getcwd(), "data"),train=False, download=True, transform=transforms.ToTensor()))

    model = S4Model(N=N, H=H, L=L, num_blocks=num_blocks, cls_out=class_out)
    trainer = L.Trainer()
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)



