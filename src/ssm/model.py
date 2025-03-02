from torch import nn
from ssm.hippo import make_DPLR_HiPPO
from ssm.kernel import fourier_kernel_DPLR
import lightning as L
import torch
import torch.nn as nn

class S4(L.LightningModule):
    def __init__(self, state_size: int, input_length: int, num_layers: int, num_blocks: int, cls_out: int):
        super(S4, self).__init__()
        self.state_size = state_size
        self.input_length = input_length
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.cls_out = cls_out
        self.blocks = nn.ModuleList([SSMBlock(state_size, input_length, num_layers) for _ in range(num_blocks)])
        self.fc = nn.Linear(input_length, cls_out)

    def training_step(self, batch, batch_idx):
        x, label = batch
        x_hat = x
        for block in self.blocks:
            x_hat = block(x_hat)
        x_hat = self.fc(x_hat)

        loss = nn.CrossEntropyLoss()(x_hat, label)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, label = batch
        x_hat = x
        for block in self.blocks:
            x_hat = block(x_hat)
        x_hat = self.fc(x_hat)

        correct = (x_hat.argmax(dim=1) == label).sum().item()
        total = len(label)
        self.log("val_acc", correct / total)
        return correct / total
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class SSMBlock(nn.Module):
    def __init__(self, state_size: int, input_length: int, num_layers: int):
        super(SSMBlock, self).__init__()
        self.state_size = state_size
        self.input_length = input_length
        self.num_layers = num_layers
        self.layers = nn.ModuleList([SSMLayer(state_size, input_length) for _ in range(num_layers)])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = u
            u = layer(u)
            u = u + residual
            u = self.gelu(u)
            u = self.dropout(u)
        return u

class SSMLayer(nn.Module):
    def __init__(self, state_size: int, input_length: int):
        super(SSMLayer, self).__init__()
        self.state_size = state_size
        self.input_length = input_length
        
        self.Lambda_real = nn.Parameter(torch.randn(state_size))
        self.Lambda_imag = nn.Parameter(torch.randn(state_size))
        self.step_size = nn.Parameter(torch.empty(1).uniform_(1e-3, 1e-1))
        self.P = nn.Parameter(torch.randn(state_size))
        self.B = nn.Parameter(torch.randn(state_size))
        self.C_tilde = nn.Parameter(torch.randn(state_size))

    def _init_matrices(self):
        Lambda_real, Lambda_imag, P, B, _ = make_DPLR_HiPPO(self.state_size)
        
        self.Lambda_real.data = torch.tensor(Lambda_real, dtype=torch.float32)
        self.Lambda_imag.data = torch.tensor(Lambda_imag, dtype=torch.float32)
        self.P.data = torch.tensor(P, dtype=torch.float32)
        self.B.data = torch.tensor(B, dtype=torch.float32)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        fourier_kernel = fourier_kernel_DPLR(self.Lambda_real, self.P, self.Lambda_imag, self.B, self.C_tilde, self.step_size, self.input_length)
        fft_u = torch.fft.fft(u, self.input_length)
        y = torch.fft.ifft(fourier_kernel * fft_u, self.input_length)
        return y.real

if __name__ == '__main__':
    ssm = SSMLayer(8, 10)
    u = torch.randn(2, 10)
    y = ssm(u)
    print(y)