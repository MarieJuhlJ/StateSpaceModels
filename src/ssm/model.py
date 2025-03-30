from torch import nn
from ssm.hippo import make_DPLR_HiPPO
from ssm.kernel import s4_kernel, s4dss_kernel
from ssm.utils import LayerRegistry
import lightning
import torch
from ssm.hippo import make_DPLR_HiPPO

@LayerRegistry.register("s4")
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

        # Generate DPLR HiPPO matrices
        Lambda, P, B, _ = make_DPLR_HiPPO(N)
        
        # Separate real and imaginary parts and register as parameters
        self.Lambda_real = nn.Parameter(Lambda.real)
        self.Lambda_imag = nn.Parameter(Lambda.imag)
        self.P_real = nn.Parameter(P.real)
        self.P_imag = nn.Parameter(P.imag)
        self.B_real = nn.Parameter(B.real)
        self.B_imag = nn.Parameter(B.imag)

        # Initialize C
        self.C_real = nn.Parameter(torch.randn(N) * (0.5**0.5))
        self.C_imag = nn.Parameter(torch.randn(N) * (0.5**0.5))

        self.step_size = nn.Parameter(torch.tensor(1e-2))
        self.kernel = s4_kernel(L)

    def forward_recurrence(self, u: torch.Tensor) -> torch.Tensor:
        #UNDER CONSTRUCTION
        assert u.shape[0] <= self.L, "Input sequence length exceeds the maximum length L"
        x_k = torch.zeros(self.L, self.N, dtype=torch.complex64, device=x.device)
        y_k = torch.zeros(self.L, 1, dtype=torch.complex64, device=x.device)
        y_k[:u.shape[0], 0] = u
        D = torch.diag(2/self.step_size - (self.Lambda_real + 1j * self.Lambda_imag))
        P = torch.complex(self.P_real, self.P_imag).unsqueeze(1)
        B = torch.complex(self.B_real, self.B_imag)
        C = 0 # DEFINE THIS
        Q_star = torch.conj(P).T
        A0 = torch.diag(torch.ones(self.N)*2/self.step_size) + torch.diag(self.Lambda_real + 1j * self.Lambda_imag) - P @ Q_star
        A1 = D - D@P * 1/(1+Q_star@D@P) @ Q_star
        A_bar = A1@A0
        B_bar = 2 * A1 @ B

        for i in range(self.L-1):
            x_k[i+1] = A_bar @ x_k[i] + y_k[i] * B_bar
            if i > self.N-1:
                y_k[i+1] = C @ x_k[i+1]
        breakpoint()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Lambda = torch.complex(self.Lambda_real, self.Lambda_imag)
        P = torch.complex(self.P_real, self.P_imag)
        B = torch.complex(self.B_real, self.B_imag)
        C_tilde = torch.complex(self.C_real, self.C_imag)

        with torch.no_grad():
            fourier_kernel = self.kernel(Lambda, P, P, B, C_tilde, self.step_size)
        fft_u = torch.fft.fft(x, self.L)
        y = torch.fft.ifft(fourier_kernel * fft_u, self.L)
        return y.real

@LayerRegistry.register("s4dss")
class S4DSSLayer(nn.Module):
    def __init__(self, N: int, L: int):
        super(S4DSSLayer, self).__init__()
        self.L = L

        Lambda, P, B, _ = make_DPLR_HiPPO(N)
        self.Lambda_real = nn.Parameter(Lambda.real)
        self.Lambda_imag = nn.Parameter(Lambda.imag)
        self.w_real = nn.Parameter(torch.randn(N))
        self.w_imag = nn.Parameter(torch.randn(N))
        self.step_size = nn.Parameter(torch.tensor(1e-2))
        self.kernel = s4dss_kernel(L)

    def forward(self, x: torch.Tensor):
        Lambda = torch.complex(self.Lambda_real, self.Lambda_imag)
        w = torch.complex(self.w_real, self.w_imag)

        with torch.no_grad():
            kernel = self.kernel(Lambda, w, self.step_size)

        fft_kernel = torch.fft.rfft(kernel, self.L)
        fft_x = torch.fft.rfft(x, self.L)

        return torch.fft.irfft(fft_kernel * fft_x)

@LayerRegistry.register("conv")
class ConvLayer(nn.Module):
    """
    ConvLayer: A single layer of implementing a convolutional layer instead of S4 layer.
    
    Args:
        N: int: Kernel length
        L: int: Sequence length
    """

    def __init__(self, N: int, L: int):
        super(ConvLayer, self).__init__()
        self.N = N
        self.L = L
        self.conv = nn.Conv1d(1, 1, N, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.unsqueeze(1)).squeeze(1)

 
class S4sequence(nn.Module):
    """
    S4sequence: A sequence block encasing an S4 layer for each feature with normalization, 
    position-wise linear layers and dropout.

    Args:
        layer_cls: str: The layer class to use
        N: int: Size of hidden state space
        H: int: Number of features
        L: int: Sequence length
    """
    def __init__(self,layer_cls: str, N: int, H: int, L:int, glu:bool=False):
        super(S4sequence, self).__init__()
        self.use_glu = glu

        self.norm = nn.LayerNorm((L,H), elementwise_affine = False, bias = False)
        self.s4layers = nn.ModuleList([LayerRegistry.create(layer_cls, N, L) for _ in range(H)])
        self.activation = nn.GELU() # Perhaps replace with ReLU based on "ReLU strikes back" paper (on LLM tasks)
        self.out1 = nn.Linear(H, H)
        if self.use_glu:
            self.out2 = nn.Linear(H, H)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm(x)

        # Loop through the S4 layers for each of the H features
        outputs = [s4.forward(x[..., idx]) for idx, s4 in enumerate(self.s4layers)]

        x = torch.stack(outputs, dim=-1)
        x = self.activation(x)
        x = self.dropout(x)

        if self.use_glu:
            x = self.out1(x)*nn.Sigmoid()(self.out2(x)) # a gated linear unit
        else:
            x = self.out1(x)
        return x + skip

class S4Model(lightning.LightningModule):
    """
    S4Model: The S4 model consisting of an encoder, a stack of S4 sequences and a classifier.
    
    Args:
        layer_cls: str: The layer class to use, options: "s4" or "conv"
        N: int: Size of hidden state space
        H: int: Number of features
        L: int: Sequence length
        num_blocks: int: Number of S4 sequence blocks
        cls_out: int: Number of classes
    """
    def __init__(self, layer_cls: str,  N: int, H:int, L:int, num_blocks:int, cls_out:int):
        super(S4Model, self).__init__()
        
        self.enc = nn.Linear(1,H)
        self.blocks = nn.ModuleList([S4sequence(layer_cls, N, H, L) for _ in range(num_blocks)])
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
        acc = (y == y_hat.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.log('train_acc', acc, on_epoch=True, on_step=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y == y_hat.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss, on_epoch=True, on_step=True)
        self.log('val_acc', acc, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y == y_hat.argmax(dim=-1)).float().mean()
        self.log('test_loss', loss, on_epoch=True, on_step=True)
        self.log('test_acc', acc, on_epoch=True, on_step=True)
        return loss
    

if __name__=="__main__":
    N = 64
    H = 128
    L = 784
    num_blocks = 4
    class_out = 10

    x = torch.randn(1, 784)
    
    # model  = S4Model(layer_cls="s4", N=N, H=H, L=L, num_blocks=num_blocks, cls_out=class_out)
    model = S4Layer(N, L)
    y = model.forward_recurrence(x)
    print(y.shape)


