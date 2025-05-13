from torch import nn
from ssm.hippo import make_DPLR_HiPPO
from ssm.kernel import s4_kernel, s4dss_kernel
from ssm.utils import LayerRegistry
import lightning
import torch
from einops import rearrange, repeat
import math
from ssm.hippo import make_DPLR_HiPPO
from ssm.ops.selective_scan_interface import selective_scan_fn

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
    def __init__(self,layer_cls: str, N: int, H: int, L:int, glu:bool=False, dropout: float=0.1):
        super(S4sequence, self).__init__()
        self.use_glu = glu

        self.norm = nn.LayerNorm((L,H), elementwise_affine = False, bias = False)
        if layer_cls != "s6":
            self.ssm = nn.ModuleList([LayerRegistry.create(layer_cls, N, L) for _ in range(H)])
        else:
            self.ssm = S6Layer(d_model=H, d_state=N)
        self.activation = nn.GELU() # Perhaps replace with ReLU based on "ReLU strikes back" paper (on LLM tasks)
        self.out1 = nn.Linear(H, H)
        if self.use_glu:
            self.out2 = nn.Linear(H, H)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm(x)

        if isinstance(self.ssm, nn.ModuleList):
            outputs = [s4.forward(x[..., idx]) for idx, s4 in enumerate(self.ssm)]
            x = torch.stack(outputs, dim=-1)
        else:
            x = self.ssm(rearrange(x, "b l d -> b d l"))

        x = self.activation(x)
        x = self.dropout(x)

        if self.use_glu:
            x = self.out1(x)*nn.activation()(self.out2(x)) # a gated linear unit
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
    def __init__(self, layer_cls: str,  N: int, H:int, L:int, num_blocks:int, cls_out:int, lr:float, weight_decay: float, dropout:float, forecasting:bool=False, num_features:int=1):
        super(S4Model, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.forecasting = forecasting
        self.num_features = num_features

        self.enc = nn.Linear(num_features,H)

        self.blocks = nn.ModuleList([S4sequence(layer_cls, N, H, L, dropout=dropout) for _ in range(num_blocks)])

        if self.forecasting:
            self.cls = nn.Linear(H, num_features)
            self.loss = nn.MSELoss()
        else:
            self.cls = nn.Linear(H, cls_out)
            self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.num_features:
            x = x.unsqueeze(-1)
        x = self.enc(x)
        for block in self.blocks:
            x = block(x)

        if not self.forecasting:
            x = torch.mean(x, dim=1)
        x = self.cls(x)
        if not self.forecasting:
            x = nn.Softmax(dim=-1)(x)
        return x
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        if not self.forecasting:
            acc = (y == y_hat.argmax(dim=-1)).float().mean()
            self.log('train_acc', acc, on_epoch=True, on_step=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=True)
        if not self.forecasting:
            acc = (y == y_hat.argmax(dim=-1)).float().mean()
            self.log('val_acc', acc, on_epoch=True, on_step=True)
        print(f"val_loss: {loss}, val_acc: {acc}")
        return loss

    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_epoch=True, on_step=True)
        if not self.forecasting:
            acc = (y == y_hat.argmax(dim=-1)).float().mean()
            self.log('test_acc', acc, on_epoch=True, on_step=True)
        
        return loss

@LayerRegistry.register("s6")
class S6Layer(nn.Module):
    """
    S6Layer: A single layer of the S6 model implementing a kernel based on a DPLR HiPPO matrix.
    
    Args:
        N: int: Size of hidden state space
        L: int: Sequence
    """
    def __init__(
            self,
        d_model=None,
        d_state=16,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-3,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(S6Layer, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.x_proj = nn.Linear(
            self.d_model, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True, **factory_kwargs)
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_model, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_model,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)

        self.D = nn.Parameter(torch.ones(self.d_model, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Batch, _, seqlen = x.shape #(b d l)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
        y = rearrange(y, "b d l -> b l d")
        return y

if __name__=="__main__":
    N = 64
    H = 128
    L = 784
    num_blocks = 4
    class_out = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 784, H) #B L D