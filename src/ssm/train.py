from ssm.model import S4Model
from ssm.data import SMNIST
import lightning
from torch.utils.data import DataLoader

def train(N=64, H=128, L=784, num_blocks=4, class_out=10):
    model = S4Model(N=N, H=H, L=L, num_blocks=num_blocks, cls_out=class_out)
    
    dataset_train = SMNIST()
    dataset_val = SMNIST(train = False)
    trainer = lightning.Trainer(accelerator="cpu", max_epochs=1)
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
