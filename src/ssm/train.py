from ssm.model import S4
from ssm.data import SMNIST
import lightning as L
from torch.utils.data import DataLoader

def train():
    dataset_train = SMNIST()
    dataset_val = SMNIST(train = False)
    model = S4(state_size=256, input_length=784, num_layers=3, num_blocks=1, cls_out=10)
    trainer = L.Trainer(accelerator="gpu", max_epochs=1)
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
