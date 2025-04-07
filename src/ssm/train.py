import os
from ssm.model import S4Model
from ssm.data import SMNIST, AudioMNIST, WeatherDataset
from ssm.utils import get_train_val_dataset
import lightning as pl
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.cuda as cuda

@hydra.main(config_name="config.yaml", config_path="../../configs")
def train(cfg: DictConfig):
    hp_config =cfg.experiment.hyperparameters
    model = S4Model(layer_cls=hp_config.layer_cls, N=hp_config.N, H=hp_config.H, L=hp_config.L, num_blocks=hp_config.num_blocks, cls_out=hp_config.class_out, lr=hp_config.lr, weight_decay=hp_config.weight_decay, dropout=hp_config.dropout, forecasting=True, num_features=12)

    dataset_train, dataset_val = get_train_val_dataset(cfg.dataset)
    train_dataloader = DataLoader(dataset_train, batch_size=hp_config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset_val, batch_size=hp_config.batch_size, shuffle=False, num_workers=4)

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    wandb_name = f"{cfg.experiment.name}-N{hp_config.N}-L784-H{hp_config.H}-NumBlocks{hp_config.num_blocks}"
    
    acc = "gpu" if cuda.is_available() else "cpu"

    if cfg.wandb:
        logger = pl.pytorch.loggers.WandbLogger(name=wandb_name, entity="franka-ppo", project="ssm", config=OmegaConf.to_container(cfg.experiment, resolve=True))
    else:
        os.makedirs("logs", exist_ok=True)
        logger = pl.pytorch.loggers.CSVLogger("logs", name=wandb_name)

    trainer = pl.Trainer(
        accelerator=acc, 
        max_epochs=hp_config.max_epochs, 
        logger=logger, 
        callbacks=[checkpoint_callback],
        log_every_n_steps=hp_config.log_every_n_steps)

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
