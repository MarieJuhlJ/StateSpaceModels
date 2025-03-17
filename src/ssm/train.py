from ssm.model import S4Model
from ssm.data import SMNIST
import lightning as pl
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

@hydra.main(config_name="config.yaml", config_path="../../configs")
def train(cfg: DictConfig):
    hp_config =cfg.experiment.hyperparameters
    model = S4Model(N=hp_config.N, H=hp_config.H, L=hp_config.L, num_blocks=hp_config.num_blocks, cls_out=hp_config.class_out)
    
    dataset_train = SMNIST()
    dataset_val = SMNIST(train = False)
    train_dataloader = DataLoader(dataset_train, batch_size=hp_config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset_val, batch_size=hp_config.batch_size, shuffle=False, num_workers=4)

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    wandb_name = f"{cfg.experiment.name}-N{hp_config.N}-L784-H{hp_config.H}-NumBlocks{hp_config.num_blocks}"
    
    trainer = pl.Trainer(
        accelerator="gpu", 
        max_epochs=hp_config.max_epochs, 
        logger=pl.pytorch.loggers.WandbLogger(name=wandb_name,project="ssm"), 
        callbacks=[checkpoint_callback],
        log_every_n_steps=hp_config.log_every_n_steps)

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
