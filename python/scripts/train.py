import pathlib

import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets import MPIIFaceGazeDataset
from webeyetrack.models import EFEModel

FILE_DIR = pathlib.Path(__file__).parent

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':

    # Create the model
    model = EFEModel()

    # Create a dataset object
    dataset = MPIIFaceGazeDataset(GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']))

    # Create the dataloader
    train_size = int(config['train']['train_size'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    # Configure EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )

    # Create TensorBoard Logger
    model_name = 'EFE'
    tb_logger = pl.loggers.TensorBoardLogger('lightning_logs/', name=model_name)

    # Create a trainer
    trainer = Trainer(
        max_epochs=config['train']['max_epochs'],
        logger=tb_logger,
        callbacks=[early_stop_callback]
    )
    # trainer.fit(model, train_loader, val_loader)