import pathlib
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets import MPIIFaceGazeDataset

# Models
from webeyetrack.models.efe import EFEModel_PL
from webeyetrack.models.gaze360 import Gaze360_PL
from webeyetrack.models.eyenet import EyeNet_PL

FILE_DIR = pathlib.Path(__file__).parent

name_to_model = {
    'EFE': EFEModel_PL,
    'Gaze360': Gaze360_PL,
    'EyeNet': EyeNet_PL,
    # 'FaceNet': FaceNet_PL,
    # 'FaceEyeNet': FaceEyeNet_PL,
}

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':

    # Create arguments to select the model (EFE, Gaze360, etc.)
    parser = argparse.ArgumentParser(description='Train a model')

    # Restrict the options of the model
    parser.add_argument('--model', type=str, choices=['EFE', 'Gaze360', 'EyeNet'], required=True, help='The model to train')
    parser.add_argument('--exp', type=str, required=True, help='The experiment name')
    args = parser.parse_args()

    # Updating config according to the experiment
    config['exp'] = args.exp
    
    # Obtain model-specific dataset configuration
    model_config = config['train']['model_specific'][args.model]

    # Create the model
    model = name_to_model[args.model]() 

    # Create a dataset object
    train_dataset = MPIIFaceGazeDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
        **model_config['dataset_params'],
        participants=config['datasets']['MPIIFaceGaze']['train_subjects'],
    )
    val_dataset = MPIIFaceGazeDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
        **model_config['dataset_params'],
        participants=config['datasets']['MPIIFaceGaze']['val_subjects'],
    )

    # Create the dataloader
    # train_size = int(config['train']['train_size'] * len(dataset))
    # val_size = len(dataset) - train_size

    # Debugging, reducing the size of the dataset
    # train_size = int(len(dataset) * 0.8)
    # val_size = int(len(dataset) * 0.2)
    # train_size = config['train']['train_size'] * 10
    # val_size = config['train']['batch_size'] * 2

    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    # val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)

    # Configure EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor='val_angular_error_epoch',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )

    # Configure ModelCheckpoint to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_angular_error_epoch',
        dirpath='checkpoints/',
        filename=f'{args.model}-best-checkpoint',
        save_top_k=1,
        mode='min',
    )

    # Create TensorBoard Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        'lightning_logs/', 
        name=args.model, 
        version=args.exp
    )

    # Log the hyperparameters
    tb_logger.log_hyperparams(config['train'])

    # Create a trainer
    trainer = Trainer(
        max_epochs=config['train']['max_epochs'],
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)

    # Load the best model checkpoint after training
    best_model_path = checkpoint_callback.best_model_path
    best_model = name_to_model[args.model].load_from_checkpoint(best_model_path)

    # Evaluate the model on the validation set
    trainer.validate(best_model, val_loader)