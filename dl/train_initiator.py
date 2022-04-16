import os
import glob
import argparse

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split


from models import Reg
from datasets import P2FRegDataset
from utils import Normalize, DeNormalize

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_datadir', type=str, default='data/images')

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--name', type=str, default='v1')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--adam_beta', type=float, default=0)

parser.add_argument('--recon_weight', type=float, default=1)

args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    wandb_logger = WandbLogger(project=f'Face2Parameter_image_registration', name=f'{args.name}')
    wandb_logger.log_hyperparams(args)


    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )


    train_images = np.array(glob.glob(os.path.join(args.train_datadir, '*')))
    train_df, valid_df = train_test_split(train_images, test_size=0.2)


    train_ds = P2FRegDataset(train_df, target_transform=transform, source_transform=transform)
    valid_ds = P2FRegDataset(valid_df, target_transform=transform, source_transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   shuffle=True,
                                                   drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_ds,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                #    shuffle=True,
                                                   drop_last=True)

    model = Reg(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('checkpoints', args.name),
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor='train/recon_loss',
        mode='min',
        save_on_train_epoch_end=True
    )

    os.makedirs(os.path.join('checkpoints', args.name), exist_ok=True)
    if os.path.exists(os.path.join('checkpoints', args.name, 'last.ckpt')):
        trainer = pl.Trainer(gpus=args.gpus,
                             precision=args.precision,
                             max_epochs=args.epochs,
                            #  strategy='ddp',
                            #  limit_train_batches=1,
                            #  log_every_n_steps=1,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback],
                             resume_from_checkpoint=os.path.join('checkpoints', args.name, 'last.ckpt')
                             )
    else:
        trainer = pl.Trainer(gpus=args.gpus,
                             precision=args.precision,
                             max_epochs=args.epochs,
                            #  strategy='ddp',
                            #  limit_train_batches=1,
                            #  log_every_n_steps=1,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback]
                             )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    wandb.finish()