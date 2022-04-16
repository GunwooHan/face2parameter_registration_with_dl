from inspect import Parameter
import os

import kornia
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision

import pytorch_lightning as pl

class ConvBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, kernel_size=3, down_sample=False):
        super(ConvBlock, self).__init__()
        self.down_sample = down_sample

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=2 if down_sample else 1, padding=0 if kernel_size == 1 else 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, down_sample=False, first=False):
        super(ResBlock, self).__init__()
        self.down_sample = down_sample
        self.first = first

        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=1, down_sample=down_sample)
        self.conv_block2 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3)
        self.conv_block3 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=1)

        if self.down_sample:
            self.shortcut = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, down_sample=True)
        elif self.first:
            self.shortcut = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, input_tensor):
        x = self.conv_block1(input_tensor)
        x = torch.relu(x)
        x = self.conv_block2(x)
        x = torch.relu(x)
        x = self.conv_block3(x)

        if self.down_sample:
            input_tensor = self.shortcut(input_tensor)
        elif self.first:
            input_tensor = self.shortcut(input_tensor)

        x = x + input_tensor
        x = torch.relu(x)
        return x


class Registration(nn.Module):
    def __init__(self):
        super(Registration, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64),
            ResBlock(in_channels=64, out_channels=64),
            ResBlock(in_channels=64, out_channels=64)
        )
        self.block2 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=128, down_sample=True),
            ResBlock(in_channels=128, out_channels=128),
            ResBlock(in_channels=128, out_channels=128),
            ResBlock(in_channels=128, out_channels=128),
        )
        self.block3 = nn.Sequential(
            ResBlock(in_channels=128, out_channels=256, down_sample=True),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
        )
        self.block4 = nn.Sequential(
            ResBlock(in_channels=256, out_channels=512, down_sample=True),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 9)

    def forward(self, target, source):
        merged = torch.cat([target, source], dim=1)
        x = self.conv1(merged)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        x = torch.reshape(x, (-1, 3, 3))
        return x

class Reg(pl.LightningModule):
    def __init__(self, args):
        super(Reg, self).__init__()
        self.args = args
        self.recon_loss = nn.L1Loss()
        self.model = Registration()

    # def forward(self, tensor):
    #     return self.model(tensor)

    def configure_optimizers(self):
        opt_g = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        scheduler_g = lr_scheduler.StepLR(opt_g, step_size=50, gamma=0.9, verbose=True)

        return {"optimizer": opt_g, "lr_scheduler": {"scheduler": scheduler_g, "interval": "epoch"}},\

    def training_step(self, train_batch, batch_idx):
        target_image, source_image = train_batch
        warp_parameter = self.model(target_image, source_image)
        result_image = kornia.geometry.transform.warp_perspective(source_image, warp_parameter, target_image.shape[2:])
        recon_loss = self.recon_loss(target_image, result_image)
        total_loss = recon_loss * self.args.recon_weight
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True)
        return {"loss": total_loss}

    def validation_step(self, val_batch, batch_idx):
        target_image, source_image = val_batch
        warp_parameter = self.model(target_image, source_image)
        result_image = kornia.geometry.transform.warp_perspective(source_image, warp_parameter, target_image.shape[2:])
        recon_loss = self.recon_loss(target_image, result_image)

        if batch_idx == 0:
            sample_count = 4 if target_image.size(0) > 4 else target_image.size(0)
            target_image = target_image[:sample_count]
            source_image = source_image[:sample_count]

            warp_parameter = self.model(target_image, source_image)
            result_image = kornia.geometry.transform.warp_perspective(source_image, warp_parameter, target_image.shape[2:])
            result_grid = torchvision.utils.make_grid(torch.cat([target_image, source_image, result_image]), nrow=sample_count).permute(1,2,0).cpu().numpy()
            result_grid = cv2.cvtColor(result_grid, cv2.COLOR_BGR2RGB)
            self.logger.log_image(key='sample_images', images=[result_grid], caption=[self.current_epoch + 1])

            if self.current_epoch % 1 == 0:
                torch.save(self.model, os.path.join('checkpoints', self.args.name, f'{self.current_epoch:03d}_model.pt'))

        total_loss = recon_loss * self.args.recon_weight

        self.log('val/recon_loss', recon_loss, on_step=True, on_epoch=True)
        return {"loss": total_loss}

            
if __name__ == '__main__':
    


    tensor = torch.randn(1, 17)
