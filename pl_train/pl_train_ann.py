import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
from torchvision import models, transforms
import torch
# from timm import create_model
from torchvision import transforms, datasets
import lightning as L
import timm
# import os
from torch.optim.lr_scheduler import StepLR

import pytorch_lightning as pl
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR,LambdaLR
from torchmetrics import Accuracy

from torchvision import transforms
# from torchvision.datasets import StanfordCars
# from torchvision.datasets.utils import download_url
import torchvision.models as models
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.pl_data_loader import StanfordCarsDataModule
from utils.label_smoothing import LabelSmoothingLoss
from torch.cuda.amp import GradScaler, autocast

## python fine-tine-resnet_ann.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 150  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer
criterion =LabelSmoothingLoss(
            classes=196, smoothing=0.1
        )  # label smoothing to improve performance
def calculate_accuracy(outputs, targets):
    """
    计算给定输出和目标的准确率。
    
    Args:
        outputs (torch.Tensor): 模型的输出，通常是 logits 或概率，形状为 (batch_size, num_classes)
        targets (torch.Tensor): 真实标签，形状为 (batch_size,)
    
    Returns:
        tuple: (correct_count, total_count)
            - correct_count (int): 正确预测的数量
            - total_count (int): 样本总数
    """
    _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
    total = targets.size(0)                    # 样本总数
    correct = predicted.eq(targets.data).cpu().sum().item()  # 正确预测的数量
    
    return correct / total
## fine-tune resnet
class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.1):
        super().__init__()
        
        # self.feature_extractor = model
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.feature_extractor = models.resnet50(
            pretrained=True
        ) 
        self.feature_extractor.fc = nn.Linear(
                self.feature_extractor.fc.in_features, num_classes
        )  # set fc layer of model with exact class number of current dataset
        
    # will be used during inference
    def forward(self, x):
        x = self.feature_extractor(x)
        return x
    
    def training_step(self, batch):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = criterion(out, gt)

        acc = calculate_accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = criterion(out, gt)

        self.log("val/loss", loss)

        acc = calculate_accuracy(out, gt)
        self.log("val/acc", acc)

        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = criterion(out, gt)
        
        return {"loss": loss, "outputs": out, "gt": gt}
    
    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        output = torch.cat([x['outputs'] for x in outputs], dim=0)
        
        gts = torch.cat([x['gt'] for x in outputs], dim=0)
        
        self.log("test/loss", loss)
        acc = calculate_accuracy(output, gts)
        self.log("test/acc", acc)
        
        self.test_gts = gts
        self.test_output = output
    
    def configure_optimizers(self):
        # Warmup 参数
        # # 创建调度器
        # scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.SGD(
            self.feature_extractor.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=128, eta_min=0)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# # 设置 TensorBoardLogger
# def parse_args():
#     parser = argparse.ArgumentParser(description="Train a ResNet model on Stanford Cars")
#     parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
#     parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate")
#     parser.add_argument('--input_size', type=int, default=224, help="Input size for images")
#     parser.add_argument('--train_dir', type=str, default='./train', help="Directory for training data")
#     parser.add_argument('--test_dir', type=str, default='./test', help="Directory for testing data")
#     parser.add_argument('--resnet_scale', type=str, default='50', choices=['18', '34', '50', '101'], help="ResNet scale")
#     parser.add_argument('--max_epochs', type=int, default=150, help="Number of epochs for training")
#     parser.add_argument('--checkpoint_dir', type=str, default="logs", help="Directory to save checkpoints")
#     parser.add_argument('--num_classes', type=int, default=196, help="classificer classes")
#     parser.add_argument('--is_distributed', action='store_true', help="Enable distributed training")
#     parser.add_argument('--is_transfer', action='store_true', help="Enable distributed training")
#     parser.add_argument('--mixed', type=str, default="bf16", help="Enable distributed training")
#     return parser.parse_args()

def parse_args():
    from config.config import parse_args_yml
    args = parse_args_yml('config/cars196/plt_train_ann.yml')
    return args

def main():
    args = parse_args()
    logger_name = args.name_space
    logger = CSVLogger(args.checkpoint_dir, name=logger_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",           # 监控验证集准确率
        mode="max",                  # 追踪最大值
        save_top_k=1,                # 保存最佳模型
        verbose=True,                # 输出日志
        filename="best_model"        # 文件名
    )

    dm = StanfordCarsDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, crop_size=args.input_size)
    lr_begin = (args.batch_size / 256) * 0.1
    learning_rate = lr_begin
    # set fc layer of model with exact class number of current dataset
    model = LitModel(num_classes=args.num_classes,
                     learning_rate=learning_rate)
    
    # Lightning
    # pl.seed_everything(2022, workers=True)
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp",precision=args.mixed)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs,
                             devices=1,  # 限制为单 GPU
                             accelerator="gpu",callbacks=[checkpoint_callback],
                             precision=args.mixed,gradient_clip_val=0
                             )
    trainer.fit(model, dm)
    print("end....")

if __name__ == "__main__":
    main()