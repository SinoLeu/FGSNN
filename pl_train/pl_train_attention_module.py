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
from models.mpsa import format_reverse,DetailAttentionModule


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
class AttentionLitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.1,
                 img_size=256,dim=2048,
                 checkpoint_path="logs/fine_tune_backbone/version_0/checkpoints/best_model.ckpt"):
        super().__init__()
        
        # self.feature_extractor = model
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        self.feature_extractor = models.resnet50(
            pretrained=True
        )
        # checkpoint_path = "logs/fine_tune_resnet50/version_1/checkpoints/best_model.ckpt"
        weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
        feature_extractor_weights = {
            key.replace("feature_extractor.", ""): value 
            for key, value in weights.items() 
            if key.startswith("feature_extractor.")
        }
        self.feature_extractor.fc = nn.Linear(
                self.feature_extractor.fc.in_features, num_classes
        )  # set fc layer of model with exact class number of current dataset
        self.feature_extractor.load_state_dict(feature_extractor_weights)
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # make parameters in model learnable
        
        # da_module = DetailAttentionModule(dim=dim,input_size=(img_size,img_size),nb_class=num_classes)
        self.da_module = DetailAttentionModule(dim=dim,input_size=(img_size,img_size),nb_class=num_classes)
        
        self.img_size = img_size
        self.dim = dim

    # will be used during inference
    def forward(self, x):
        _, feat = self.feature_extractor(x, return_mid=True)
        
        formatted_feat = format_reverse(feat[:4]) 

        out = self.da_module(formatted_feat)
        return out
    
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
        # warmup_steps = 1000
        # lr_start = 0.0001
        # lr_target = 0.1
        optimizer = torch.optim.SGD(
            self.da_module.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# # 设置 TensorBoardLogger

def parse_args():
    from config.config import parse_args_yml
    args = parse_args_yml('config/cars196/pl_train_attention_module.yml')
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

    dm = StanfordCarsDataModule(batch_size=args.batch_size, 
                                train_dir=args.train_dir, 
                                test_dir=args.test_dir,
                                input_size=args.input_size)
    lr_begin = (args.batch_size / 256) * 0.1
    learning_rate = lr_begin
    # 加载并应用权重
    
    # print(checkpoint['state_dict'].keys())
    # set fc layer of model with exact class number of current dataset
    model = AttentionLitModel(num_classes=args.num_classes,
                     learning_rate=learning_rate,
                     img_size=args.input_size,checkpoint_path=args.checkpoint_path)
    
    # # Lightning
    # # pl.seed_everything(2022, workers=True)
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp",precision=args.mixed)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs,
                             devices=1,  # 限制为单 GPU
                             accelerator="gpu",callbacks=[checkpoint_callback],
                             precision=args.mixed,gradient_clip_val=0
                             )
    trainer.fit(model, dm)
    # print("end....")

if __name__ == "__main__":
    main()