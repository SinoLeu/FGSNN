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
from utils.pl_data_loader import PlDataModule
from utils.label_smoothing import LabelSmoothingLoss
from torch.cuda.amp import GradScaler, autocast
from models.mpsa import MultiStageFeatureModule
import timm
from utils.utils import freeze_model_parameters
import os
os.environ['CURL_CA_BUNDLE'] = ''
## python fine-tine-resnet_ann.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 150  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer
class CombineNet(nn.Module):
    """Combined network with ResNet backbone and MultiStageFeatureModule."""
    def __init__(self,arch, model, stage_module,dim=[256,512,1024],classes=120,output_size=(1, 1)):
        super().__init__()
        self.arch =  arch
        self.model = model
        self.da_module = stage_module
        # self.linear1 = nn.Linear(dim[0],classes)
        # self.linear2 = nn.Linear(dim[1],classes)
        # self.linear3 = nn.Linear(dim[2],classes)
        # self.pool = nn.AdaptiveAvgPool2d(output_size)
    def forward(self, x):
        
        # [stage1,stage2,stage3,stage4] = base_model.forward_intermediates
        _, feat = self.model.forward_intermediates(x)
        # y = self.model(x)
        # if self.arch == 'resnet50':
            
        # print(feat[1].shape,feat[2].shape,feat[3].shape)
        formatted_feat = feat[1:]
        out = self.da_module(formatted_feat)
        # y1 = self.linear1(self.pool(feat[1]).squeeze())
        # y2 = self.linear2(self.pool(feat[2]).squeeze())
        # y3 = self.linear3(self.pool(feat[3]).squeeze())
        return out


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

def load_combinenet_model(args):
    backbone = timm.create_model(args.arch, pretrained=True, num_classes=args.num_classes)
    state_dict = torch.load(args.pre_trained_path, map_location='cpu')
    
    backbone.load_state_dict(state_dict['state_dict'])
    freeze_model_parameters(backbone)
    args_dim_dict = {
        'resnet50':[256,512,1024],
        'swin_small_patch4_window7_224':[192,384,768],
        'swin_base_patch4_window7_224':[256,512,1024],
    }
    channel_dim_list = args_dim_dict[args.arch]
    da_module = MultiStageFeatureModule(nb_class=args.num_classes,channel_dim_list=channel_dim_list)
    net = CombineNet(arch=args.arch,model=backbone, stage_module=da_module)
    return net

## fine-tune resnet
class LitModel(pl.LightningModule):
    def __init__(self,args=None):
        super().__init__()
        ## learning_rate=1e-4,
        print(f"..... train {args.arch} ...")
        # self.feature_extractor = model
        self.save_hyperparameters()
        self.learning_rate = args.learning_rate
        self.num_classes = args.num_classes
        self.criterion = LabelSmoothingLoss(
            classes=args.num_classes, smoothing=args.smoothing
        )  # label smoothing to improve performance
        self.feature_extractor = load_combinenet_model(args)
        # self.feature_extractor = timm.create_model(arch, pretrained=True, num_classes=num_classes)
        
    # will be used during inference
    def forward(self, x):
        # out,y1,y2,y3 = self.feature_extractor(x)
        # return out,y1,y2,y3
        y = self.feature_extractor(x)
        return y
    
    def training_step(self, batch):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt) + self.feature_extractor.da_module.get_diversity_loss() 

        acc = calculate_accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        self.log("val/loss", loss)

        acc = calculate_accuracy(out, gt)
        self.log('val/acc', acc, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)
        
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
        optimizer = torch.optim.Adam(
            self.feature_extractor.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-3
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=3, 
            factor=0.5, 
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }

# # 设置 TensorBoardLogger
def parse_args():
    from config.config import parse_args_yml
    # args = parse_args_yml('config/aircraft/plt_train_multistage_module.yml')
    # args = parse_args_yml('config/cars196/plt_train_multistage_module.yml')
    # args = parse_args_yml('config/cubs200/plt_train_multistage_module.yml')
    args = parse_args_yml('config/dogs/plt_train_multistage_module.yml')
    # args = parse_args_yml('config/nabrids/plt_train_multistage_module.yml')
    # args = parse_args_yml('config/food_101/plt_train_multistage_module.yml')
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

    dm = PlDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, crop_size=args.input_size)
    # set fc layer of model with exact class number of current dataset
    model = LitModel(args=args)
    
    # Lightning
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