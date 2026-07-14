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
from spikingjelly.activation_based import neuron, functional, surrogate, layer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR,LambdaLR
from torchmetrics import Accuracy

from torchvision import transforms
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.pl_data_loader import StanfordCarsDataModule
from models.s_model import get_sewresnet
from collections import OrderedDict

def soft_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    # 对logits应用温度缩放
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature
    
    # 计算软化的概率分布
    student_soft = F.softmax(student_scaled, dim=1)
    teacher_soft = F.softmax(teacher_scaled, dim=1)
    
    # 计算log概率，用于KL散度
    student_log_soft = F.log_softmax(student_scaled, dim=1)
    
    # 计算KL散度
    # 注意：我们需要乘以temperature^2来调整损失的尺度
    kd_loss = F.kl_div(
        student_log_soft,
        teacher_soft,
        reduction=reduction
    ) * (temperature ** 2)
    
    return kd_loss


def remove_module_from_state_dict(state_dict):
    """
    遍历模型的 state_dict，将所有包含 'module' 的部分去掉。

    参数:
        state_dict (OrderedDict): 模型的 state_dict。

    返回:
        OrderedDict: 修改后的 state_dict。
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict

criterion = nn.CrossEntropyLoss()

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
class KDLitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.1,teacgher_path="logs/fine_tune_resnet50/version_1/checkpoints/best_model.ckpt"):
        super().__init__()
        
        # self.feature_extractor = model
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        ### load teacher checkpoint .....
        self.teacher = models.resnet50(
            pretrained=True
        )
        # checkpoint_path = "logs/fine_tune_resnet50/version_1/checkpoints/best_model.ckpt"
        weights = torch.load(teacgher_path, map_location=torch.device('cpu'))['state_dict']
        teacher_feature_extractor_weights = {
            key.replace("feature_extractor.", ""): value 
            for key, value in weights.items() 
            if key.startswith("feature_extractor.")
        }
        self.teacher .fc = nn.Linear(
                self.teacher .fc.in_features, num_classes
        )  # set fc layer of model with exact class number of current dataset
        self.teacher.load_state_dict(teacher_feature_extractor_weights)
        
        ## load student checkpoint
        self.feature_extractor = get_sewresnet(arch='50',T=4,connect_f='ADD')
        ## (pretrained=False, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
        state_dict = torch.load('snn_checkpoint/sew_resnet-50.pth', map_location=torch.device('cpu'))
        state_dict = remove_module_from_state_dict(state_dict['model'])
        self.feature_extractor.load_state_dict(state_dict)
        self.feature_extractor.fc = nn.Linear(
            self.feature_extractor.fc.in_features, num_classes
        )  # set fc layer of model with exact class number of current dataset

        # self.teacher = None
        
    # will be used during inference
    def forward(self, x):
        x = self.feature_extractor(x)
        return x
    
    def training_step(self, batch):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        out_t = self.teacher(out)
        loss = criterion(out, gt) + soft_kd_loss(out,out_t) 
        
        acc = calculate_accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)
        ## strand reset snn 
        functional.reset_net(self.feature_extractor)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = criterion(out, gt)

        self.log("val/loss", loss)

        acc = calculate_accuracy(out, gt)
        self.log("val/acc", acc)
        ## strand reset snn 
        functional.reset_net(self.feature_extractor)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = criterion(out, gt)
        ## strand reset snn 
        functional.reset_net(self.feature_extractor)
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
        ## strand reset snn 
        functional.reset_net(self.feature_extractor)
        
    def configure_optimizers(self):
        # # 创建调度器
        # optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.SGD(
            self.feature_extractor.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# # 设置 TensorBoardLogger
def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet model on Stanford Cars")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--input_size', type=int, default=224, help="Input size for images")
    parser.add_argument('--train_dir', type=str, default='./train', help="Directory for training data")
    parser.add_argument('--test_dir', type=str, default='./test', help="Directory for testing data")
    parser.add_argument('--resnet_scale', type=str, default='50', choices=['18', '34', '50', '101'], help="ResNet scale")
    parser.add_argument('--max_epochs', type=int, default=150, help="Number of epochs for training")
    parser.add_argument('--checkpoint_dir', type=str, default="logs", help="Directory to save checkpoints")
    parser.add_argument('--num_classes', type=int, default=196, help="classificer classes")
    parser.add_argument('--is_distributed', action='store_true', help="Enable distributed training")
    parser.add_argument('--is_transfer', action='store_true', help="Enable distributed training")
    parser.add_argument('--mixed', type=str, default="bf16", help="Enable distributed training")
    return parser.parse_args()


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
    model = KDLitModel(num_classes=args.num_classes, transfer=args.is_transfer, learning_rate=learning_rate, resnet_scale=args.resnet_scale)
    
    # from torch.utils.data.distributed import DistributedSampler
    # Lightning
    # pl.seed_everything(2022, workers=True)
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp",precision=args.mixed)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs,devices=1,accelerator="gpu",callbacks=[checkpoint_callback],precision=args.mixed)
    trainer.fit(model, dm)
    print("end....")

if __name__ == "__main__":
    main()

## python pl_train_ann.py --batch_size 64 --input_size 256 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 128  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer --mixed bf16