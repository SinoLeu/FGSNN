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
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR,LambdaLR
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.pl_data_loader import StanfordCarsDataModule
from models.s_model import get_sewresnet
from collections import OrderedDict
from models.mpsa import format_reverse,DetailAttentionModule
# from torchmetrics import Accuracy
# from pytorch_lightning.loggers import WandbLogger
# from torchvision import transforms
# from torch.cuda.amp import GradScaler, autocast
from save_cam import generate_gradcam_results,GradCamDataset
from utils.utils import freeze_model_parameters,getForwardCAM,compute_kl_divergence


def mmd_loss(source_logits, target_logits, sigma=1.0):
    
    # 确保输入维度一致
    assert source_logits.size(1) == target_logits.size(1), "源域和目标域的类别数必须相同"
    
    # 获取批量大小
    n_s = source_logits.size(0)  # 源域样本数
    n_t = target_logits.size(0)  # 目标域样本数
    
    # 计算源域内、目标域内和跨域的核矩阵
    # 使用 torch.cdist 计算欧氏距离矩阵
    dist_ss = torch.cdist(source_logits, source_logits, p=2) ** 2  # 源域样本间的距离平方
    dist_tt = torch.cdist(target_logits, target_logits, p=2) ** 2  # 目标域样本间的距离平方
    dist_st = torch.cdist(source_logits, target_logits, p=2) ** 2  # 源域与目标域样本间的距离平方
    
    # 计算高斯核
    kernel_ss = torch.exp(-dist_ss / (2 * sigma ** 2))
    kernel_tt = torch.exp(-dist_tt / (2 * sigma ** 2))
    kernel_st = torch.exp(-dist_st / (2 * sigma ** 2))
    
    # 计算 MMD 的三项
    term_ss = kernel_ss.sum() / (n_s * n_s)  # 源域内核均值
    term_tt = kernel_tt.sum() / (n_t * n_t)  # 目标域内核均值
    term_st = kernel_st.sum() / (n_s * n_t)  # 跨域核均值
    
    # 完整 MMD 损失
    mmd = term_ss + term_tt - 2 * term_st
    
    return mmd

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
class OurKDLitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.1,
                 img_size=256,dim=2048,
                 teacher_path="logs/fine_tune_resnet50/version_1/checkpoints/best_model.ckpt"
                 ):
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
        weights = torch.load(teacher_path, map_location=torch.device('cpu'))['state_dict']
        teacher_weight = {key:value for key,value in weights.items() if 'feature_extractor' in key}
        teacher_feature_extractor_weights = {
            key.replace("feature_extractor.", ""): value 
            for key, value in teacher_weight.items() 
            if key.startswith("feature_extractor.")
        }
        self.teacher .fc = nn.Linear(
                self.teacher .fc.in_features, num_classes
        )  # set fc layer of model with exact class number of current dataset
        self.teacher.load_state_dict(teacher_feature_extractor_weights)
        ## forzen teacher
        for param in self.teacher.parameters():
            param.requires_grad = False  # make parameters in model learnable
            
        ## load student checkpoint
        self.feature_extractor = get_sewresnet(arch='50',T=4,connect_f='ADD')
        ## (pretrained=False, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
        state_dict = torch.load('snn_checkpoint/sew_resnet-50.pth', map_location=torch.device('cpu'))
        state_dict = remove_module_from_state_dict(state_dict['model'])
        self.feature_extractor.load_state_dict(state_dict)
        self.feature_extractor.fc = nn.Linear(
            self.feature_extractor.fc.in_features, num_classes
        )  # set fc layer of model with exact class number of current dataset
        self.da_module = DetailAttentionModule(dim=dim,input_size=(img_size,img_size),nb_class=num_classes)
        da_module_weight = {key:value for key,value in weights.items() if 'da_module' in key}
        da_module_weight = {
            key.replace("da_module.", ""): value 
            for key, value in da_module_weight.items() 
            if key.startswith("da_module.")
        }
        self.da_module.load_state_dict(da_module_weight)
        ## forzen da_module
        for param in self.da_module.parameters():
            param.requires_grad = False  # make parameters in model learnable
            
        self.re_size = img_size
        self.dim = dim

    def _forward_teacher(self, x,spike_features):
        # 通过 ResNet-50 提取特征
        y1, feat = self.teacher(x, return_mid=True) 
        formatted_feat = format_reverse(feat[:4])
        # 通过 DetailAttentionModule 处理特征
        out_tea = self.da_module(formatted_feat)
        out_stu = self.da_module(format_reverse(spike_features))
        return y1,out_tea,out_stu


    # will be used during inference
    def forward(self, x,return_inter=True):
        x,mid = self.feature_extractor(x,return_inter=return_inter)
        return x,mid
    
    def training_step(self, batchs):
        # batch, gt = batch[0], batch[1]
        # grad_cam = data['gradcam']['cams'].to(device)
        batch = batchs['image']
        gt = batchs['label']
        grad_cam = batchs['gradcam']['cams']
        out = self.forward(batch,return_inter=True)
        spike_activate_map = getForwardCAM(mid_out_s[-1]).unsqueeze(1)
        ## return_mid
        mid_out_s = [mid.mean(dim=0) for mid in mid_out_s]
        
        out_t,out_tea,out_stu = self._forward_teacher(out,mid_out_s)
        layer3_out_fire_rate = F.interpolate(spike_activate_map, size=(self.re_size, self.re_size), mode='bilinear', align_corners=False).squeeze(1)
        l3 = 15*compute_kl_divergence(layer3_out_fire_rate,grad_cam,Tau = 1.0)
        loss = criterion(out, gt) + soft_kd_loss(out,out_t) + mmd_loss(out_stu,out_tea) + 2.4*l3
        
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
    logger_name = f"fine_tune_resnet{args.resnet_scale}"
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
    model = OurKDLitModel(num_classes=args.num_classes, 
                          transfer=args.is_transfer, learning_rate=learning_rate, 
                          resnet_scale=args.resnet_scale,
                          img_size=args.input_size,teacher_path=args.teacher_path)
    
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