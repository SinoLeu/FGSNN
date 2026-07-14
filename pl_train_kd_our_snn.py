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
from utils.pl_data_loader import PlCAMDataModule,PlDataModule
# from models.s_model import get_sewresnet
from collections import OrderedDict
import timm
import logging
# from utils.utils import soft_kd_loss,mmd_loss,getForwardCAM,compute_kl_divergence,freeze_model_parameters
from utils.utils import mmd_loss,getForwardCAM,compute_kl_divergence,freeze_model_parameters,soft_loss_smooth,logits_external_loss,logits_internal_loss
# from spikingjelly.clock_driven.neuron import LIFNode
from spikingjelly.activation_based import neuron
from models.mpsa import MultiStageFeatureModule
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

logging.getLogger().setLevel(logging.ERROR)
import os
os.environ['CURL_CA_BUNDLE'] = ''

def create_resformer(args):
    from models.spikingresformer import spikingresformer_ti, spikingresformer_s, spikingresformer_m, spikingresformer_l
    arch_dict = {
        'ti':spikingresformer_ti, 's':spikingresformer_s, 'm':spikingresformer_m, 'l':spikingresformer_l
    }
    model = arch_dict[args.stu_arch](T=args.T)
    state_dict = torch.load(f'snn_checkpoint/spikingresformer_{args.stu_arch}.pth', map_location=torch.device('cpu'))
    state_dict = remove_module_from_state_dict(state_dict['model'])
    model.load_state_dict(state_dict,strict=False)
    
    model.classifier = nn.Linear(
        model.classifier.in_features, args.num_classes
    ) 
    return model

def create_qk_former(args):
    from models.qkformer import QKFormer_10_384,QKFormer_10_512,QKFormer_10_768
    arch_dict = {
        'ti':QKFormer_10_384,'s':QKFormer_10_512,'m':QKFormer_10_768
    }
    name_dict = {
        'ti':'10-384-224', 's':'10-512-224', 'm':'10-768-224'
    }
    model = arch_dict[args.stu_arch](T=args.T)
    state_dict = torch.load(f'snn_checkpoint/HST-{name_dict[args.stu_arch]}.pth', map_location=torch.device('cpu'))
    state_dict = remove_module_from_state_dict(state_dict['model'])
    model.load_state_dict(state_dict,strict=False)
    model.head = nn.Linear(
        model.head.in_features, args.num_classes
    )  # set fc layer of model with exact class number of current dataset
    
    return model

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



class CombineNet(nn.Module):
    """Combined network with ResNet backbone and MultiStageFeatureModule."""
    def __init__(self, model, stage_module):
        super().__init__()
        self.model = model
        self.da_module = stage_module

    def forward(self, x, spike_features):
        # [stage1,stage2,stage3,stage4] = base_model.forward_intermediates
        _, feat = self.model.forward_intermediates(x)
        y1 = self.model(x)
        # print(feat[1].shape,feat[2].shape,feat[3].shape)
        formatted_feat = feat[1:]
        out_tea = self.da_module(formatted_feat)
        out_stu = self.da_module(spike_features)
        return y1,out_tea,out_stu
        # return y1


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
    backbone = timm.create_model(args.tea_arch, pretrained=True, num_classes=args.num_classes, verify=False)
    args_dim_dict = {
        'swin_small_patch4_window7_224':[192,384,768],
        'swin_base_patch4_window7_224':[256,512,1024],
    }
    channel_dim_list = args_dim_dict[args.tea_arch]
    da_module = MultiStageFeatureModule(nb_class=args.num_classes,channel_dim_list=channel_dim_list)
    net = CombineNet(model=backbone, stage_module=da_module)
    state_dict = torch.load(args.multistage_pth_path, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict['state_dict'])
    freeze_model_parameters(net)
    return net

class LitModel(pl.LightningModule):
    def __init__(self,args=None):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = args.learning_rate
        self.num_classes = args.num_classes
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) 
        self.noise_weight = args.noise_weight
        self.top_k = args.top_k
        self.kd_weight = args.kd_weight
        ##
        # self.feature_extractor = create_qk_former(args)
        if args.name == 'resformer':
            self.feature_extractor = create_resformer(args)
        elif args.name == 'qkformer':
            self.feature_extractor = create_qk_former(args)
        self.teacher = load_combinenet_model(args)
        self.crop_size = args.input_size
        self.hyper_cam = args.hyper_cam
        
        # self.kd_weight = args.kd_weight
        # for qkformer 
        # dim_dict = {
        #     'ti':[96,192,384],
        #     's':[128,256,512],
        #     'm':[192,384,768],
        # }
        # for resformer 
        dim_dict = {
            'l':[128, 512, 1024],
            'm':[64, 384, 768],
            # [64, 384, 768]
        }
        teacher_dim = {
            'swin_small_patch4_window7_224':[192,384,768],
            'swin_base_patch4_window7_224':[256,512,1024],
        }
        stu_dim_list = dim_dict[args.stu_arch]
        tea_dim_list = teacher_dim[args.tea_arch]
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=stu_dim_list[0],
                out_channels=tea_dim_list[0],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.Dropout(0.2),
            nn.BatchNorm2d(tea_dim_list[0]),
            neuron.LIFNode()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=stu_dim_list[1],
                out_channels=tea_dim_list[1],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.Dropout(0.2),
            nn.BatchNorm2d(tea_dim_list[1]),
            neuron.LIFNode()             
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=stu_dim_list[2],
                out_channels=tea_dim_list[2],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(tea_dim_list[2]),
            neuron.LIFNode()
        )
    # will be used during inference
    def forward(self, x,return_inter=True):
        x = self.feature_extractor(x,return_inter=return_inter)
        return x
    
    def training_step(self, batchs):
        batch = batchs['image']
        gt = batchs['label']
        # batch, gt = batch[0], batch[1]
        grad_cam = batchs['gradcam']
        out,mid_out_s = self.forward(batch,return_inter=True)
        spike_activate_map = getForwardCAM(mid_out_s[-1]).unsqueeze(1)
        mid_out_s = [mid.mean(dim=0) for mid in mid_out_s]
        # print(mid_out_s[2].shape)
        proj_mid1 = self.conv1(mid_out_s[0])
        proj_mid2 = self.conv2(mid_out_s[1])
        proj_mid3 = self.conv3(mid_out_s[2])
        mid_out_s = [ proj_mid1,proj_mid2,proj_mid3  ]
        out_t, out_tea, out_stu  = self.teacher(batch,mid_out_s)
        # out_t = self.teacher(batch,mid_out_s)  
        
        layer3_out_fire_rate = F.interpolate(spike_activate_map, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False).squeeze(1)
        l3 = compute_kl_divergence(layer3_out_fire_rate,grad_cam,Tau = 2.0)
        
        # s_noise_logits = generate_gaussian_noise_weight(out) + logits_gram_loss(out,out_t,temperature=2.0)
        # self.hyper_cam*l3 
        # + 7e-4*mmd_loss(out_stu, out_tea)
        # + l3 
        #  logits_internal_loss(out,out_t,temperature=2.0) + logits_external_loss(out,out_t,temperature=2.0)
        loss = self.criterion(out, gt) + soft_loss_smooth(out,out_t,noise_weight=self.noise_weight) + mmd_loss(out_stu, out_tea) + l3 
        acc = calculate_accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)
        ## strand snn 
        functional.reset_net(self.feature_extractor)
        functional.reset_net(self.conv1)
        functional.reset_net(self.conv2)
        functional.reset_net(self.conv3)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch,return_inter=False)
        loss = self.criterion(out, gt)
        self.log("val/loss", loss)
        acc = calculate_accuracy(out, gt)
        self.log('val/acc', acc, prog_bar=True, on_epoch=True, sync_dist=True)
        ## strand snn 
        functional.reset_net(self.feature_extractor)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch,return_inter=False)
        loss = self.criterion(out, gt)
        ## strand snn 
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
        ## strand snn 
        functional.reset_net(self.feature_extractor)

        
    def configure_optimizers(self):
        # Warmup 参数
        # # 创建调度器
        optimizer = torch.optim.Adam(
            nn.ModuleList([self.feature_extractor,self.conv1,self.conv2,self.conv3]).parameters(), lr=self.learning_rate, 
            # self.feature_extractor.parameters(), lr=self.learning_rate, 
            #  lr=1e-4,
            weight_decay=1e-4
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


def parse_args():
    from config.config import parse_args_yml
    # args = parse_args_yml('config/aircraft/plt_our_kdtrain_snn_aircraft.yml')
    # args = parse_args_yml('config/cars196/plt_our_kdtrain_snn_cars.yml')
    args = parse_args_yml('config/cubs200/plt_our_kdtrain_snn_cub.yml')
    # args = parse_args_yml('config/dogs/plt_our_kdtrain_snn_dogs.yml')
    # args = parse_args_yml('config/nabrids/plt_our_kdtrain_snn_nabrids.yml')
    return args

# def main():
#     args = parse_args()
#     logger_name = args.name_space
#     logger = CSVLogger(args.checkpoint_dir, name=logger_name)

#     checkpoint_callback = ModelCheckpoint(
#         monitor="val/acc",           # 监控验证集准确率
#         mode="max",                  # 追踪最大值
#         save_top_k=1,                # 保存最佳模型
#         verbose=True,                # 输出日志
        
#         filename="best_model"        # 文件名
#     )

#     dm = PlCAMDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, crop_size=args.input_size,cam_path=args.cam_path)
#     # dm = PlDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, crop_size=args.input_size)
#     # set fc layer of model with exact class number of current dataset
#     model = LitModel(args=args)
    
#     if args.is_distributed:
#         trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp",precision=args.mixed)
#     else:
#         trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, devices=1, accelerator="gpu",callbacks=[checkpoint_callback], precision=args.mixed, gradient_clip_val=0)
#     trainer.fit(model, dm)
#     print("end....")

# if __name__ == "__main__":
#     main()


## python3 pl_train_kd_our_snn.py
## python3 