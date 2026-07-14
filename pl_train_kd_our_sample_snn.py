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
from utils.utils import mmd_loss,getForwardCAM,compute_kl_divergence,freeze_model_parameters,soft_loss_smooth,logits_external_loss,logits_internal_loss,soft_loss_hard
# from spikingjelly.clock_driven.neuron import LIFNode
from spikingjelly.activation_based import neuron
from models.mpsa import MultiStageFeatureModule
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch
import torchvision.ops as ops

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

def create_resnet(args):
    from models.s_model import get_sewresnet
    model = get_sewresnet(args.stu_arch, T=args.T,connect_f='ADD')
    state_dict = torch.load(f'snn_checkpoint/sew_resnet-50.pth', map_location=torch.device('cpu'))
    state_dict = remove_module_from_state_dict(state_dict['model'])
    model.load_state_dict(state_dict,strict=False)
    model.fc = nn.Linear(
        model.fc.in_features, args.num_classes
    )  # set fc layer of model with exact class number of current dataset
    return model

def create_qk_former(args):
    from models.qkformer import QKFormer_10_384,QKFormer_10_512,QKFormer_10_768,QKFormer_10_768_384
    arch_dict = {
        'ti':QKFormer_10_384,'s':QKFormer_10_512,'m':QKFormer_10_768,'m-384':QKFormer_10_768_384
    }
    name_dict = {
        'ti':'10-384-224', 's':'10-512-224', 'm':'10-768-224','m-384':'10-768-384'
    }
    img_size_h = 224
    img_size_w = 224
    # if args.stu_arch == 'm-384':
        
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

    def forward(self, x):
        # [stage1,stage2,stage3,stage4] = base_model.forward_intermediates
        # _, feat = self.model.forward_intermediates(x)
        y1 = self.model(x)
        # print(feat[1].shape,feat[2].shape,feat[3].shape)
        # formatted_feat = feat[1:]
        # out_tea = self.da_module(formatted_feat)
        # out_stu = self.da_module(spike_features)
        # return y1,out_tea,out_stu
        return y1


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
    net = timm.create_model(args.tea_arch, pretrained=False, num_classes=args.num_classes)
    # args_dim_dict = {
    #     'swin_small_patch4_window7_224':[192,384,768],
    #     'swin_base_patch4_window7_224':[256,512,1024],
    # }
    # channel_dim_list = args_dim_dict[args.tea_arch]
    # da_module = MultiStageFeatureModule(nb_class=args.num_classes,channel_dim_list=channel_dim_list)
    # net = CombineNet(model=backbone, stage_module=da_module)
    state_dict = torch.load(args.pre_trained_tea_path, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict['state_dict'])
    freeze_model_parameters(net)
    return net

def sample_roi_from_prob(images, activate_map, labels, num_rois=5, roi_size=(32, 32)):
    """
    Args:
        images: Tensor of shape (b, c, h, w)
        activate_map: Tensor of shape (b, 1, h, w)
        labels: Tensor of shape (b,) or (b, num_classes)
        num_rois: number of ROIs to sample per image
        roi_size: size of each ROI (height, width)

    Returns:
        roi_images: Tensor of shape (b*num_rois, c, roi_h, roi_w)
        roi_labels: Tensor of shape (b*num_rois,) or (b*num_rois, num_classes)
    """
    b, _, h, w = activate_map.shape
    device = activate_map.device

    # Flatten and normalize activation map
    prob = activate_map.view(b, -1)
    prob = prob / (prob.sum(dim=1, keepdim=True) + 1e-8)

    # Sample pixel positions
    sampler = torch.distributions.Categorical(prob)
    sampled_indices = sampler.sample(sample_shape=(num_rois,)).T  # (b, num_rois)

    # Convert to coordinates
    ys = sampled_indices // w
    xs = sampled_indices % w

    # Create boxes [x1, y1, x2, y2]
    roi_size_half_h = roi_size[0] // 2
    roi_size_half_w = roi_size[1] // 2

    x1 = (xs - roi_size_half_w).clamp(0, w)
    y1 = (ys - roi_size_half_h).clamp(0, h)
    x2 = (xs + roi_size_half_w).clamp(0, w)
    y2 = (ys + roi_size_half_h).clamp(0, h)

    rois = torch.stack([x1, y1, x2, y2], dim=-1).float()  # (b, num_rois, 4)

    # Add batch indices for roi_align
    batch_indices = torch.arange(b, device=device).repeat_interleave(num_rois).view(-1, 1)
    rois = rois.view(-1, 4)
    rois = torch.cat([batch_indices, rois], dim=1)  # (b*num_rois, 5)

    # Extract ROI images
    roi_images = ops.roi_align(images, rois, output_size=roi_size)

    # Replicate labels
    if labels.dim() == 1:
        # For class indices: (b,) -> (b*num_rois,)
        roi_labels = labels.repeat_interleave(num_rois)
    elif labels.dim() == 2:
        # For one-hot or multi-label: (b, num_classes) -> (b*num_rois, num_classes)
        roi_labels = labels.unsqueeze(1).repeat(1, num_rois, 1).view(-1, labels.size(1))
    else:
        raise ValueError("Labels must be 1D or 2D")

    return roi_images, roi_labels


import torch
import torch.nn.functional as F
import torchvision.ops as ops


import torch
import torch.nn.functional as F


def sample_roi_from_prob_diversity_gumbel(images, activate_map, labels, num_rois=5, roi_size=(32, 32), crop_size=None, tau=0.5):
    b, c, h, w = images.shape
    device = images.device

    # Flatten activation map
    logits = activate_map.view(b, -1)  # shape: (b, h*w)
    logits = logits.unsqueeze(1).expand(b, num_rois, h*w).reshape(b*num_rois, h*w)  # (b*num_rois, h*w)

    # Apply Gumbel-Softmax to get sampling weights
    gumbel_weights = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)  # (b*num_rois, h*w)

    # Create normalized grid coordinates [0, 1]
    grid_x = torch.arange(w, device=device).float() / (w - 1)  # (w,)
    grid_y = torch.arange(h, device=device).float() / (h - 1)  # (h,)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid_x = grid_x.reshape(-1)  # (h*w,)
    grid_y = grid_y.reshape(-1)  # (h*w,)

    # Compute expected coordinates
    expected_x = (gumbel_weights * grid_x.unsqueeze(0)).sum(dim=1)  # (b*num_rois,)
    expected_y = (gumbel_weights * grid_y.unsqueeze(0)).sum(dim=1)  # (b*num_rois,)

    # Normalize to [-1, 1] for grid_sample
    grid_x_norm = 2.0 * expected_x - 1.0  # (b*num_rois,)
    grid_y_norm = 2.0 * expected_y - 1.0  # (b*num_rois,)

    # Create a grid for the ROI
    roi_h, roi_w = roi_size
    roi_grid_x = torch.linspace(-1.0, 1.0, roi_w, device=device)  # (roi_w,)
    roi_grid_y = torch.linspace(-1.0, 1.0, roi_h, device=device)  # (roi_h,)
    roi_grid_y, roi_grid_x = torch.meshgrid(roi_grid_y, roi_grid_x, indexing='ij')  # (roi_h, roi_w)

    # Scale the ROI grid to cover a region proportional to roi_size
    scale_x = roi_w / w  # Scale factor relative to input image width
    scale_y = roi_h / h  # Scale factor relative to input image height
    roi_grid_x = roi_grid_x * scale_x  # Scale the ROI grid
    roi_grid_y = roi_grid_y * scale_y

    # Center the ROI grid at the expected coordinates
    roi_grid_x = roi_grid_x.unsqueeze(0) + grid_x_norm.view(-1, 1, 1)  # (b*num_rois, roi_h, roi_w)
    roi_grid_y = roi_grid_y.unsqueeze(0) + grid_y_norm.view(-1, 1, 1)  # (b*num_rois, roi_h, roi_w)

    # Stack to create the sampling grid
    grid = torch.stack([roi_grid_x, roi_grid_y], dim=-1)  # (b*num_rois, roi_h, roi_w, 2)

    # Expand images for each ROI
    images_expanded = images.unsqueeze(1).expand(-1, num_rois, -1, -1, -1).reshape(b*num_rois, c, h, w)

    # Sample ROIs
    roi_images = F.grid_sample(
        images_expanded,
        grid,
        mode='bilinear',
        padding_mode='border',  # Use 'border' to avoid blank outputs at edges
        align_corners=True
    )  # (b*num_rois, c, roi_h, roi_w)

    # Replicate labels
    if labels.dim() == 1:
        roi_labels = labels.repeat_interleave(num_rois)  # (b*num_rois,)
    elif labels.dim() == 2:
        roi_labels = labels.unsqueeze(1).repeat(1, num_rois, 1).view(-1, labels.size(1))  # (b*num_rois, num_classes)
    else:
        raise ValueError("Labels must be 1D or 2D")
    ## 将所有roi images 上采样到 crop_size
    if crop_size is not None:
        roi_images = F.interpolate(roi_images, size=crop_size, mode='bilinear', align_corners=True)
    return roi_images, roi_labels

def roi_kl_distill_loss(student_logits, teacher_roi_logits, num_rois=15, T=2.0):
    """
    将 student_logits 扩展 num_rois 倍后与 teacher_roi_logits 对齐，
    并使用 KL 散度进行知识蒸馏。
    
    Args:
        student_logits (Tensor): 学生模型输出，shape=(B, C)
        teacher_roi_logits (Tensor): 教师模型对 ROI 的输出，shape=(B*num_rois, C)
        num_rois (int): 每张图像采样多少个 ROI（如 15）
        T (float): 温度参数，用于 soft softmax
        
    Returns:
        distill_loss (Tensor): 蒸馏损失值，标量
    """
    B, C = student_logits.size()
    
    # Step 1: 扩展 student_logits 到 (B * num_rois, C)
    student_expanded = student_logits.unsqueeze(1).repeat(1, num_rois, 1)  # (B, num_rois, C)
    student_expanded = student_expanded.view(B * num_rois, C)              # (B*num_rois, C)

    # # Step 2: 对 logits 应用 softmax + log_softmax
    # log_student = F.log_softmax(student_expanded / T, dim=1)
    # # with torch.no_grad():
    # log_teacher = F.log_softmax(teacher_roi_logits / T, dim=1)

    # # Step 3: 计算 KL 散度
    # distill_loss = F.kl_div(log_student, log_teacher, reduction='batchmean')
    # print(f'student_expanded shape: {student_expanded.shape}, teacher_roi_logits shape: {teacher_roi_logits.shape}')
    # Step 4: 乘上温度平方（Hinton 知识蒸馏论文推荐）
    # distill_loss = distill_loss * (T ** 2)
    # external_loss = logits_external_loss(student_expanded, teacher_roi_logits, temperature=T)
    # internal_loss = logits_internal_loss(student_expanded, teacher_roi_logits, temperature=T)
    # + external_loss + internal_loss
    return soft_loss_hard(student_expanded,teacher_roi_logits,temperature=T) 

def compute_cam(feature_maps, class_weights, predicted_classes):
    """
    Args:
        feature_maps: Tensor of shape (B, D, H, W)
        class_weights: Tensor of shape (C, D) -> 来自 linear 层权重
        predicted_classes: Tensor of shape (B,)
        size_upsample: tuple (H_up, W_up)

    Returns:
        cams: Tensor of shape (B, H_up, W_up)
    """
    
    feature_maps = feature_maps.permute(0, 3, 1, 2)  # (B, D, H, W)
    B, D, H, W = feature_maps.shape
    device = feature_maps.device

    # Step 1: 获取每个样本对应的类别的权重 (B, D)
    weights = class_weights[predicted_classes]  # shape: (B, D)
    # print(f'weights shape: {weights.shape}')  # (B, D)
    # Step 2: 加权求和：(B, D, H, W) * (B, D, 1, 1) -> (B, H, W)
    cams = (feature_maps * weights.view(B, D, 1, 1)).sum(dim=1)  # shape: (B, H, W)

    # Step 3: ReLU 激活
    cams = torch.relu(cams)

    # Step 4: 归一化到 [0, 1]
    cam_mins = cams.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
    cam_maxs = cams.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
    cams = (cams - cam_mins) / (cam_maxs - cam_mins + 1e-8)
    
    return cams

class LitModel(pl.LightningModule):
    def __init__(self,args=None):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = args.learning_rate
        self.num_classes = args.num_classes
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) 
        self.noise_weight = args.noise_weight
        self.opter = args.opter
        ##
        # self.feature_extractor = create_qk_former(args)
        if args.name == 'resformer':
            self.feature_extractor = create_resformer(args)
        elif args.name == 'qkformer':
            self.feature_extractor = create_qk_former(args)
        elif args.name == 'sewresnet':
            self.feature_extractor = create_resnet(args)
        self.teacher = load_combinenet_model(args)
        self.crop_size = args.input_size
        self.hyper_cam = args.hyper_cam
        self.max_epochs = args.max_epochs
        self.num_classes = args.num_classes
        self.Tau = args.Tau
        self.test_outputs = []
        self.is_distributed = args.is_distributed
        # self.prob_mask = args.prob_mask
    # will be used during inference
    def forward(self, x,return_inter=True):
        x = self.feature_extractor(x,return_inter=return_inter)
        return x
    
    def training_step(self, batchs):
        # batch, gt = batch[0], batch[1]
        batch = batchs['image']
        gt = batchs['label']
        # batch, gt = batch[0], batch[1]
        grad_cam = batchs['gradcam']
        out,mid_out_s = self.forward(batch,return_inter=True)
        mid_spike = mid_out_s[-1]
        # print(mid_spike.shape)

        spike_activate_map = getForwardCAM(mid_spike).unsqueeze(1)
        # print(spike_activate_map)
        
        # print(layer3_out_fire_rate)
        out_t = self.teacher(batch)
        # predicted_class = out_t.argmax(dim=1)
        # class_weights = self.teacher.head.fc.weight.data  # shape: (num_classes, D)  7e-4*bkd_loss 
        # features = self.teacher.forward_features(batch)
        # grad_cam = compute_cam(features, class_weights, predicted_class)
        
        # print(batch.shape,layer3_out_fire_rate.shape,gt.shape)
        # print(grad_cam.shape,spike_activate_map.shape).unsqueeze(1)
        # roi_images, roi_labels = sample_roi_from_prob(batch, layer3_out_fire_rate, gt, num_rois=10, roi_size=(self.crop_size, self.crop_size))
        # grad_cam = F.interpolate(grad_cam.unsqueeze(1), size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
        layer3_out_fire_rate = F.interpolate(spike_activate_map, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
        # roi_images, roi_labels = sample_roi_from_prob_diversity_gumbel(batch, layer3_out_fire_rate, gt, 
        #                                                         num_rois=8, roi_size=(64, 64),crop_size=self.crop_size)

        # out_t_sample = self.teacher(roi_images)
        # # # print(out_t_sample.shape)
        # sample_loss = self.criterion_ce(out_t_sample, roi_labels)
        
        l3 = compute_kl_divergence(layer3_out_fire_rate, grad_cam, Tau=self.Tau)  

        noisy_kd = soft_loss_smooth(student_logits=out,teacher_logits=out_t,noise_weight=self.noise_weight,temperature=self.Tau)
 
        loss = self.criterion(out, gt)  + l3*self.hyper_cam + noisy_kd 
        # + 1e-4*sample_loss  
        # + 1e-4*roi_kl_distill_loss(out, out_t_sample,num_rois=8,T=self.Tau)  
        acc = calculate_accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)
        functional.reset_net(self.feature_extractor)
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
        
        def calculate_fire_ratio(tensor):
            tensor = tensor.long()
            total_elements = tensor.numel()
            fire_count = (tensor >= 1).sum().item()
            fire_ratio = fire_count / total_elements
            return fire_ratio
        
        batch, gt = batch[0], batch[1]
        out,mid_features = self.forward(batch, return_inter=True)
        ratio_1 =  calculate_fire_ratio(mid_features[-1].squeeze())  # mid_features[-1] is the last feature map
        print(f'{ratio_1}%')
        loss = self.criterion(out, gt)
        functional.reset_net(self.feature_extractor)  # Reset SNN state
        # Store outputs in self.test_outputs
        self.test_outputs.append({"loss": loss, "outputs": out, "gt": gt})
        acc = calculate_accuracy(out, gt)
        self.log('test/acc', acc, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss  # Optional: return loss for default logging
    
        

    def on_test_epoch_end(self):
            # Aggregate outputs from self.test_outputs
            outputs = self.test_outputs
            
            if len(outputs) == 0:
                print("Warning: No outputs collected in test_step")
                return
            
            # Handle DDP: Gather outputs across ranks
            if self.is_distributed:
                import torch.distributed as dist
                gathered_outputs = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_outputs, outputs)
                outputs = [item for sublist in gathered_outputs for item in sublist]  # Flatten

            # Process outputs (only on rank 0 to avoid duplicate logging)
            if not self.is_distributed or dist.get_rank() == 0:
                loss = torch.stack([x['loss'] for x in outputs]).mean()
                output = torch.cat([x['outputs'] for x in outputs], dim=0)
                gts = torch.cat([x['gt'] for x in outputs], dim=0)
                
                self.log("test/loss", loss, sync_dist=True)  # Sync across ranks in DDP
                acc = calculate_accuracy(output, gts)  # Assume calculate_accuracy is defined
                self.log("test/acc", acc, sync_dist=True)  # Sync across ranks
                
                self.test_gts = gts
                self.test_output = output
            
            self.test_outputs.clear()  # Clear outputs for next epoch
        
    def configure_optimizers(self):
        # Warmup 参数
        # # 创建调度器
        # optimizer = torch.optim.Adam(
        #     # nn.ModuleList([self.feature_extractor,self.conv1,self.conv2,self.conv3]).parameters(), lr=self.learning_rate, 
        #     self.feature_extractor.parameters(), lr=self.learning_rate, 
        #     #  lr=1e-4,
        #     weight_decay=1e-4
        # )
        if self.opter == 'Adam':
            optimizer = torch.optim.Adam(
                # nn.ModuleList([self.feature_extractor,self.conv1,self.conv2,self.conv3]).parameters(), lr=self.learning_rate, 
                self.feature_extractor.parameters(), lr=self.learning_rate, 
                #  lr=1e-4,
                weight_decay=1e-4
            )
        elif self.opter == 'Nadam':
            optimizer = torch.optim.NAdam(
                # nn.ModuleList([self.feature_extractor,self.conv1,self.conv2,self.conv3]).parameters(), lr=self.learning_rate, 
                self.feature_extractor.parameters(), lr=self.learning_rate, 
                #  lr=1e-4,
                weight_decay=1e-4
            )
        elif self.opter == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.feature_extractor.parameters(), lr=self.learning_rate, 
                #  lr=1e-4,
                weight_decay=1e-4
            )
        elif self.opter == 'SGD':
            optimizer = torch.optim.SGD(
                # nn.ModuleList([self.feature_extractor,self.conv1,self.conv2,self.conv3]).parameters(), lr=self.learning_rate, 
                self.feature_extractor.parameters(), lr=self.learning_rate, 
                #  lr=1e-4,
                weight_decay=1e-4
            )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     patience=3, 
        #     factor=0.5, 
        #     verbose=True
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.max_epochs, 
            eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }


# def parse_args():
#     from config.config import parse_args_yml
#     # args = parse_args_yml('config/aircraft/plt_our_kdtrain_snn_aircraft.yml')
#     args = parse_args_yml('config/cars196/plt_our_kdtrain_snn_cars.yml')
#     # args = parse_args_yml('config/cubs200/plt_our_kdtrain_snn_cub.yml')
#     # args = parse_args_yml('config/cubs200/plt_our_kdtrain_snn_cub.yml')
#     # args = parse_args_yml('config/dogs/plt_our_kdtrain_snn_dogs.yml')
#     # args = parse_args_yml('config/nabrids/plt_our_kdtrain_snn_nabrids.yml')
#     return args
def parse_args():
    import argparse
    import yaml
    from config.config import parse_args_yml
    def print_yaml_content(file_path):
        with open(file_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
            print(yaml.dump(yaml_content, default_flow_style=False))
        
    # args = parse_args_yml('config/aircraft/plt_kdtrain_snn_aircraft.yml')
    # args = parse_args_yml('config/aircraft/plt_train_snn_aircraft.yml')
    # args = parse_args_yml('config/dogs/plt_erkdtrain_snn_dogs.yml')
    # args = parse_args_yml('config/cars196/plt_erkdtrain_snn_cars.yml')
    # args = parse_args_yml('config/cubs200/plt_kdtrain_snn_cub.yml')
    ##  nohup python pl_train_erkd_snn.py --config config/dogs/plt_erkdtrain_snn_dogs.yml &
    ##  nohup python pl_train_erkd_snn.py --config config/cars196/plt_erkdtrain_snn_cars.yml &
    ##  nohup python pl_train_erkd_snn.py --config config/cubs200/plt_erkdtrain_snn_cub.yml &
    ##  nohup python pl_train_erkd_snn.py --config config/aircraft/plt_erkdtrain_snn_aircraft.yml & 
    #  nohup python pl_train_kd_our_sample_snn.py --config config/cars196/plt_our_kdtrain_snn_cars.yml --print-yaml &
    # nohup python pl_train_kd_our_sample_snn.py --config plt_our_kdtrain_snn_cars_resformer.yml --print-yaml &
    # nohup python pl_train_kd_our_sample_snn.py --config config/cubs200/plt_our_kdtrain_snn_cub.yml --print-yaml &
    # python pl_train_kd_our_sample_snn.py --config config/dogs/plt_our_kdtrain_snn_dogs.yml --print-yaml &
    # nohup python pl_train_kd_our_sample_snn.py --config config/dogs/plt_our_kdtrain_snn_dogs.yml --print-yaml  &
    # erkdsnn qkformer
    # nohup python pl_train_kd_our_sample_snn.py --config config/aircraft/plt_our_kdtrain_snn_aircraft.yml &
    ##  nohup python pl_train_kd_our_sample_snn.py --config config/cubs200/plt_our_kdtrain_snn_cub.yml --print-yaml &
    ##  nohup  python pl_train_kd_our_sample_snn.py --config config/aircraft/plt_our_kdtrain_snn_aircraft.yml &
    # nohup  python pl_train_kd_our_sample_snn.py --config config/nabrids/plt_our_kdtrain_snn_nabrids.yml &
    parser = argparse.ArgumentParser(description='Parse YAML config and optionally print content')
    parser.add_argument('--print-yaml', action='store_true', help='Print raw YAML content')
    parser.add_argument('--config', default='config/dogs/plt_erkdtrain_snn_dogs.yml', help='Path to YAML config file')
    # args = parse_args_yml('config/nabrids/plt_our_kdtrain_snn_nabrids.yml')
    # plt_erkdtrain_snn_nabrids.yml
    args = parser.parse_args()
    if args.print_yaml:
        print_yaml_content(args.config)
    config_args = parse_args_yml(args.config)
    print(vars(config_args))  # Print parsed args
    return config_args

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

    dm = PlCAMDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, crop_size=args.input_size,cam_path=args.cam_path)
    # dm = PlDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, crop_size=args.input_size)
    # set fc layer of model with exact class number of current dataset
    model = LitModel(args=args)
    
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp",precision=args.mixed)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, devices=1, accelerator="gpu",callbacks=[checkpoint_callback], precision=args.mixed, gradient_clip_val=0)
    trainer.fit(model, dm)
    print("end....")

def validate_from_checkpoint():
    args = parse_args()
    from torchvision.datasets.folder import ImageFolder

    # 初始化 DataModule
    dm = PlCAMDataModule(
        batch_size=args.batch_size,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        crop_size=args.input_size,
        cam_path=args.cam_path
    )

    # 加载模型从 checkpoint
    model = LitModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        args=args
    )

    
    # 直接取出
    # 初始化 Logger
    logger_name = args.name_space if hasattr(args, 'name_space') else 'validation'
    logger = CSVLogger(args.checkpoint_dir if hasattr(args, 'checkpoint_dir') else 'logs', name=logger_name)

    # 初始化 Trainer
    if args.is_distributed:
        trainer = pl.Trainer(
            logger=logger,
            accelerator="gpu",
            strategy="ddp",
            precision=args.mixed
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            devices=1,
            accelerator="gpu",
            precision=args.mixed
        )

    # 运行测试集验证
    trainer.test(model, datamodule=dm)
    print("Validation completed.")
    ## save LitModel.feature_extractor chenckpoint
    torch.save(model.feature_extractor.state_dict(), 'snn_checkpoint/feature_extractor_dogs.pth')

if __name__ == "__main__":
    main()
    # validate_from_checkpoint()


# python pl_train_kd_our_sample_snn.py --config config/dogs/plt_our_kdtrain_snn_dogs.yml  --print-yaml 

# /home/h666/code-acm/snn-fvgr/code/snn-fgvr-former/logs/our_sample_dogs/version_1
# python pl_train_kd_our_sample_snn.py --config logs/our_sample_dogs/version_1/hparams.yaml --print-yaml


# nohup python pl_train_kd_our_sample_snn.py --config config/cars196/plt_our_kdtrain_snn_cars.yml --print-yaml > cars196_train.log 2>&1 &
# nohup python pl_train_kd_our_sample_snn.py --config config/dogs/plt_our_kdtrain_snn_dogs.yml --print-yaml > dogs_train.log 2>&1 &
# nohup python pl_train_kd_our_sample_snn.py --config config/aircraft/plt_our_kdtrain_snn_aircraft.yml --print-yaml > aircraft_train.log 2>&1 &
# nohup python pl_train_kd_our_sample_snn.py --config config/aircraft/plt_our_kdtrain_snn_aircraft.yml --print-yaml > aircraft_train.log 2>&1 &

# scp -P 6000 STDC1-Seg-20250929T064822Z-1-001.zip h666@39.104.59.132:/home/h666/eng-code
# scp -P 6000 STDC2-Seg-20250929T064822Z-1-001.zip h666@39.104.59.132:/home/h666/eng-code
# scp -P 6000 STDCNet813M_73.91.tar h666@39.104.59.132:/home/h666/eng-code
# scp -P 6000 STDCNet1446_76.47.tar h666@39.104.59.132:/home/h666/eng-code