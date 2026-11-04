# train_with_sr.py
# 在你的原训练脚本基础上做最小改动，新增 SR 模块实例与训练流程使用 SR 上采结果做高分辨率监督

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from create_dataset import *
from utils import *
from SRPSFusion_SR import SRPSFusion  # 假设你已将 SRPSFusion 文件替换为带 SR 的版本
from options_SR import *
from saver import Saver, resume
from time import time
from tqdm import tqdm
from optimizer import Optimizer
import datetime
import os
import torch.nn.functional as F

def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()
    # define model, optimiser and scheduler
    device = torch.device("cuda:{}".format(opts.gpu) if torch.cuda.is_available() else "cpu")
    model = SRPSFusion(opts.class_nb).to(device)

    # --- 新增：SR 网络在模型内部注册（如果你的 SRPSFusion 已包含 sr_shared 则无需在此重新创建） ---
    # 如果你希望单独加载 SR 预训练权重，可以通过 opts.sr_pretrain 指定路径：
    # if hasattr(model, 'sr_shared') and getattr(opts, 'sr_pretrain', None):
    #     try:
    #         sr_ckpt = torch.load(opts.sr_pretrain, map_location=device)
    #         # 假设 ckpt 是 state_dict
    #         model.sr_shared.load_state_dict(sr_ckpt)
    #         if getattr(opts, 'freeze_sr', False):
    #             for p in model.sr_shared.parameters():
    #                 p.requires_grad = False
    #         print("Loaded SR pretrain:", opts.sr_pretrain)
    #     except Exception as e:
    #         print("Failed to load SR pretrain:", e)
    if hasattr(model, 'sr_shared') and getattr(opts, 'sr_pretrain', None):
        try:
            ck = torch.load(opts.sr_pretrain, map_location=device)
            if isinstance(ck, dict):
                # 支持两种：1) 直接 sr state_dict  2) 完整 ckpt（包含 'sr_state_dict' / 'state_dict' / 'model' 等）
                if 'sr_state_dict' in ck:
                    state = ck['sr_state_dict']
                elif 'sr_shared' in ck:
                    state = ck['sr_shared']
                elif 'state_dict' in ck and isinstance(ck['state_dict'], dict) and all(
                        k.startswith('sr_shared') for k in ck['state_dict'].keys()):
                    # 从整体 state_dict 中提取 sr_shared 部分（如果命名空间里有 sr_shared）
                    state = {k.replace('sr_shared.', ''): v for k, v in ck['state_dict'].items() if
                             k.startswith('sr_shared.')}
                elif all(k.startswith('head') or k.startswith('body') or k.startswith('up_blocks') or k.startswith(
                        'tail') for k in ck.keys()):
                    # 直接是 sr state_dict
                    state = ck
                else:
                    # 最后尝试直接把 ck 当作 state_dict
                    state = ck
            else:
                state = ck
            model.sr_shared.load_state_dict(state, strict=False)
            if getattr(opts, 'freeze_sr', False):
                for p in model.sr_shared.parameters():
                    p.requires_grad = False
            print("Loaded SR pretrain:", opts.sr_pretrain)
        except Exception as e:
            print("Failed to load SR pretrain:", e)

    # define dataset
    train_dataset = MSRSData(opts, is_train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.nThreads,
        shuffle=True)
    test_dataset = MSRSData(opts, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=opts.nThreads,
        shuffle=False)

    ep_iter = len(train_loader)
    max_iter = opts.n_ep * ep_iter
    print('Training iter: {}'.format(max_iter))
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-3
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optimizer = Optimizer(
        model=model,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power)
    ep = 0
    total_it = 0
    log_dir = os.path.join(opts.display_dir, 'logger', opts.name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logger_config(log_path=log_path, logging_name='Timer')

    # Train and evaluate multi-task network
    multi_task_trainer(train_loader,
                       test_loader,
                       model,
                       device,
                       optimizer,
                       opts,
                       logger,
                       ep,
                       total_it)

def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, opt, logger, start_ep=0,
                       total_it=0):
    total_epoch = opt.n_ep
    saver = Saver(opt)
    best_mIou = 0.0

    start = glob_st = time()

    class VIEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.firstconv_rgb = model.firstconv_rgb
            self.firstbn_rgb = model.firstbn_rgb
            self.firstrelu_rgb = model.firstrelu_rgb
            self.firstmaxpool_rgb = model.firstmaxpool_rgb
            self.encoder_rgb1 = model.encoder_rgb1
            self.encoder_rgb2 = model.encoder_rgb2
            self.encoder_rgb3 = model.encoder_rgb3
            self.encoder_rgb4 = model.encoder_rgb4

        def parameters(self):
            params = []
            for module in [
                self.firstconv_rgb,
                self.firstbn_rgb,
                self.encoder_rgb1,
                self.encoder_rgb2,
                self.encoder_rgb3,
                self.encoder_rgb4
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    class IREncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.firstconv_ir = model.firstconv_ir
            self.firstbn_ir = model.firstbn_ir
            self.firstrelu_ir = model.firstrelu_ir
            self.firstmaxpool_ir = model.firstmaxpool_ir
            self.encoder_ir1 = model.encoder_ir1
            self.encoder_ir2 = model.encoder_ir2
            self.encoder_ir3 = model.encoder_ir3
            self.encoder_ir4 = model.encoder_ir4

        def parameters(self):
            params = []
            for module in [
                self.firstconv_ir,
                self.firstbn_ir,
                self.encoder_ir1,
                self.encoder_ir2,
                self.encoder_ir3,
                self.encoder_ir4
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    class Fusion(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.fuse5 = model.fuse5
            self.fuse4 = model.fuse4
            self.fuse3 = model.fuse3
            self.fuse2 = model.fuse2
            self.fuse1 = model.fuse1

        def parameters(self):
            params = []
            for module in [
                self.fuse5,
                self.fuse4,
                self.fuse3,
                self.fuse2,
                self.fuse1,
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    class Decoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.Decoder5 = model.Decoder5
            self.Decoder4 = model.Decoder4
            self.Decoder3 = model.Decoder3
            self.Decoder2 = model.Decoder2
            self.Decoder1 = model.Decoder1

        def parameters(self):
            params = []
            for module in [
                self.Decoder5,
                self.Decoder4,
                self.Decoder3,
                self.Decoder2,
                self.Decoder1,
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    class Pred(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.pred_fusion = model.pred_fusion
            self.pred_fusion2 = model.pred_fusion2
            self.pred_fusion3 = model.pred_fusion3
            self.pred_fusion4 = model.pred_fusion4
            self.pred_fusion5 = model.pred_fusion5

            self.pred_vi = model.pred_vi
            self.pred_ir = model.pred_ir

        def parameters(self):
            params = []
            for module in [
                self.pred_fusion,
                self.pred_fusion2,
                self.pred_fusion3,
                self.pred_fusion4,
                self.pred_fusion5,

                self.pred_vi,
                self.pred_ir,
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    class Layer1(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.firstconv_rgb = model.firstconv_rgb
            self.firstbn_rgb = model.firstbn_rgb
            self.firstrelu_rgb = model.firstrelu_rgb

            self.firstconv_ir = model.firstconv_ir
            self.firstbn_ir = model.firstbn_ir
            self.firstrelu_ir = model.firstrelu_ir

            self.fuse1 = model.fuse1
            self.Decoder1 = model.Decoder1
            self.pred_fusion = model.pred_fusion
            self.pred_vi = model.pred_vi
            self.pred_ir = model.pred_ir

        def parameters(self):
            params = []
            for module in [
                self.firstconv_rgb,
                self.firstbn_rgb,
                self.firstrelu_rgb,
                self.firstconv_ir,
                self.firstbn_ir,
                self.firstrelu_ir,
                self.fuse1,
                self.Decoder1,
                self.pred_fusion,
                self.pred_vi,
                self.pred_ir,
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    class Layer2(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder_rgb1 = model.encoder_rgb1
            self.encoder_ir1 = model.encoder_ir1

            self.fuse2 = model.fuse2
            self.Decoder2 = model.Decoder2
            self.pred_fusion2 = model.pred_fusion2

        def parameters(self):
            params = []
            for module in [
                self.encoder_rgb1,
                self.encoder_ir1,
                self.fuse2,
                self.Decoder2,
                self.pred_fusion2,
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    class Layer3(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder_rgb2 = model.encoder_rgb2
            self.encoder_ir2 = model.encoder_ir2

            self.fuse3 = model.fuse3
            self.Decoder3 = model.Decoder3
            self.pred_fusion3 = model.pred_fusion3

        def parameters(self):
            params = []
            for module in [
                self.encoder_rgb2,
                self.encoder_ir2,
                self.fuse3,
                self.Decoder3,
                self.pred_fusion3,
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    class Layer4(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder_rgb3 = model.encoder_rgb3
            self.encoder_ir3 = model.encoder_ir3

            self.fuse4 = model.fuse4
            self.Decoder4 = model.Decoder4
            self.pred_fusion4 = model.pred_fusion4

        def parameters(self):
            params = []
            for module in [
                self.encoder_rgb3,
                self.encoder_ir3,
                self.fuse4,
                self.Decoder4,
                self.pred_fusion4,
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    class Layer5(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder_rgb4 = model.encoder_rgb4
            self.encoder_ir4 = model.encoder_ir4

            self.fuse5 = model.fuse5
            self.Decoder5 = model.Decoder5
            self.pred_fusion5 = model.pred_fusion5

        def parameters(self):
            params = []
            for module in [
                self.encoder_rgb4,
                self.encoder_ir4,
                self.fuse5,
                self.Decoder5,
                self.pred_fusion5,
            ]:
                params += list(module.parameters())
            return params

        def forward(self, x):
            return x

    VIEncoder = VIEncoder(multi_task_model)
    IREncoder = IREncoder(multi_task_model)
    Fusion = Fusion(multi_task_model)
    Decoder = Decoder(multi_task_model)
    Pred = Pred(multi_task_model)

    Layer1 = Layer1(multi_task_model)
    Layer2 = Layer2(multi_task_model)
    Layer3 = Layer3(multi_task_model)
    Layer4 = Layer4(multi_task_model)
    Layer5 = Layer5(multi_task_model)

    track_modules = {
        'VIEncoder': VIEncoder,
        'IREncoder': IREncoder,
        'Fusion': Fusion,
        'Decoder': Decoder,
        'Pred': Pred,

        'Layer1': Layer1,
        'Layer2': Layer2,
        'Layer3': Layer3,
        'Layer4': Layer4,
        'Layer5': Layer5,
    }

    weight_changes = {module_name: [] for module_name in track_modules}
    module_prev_weights = {module_name: None for module_name in track_modules}
    prev_all_weights = None

    results_df = pd.DataFrame(columns=['Epoch', 'Total_Weight_Change'] +
                                      [f'{name}_Weight_Change' for name in track_modules] +
                                      ['Fusion_Loss', 'VI_Recon_Loss', 'IR_Recon_Loss'])

    result_dir = os.path.join(opt.result_dir, opt.name)
    os.makedirs(result_dir, exist_ok=True)

    for ep in range(start_ep, total_epoch):
        multi_task_model.train()
        epoch_fusion_loss = 0.0
        epoch_vi_re_loss = 0.0
        epoch_ir_re_loss = 0.0
        for it, (img_ir, img_vi, label) in enumerate(train_loader):
            total_it += 1
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            label = label.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)

            # 注意：model.forward 现在返回更多元素（包括 SR 上采的 fused_img2_sr ...）
            outputs = multi_task_model(img_vi, img_ir)
            if len(outputs) == 7:
                # 兼容旧版本
                fused_img, re_vi, re_ir, fused_img2, fused_img3, fused_img4, fused_img5 = outputs
                fused_img2_sr = fused_img3_sr = fused_img4_sr = fused_img5_sr = None
                vi_img2 = vi_img3 = vi_img4 = vi_img5 = None
                ir_img2 = ir_img3 = ir_img4 = ir_img5 = None
                vi_img2_sr = vi_img3_sr = vi_img4_sr = vi_img5_sr = None
                ir_img2_sr = ir_img3_sr = ir_img4_sr = ir_img5_sr = None
            else:
                (fused_img, re_vi, re_ir,
                 fused_img2, fused_img3, fused_img4, fused_img5,
                 fused_img2_sr, fused_img3_sr, fused_img4_sr, fused_img5_sr,
                 vi_img2, vi_img3, vi_img4, vi_img5,
                 ir_img2, ir_img3, ir_img4, ir_img5,
                 vi_img2_sr, vi_img3_sr, vi_img4_sr, vi_img5_sr,
                 ir_img2_sr, ir_img3_sr, ir_img4_sr, ir_img5_sr) = outputs

            optimizer.zero_grad()

            # 各层重构损失权重
            recon_weights = {
                'layer1': 1.0,  # 最浅层权重最高
                'layer2': 0.8,
                'layer3': 0.6,
                'layer4': 0.4,
                'layer5': 0.2  # 最深层权重最低
            }

            # 保留原有的下采监督（不变）——以保证你原来的多尺度损失仍在
            vi_Y2 = F.max_pool2d(vi_Y, kernel_size=2, stride=2)
            vi_Y3 = F.max_pool2d(vi_Y2, kernel_size=2, stride=2)
            vi_Y4 = F.max_pool2d(vi_Y3, kernel_size=2, stride=2)
            vi_Y5 = F.max_pool2d(vi_Y4, kernel_size=2, stride=2)
            img_ir2 = F.max_pool2d(img_ir, kernel_size=2, stride=2)
            img_ir3 = F.max_pool2d(img_ir2, kernel_size=2, stride=2)
            img_ir4 = F.max_pool2d(img_ir3, kernel_size=2, stride=2)
            img_ir5 = F.max_pool2d(img_ir4, kernel_size=2, stride=2)

            # ---------------- 新：如果 sr 上采可用，优先使用 sr_upscaled 与 原图 对齐监督 ----------------
            # fusion_loss1: 仍用原 fusion_loss1（full-res fused_img vs vi_Y/img_ir）
            fusion_loss1, int_loss, grad_loss, corr_loss = Fusion_loss(vi_Y, img_ir, fused_img, weights=[1, 10, 10], device=device)

            # 若 sr_upscaled 存在，就用 sr_upscaled 与 原图对齐监督（高分辨率监督），并同时保留原先的下采监督作为辅助
            if fused_img2_sr is not None:
                fusion_loss2_sr, _, _, _ = Fusion_loss(vi_Y, img_ir, fused_img2_sr, weights=[0, 10, 10], device=device)
            else:
                fusion_loss2_sr = torch.tensor(0.0, device=device)

            if fused_img3_sr is not None:
                fusion_loss3_sr, _, _, _ = Fusion_loss(vi_Y, img_ir, fused_img3_sr, weights=[0, 10, 10], device=device)
            else:
                fusion_loss3_sr = torch.tensor(0.0, device=device)

            if fused_img4_sr is not None:
                fusion_loss4_sr, _, _, _ = Fusion_loss(vi_Y, img_ir, fused_img4_sr, weights=[0, 10, 10], device=device)
            else:
                fusion_loss4_sr = torch.tensor(0.0, device=device)

            if fused_img5_sr is not None:
                fusion_loss5_sr, _, _, _ = Fusion_loss(vi_Y, img_ir, fused_img5_sr, weights=[0, 10, 10], device=device)
            else:
                fusion_loss5_sr = torch.tensor(0.0, device=device)

            # 为兼容并保守起见，仍然计算原先基于下采的 losses（作为辅助；你也可以在 opts 中调整是否启用）
            fusion_loss2, _, _, _ = Fusion_loss(vi_Y2, img_ir2, fused_img2, weights=[1, 10, 10], device=device)
            fusion_loss3, _, _, _ = Fusion_loss(vi_Y3, img_ir3, fused_img3, weights=[1, 10, 10], device=device)
            fusion_loss4, _, _, _ = Fusion_loss(vi_Y4, img_ir4, fused_img4, weights=[1, 10, 10], device=device)
            fusion_loss5, _, _, _ = Fusion_loss(vi_Y5, img_ir5, fused_img5, weights=[1, 10, 10], device=device)

            # 第一层重构损失（全分辨率，不需要超分）
            vi_re_loss, vi_int_loss1, vi_grad_loss1 = Re_loss(re_vi, vi_Y, ir_flag=False)
            ir_re_loss, ir_int_loss1, ir_grad_loss1 = Re_loss(re_ir, img_ir, ir_flag=True)
            # 第二层重构损失 - 使用超分后的结果与原始图像计算
            if vi_img2_sr is not None:
                vi_re_loss2, vi_int_loss2, vi_grad_loss2 = Re_loss(vi_img2_sr, vi_Y, ir_flag=False)
                ir_re_loss2, ir_int_loss2, ir_grad_loss2 = Re_loss(ir_img2_sr, img_ir, ir_flag=True)
            else:
                vi_re_loss2 = ir_re_loss2 = torch.tensor(0.0, device=device)
            # 第三层重构损失 - 使用超分后的结果与原始图像计算
            if vi_img3_sr is not None:
                vi_re_loss3, vi_int_loss3, vi_grad_loss3 = Re_loss(vi_img3_sr, vi_Y, ir_flag=False)
                ir_re_loss3, ir_int_loss3, ir_grad_loss3 = Re_loss(ir_img3_sr, img_ir, ir_flag=True)
            else:
                vi_re_loss3 = ir_re_loss3 = torch.tensor(0.0, device=device)
            # 第四层重构损失 - 使用超分后的结果与原始图像计算
            if vi_img4_sr is not None:
                vi_re_loss4, vi_int_loss4, vi_grad_loss4 = Re_loss(vi_img4_sr, vi_Y, ir_flag=False)
                ir_re_loss4, ir_int_loss4, ir_grad_loss4 = Re_loss(ir_img4_sr, img_ir, ir_flag=True)
            else:
                vi_re_loss4 = ir_re_loss4 = torch.tensor(0.0, device=device)
            # 第五层重构损失 - 使用超分后的结果与原始图像计算
            if vi_img5_sr is not None:
                vi_re_loss5, vi_int_loss5, vi_grad_loss5 = Re_loss(vi_img5_sr, vi_Y, ir_flag=False)
                ir_re_loss5, ir_int_loss5, ir_grad_loss5 = Re_loss(ir_img5_sr, img_ir, ir_flag=True)
            else:
                vi_re_loss5 = ir_re_loss5 = torch.tensor(0.0, device=device)
            # 总重构损失
            # total_vi_re_loss = (recon_weights['layer1'] * vi_re_loss +
            #                     recon_weights['layer2'] * vi_re_loss2 +
            #                     recon_weights['layer3'] * vi_re_loss3 +
            #                     recon_weights['layer4'] * vi_re_loss4 +
            #                     recon_weights['layer5'] * vi_re_loss5)
            # total_ir_re_loss = (recon_weights['layer1'] * ir_re_loss +
            #                     recon_weights['layer2'] * ir_re_loss2 +
            #                     recon_weights['layer3'] * ir_re_loss3 +
            #                     recon_weights['layer4'] * ir_re_loss4 +
            #                     recon_weights['layer5'] * ir_re_loss5)

            # 组装总损失：主尺度使用 fusion_loss1；多尺度既保留原来的下采损失也加入 sr 上采后的高分辨率损失作为增强
            # 权重策略保持与原来一致，同时加入 sr losses（权重可通过 opt 调整）
            sr_weight = getattr(opt, 'sr_supervision_weight', 1.0)  # 默认乘 1.0，可在 options 中设定
            # combine multiscale: 使用原跟随你原来的阶段式权重策略，这里示范最终阶段合并所有 sr losses
            if (ep + 1) > 0 and (ep + 1) <= (total_epoch // 2 + 0 * total_epoch // 10):
                train_loss = 1 * fusion_loss1 + 10 * vi_re_loss + 10 * ir_re_loss
            elif (ep + 1) > (total_epoch // 2 + 0 * total_epoch // 10) and (ep + 1) <= (
                    total_epoch // 2 + 1 * total_epoch // 10):
                # total_loss = fusion_loss1 + fusion_loss2 + fusion_loss2_sr
                # w1 = fusion_loss1 / (total_loss + 1e-9)
                # w2 = fusion_loss2 / (total_loss + 1e-9)
                # train_loss = (1 + w1) * fusion_loss1 + w2 * fusion_loss2 + sr_weight * fusion_loss2_sr + 10 * vi_re_loss + 10 * ir_re_loss

                total_loss = fusion_loss1 + fusion_loss2_sr
                w1 = fusion_loss1 / (total_loss + 1e-9)
                w2 = fusion_loss2_sr / (total_loss + 1e-9)
                total_lossre = vi_re_loss + vi_re_loss2 + ir_re_loss + ir_re_loss2
                w1re = (vi_re_loss + ir_re_loss) / (total_lossre + 1e-9)
                w2re = (vi_re_loss2 + ir_re_loss2) / (total_lossre + 1e-9)
                train_loss = ((1 + w1) * fusion_loss1 + sr_weight * w2 * fusion_loss2_sr +
                              10 * ((1 + w1re) * vi_re_loss + w2re * vi_re_loss2
                                    + (1 + w1re) * ir_re_loss + w2re * ir_re_loss2))
            elif (ep + 1) > (total_epoch // 2 + 1 * total_epoch // 10) and (ep + 1) <= (
                    total_epoch // 2 + 2 * total_epoch // 10):
                # total_loss = fusion_loss1 + fusion_loss2 + fusion_loss3 + fusion_loss2_sr + fusion_loss3_sr
                # w1 = fusion_loss1 / (total_loss + 1e-9)
                # w2 = fusion_loss2 / (total_loss + 1e-9)
                # w3 = fusion_loss3 / (total_loss + 1e-9)
                # train_loss = (1 + w1) * fusion_loss1 + w2 * fusion_loss2 + w3 * fusion_loss3 + sr_weight * (fusion_loss2_sr + fusion_loss3_sr) + 10 * vi_re_loss + 10 * ir_re_loss

                total_loss = fusion_loss1 + fusion_loss2_sr + fusion_loss3_sr
                w1 = fusion_loss1 / (total_loss + 1e-9)
                w2 = fusion_loss2_sr / (total_loss + 1e-9)
                w3 = fusion_loss3_sr / (total_loss + 1e-9)
                total_lossre = vi_re_loss + vi_re_loss2 + vi_re_loss3 + ir_re_loss + ir_re_loss2 + ir_re_loss3
                w1re = (vi_re_loss + ir_re_loss) / (total_lossre + 1e-9)
                w2re = (vi_re_loss2 + ir_re_loss2) / (total_lossre + 1e-9)
                w3re = (vi_re_loss3 + ir_re_loss3) / (total_lossre + 1e-9)
                train_loss = ((1 + w1) * fusion_loss1 + sr_weight * (w2 * fusion_loss2_sr + w3 * fusion_loss3_sr) +
                              10 * ((1 + w1re) * vi_re_loss + w2re * vi_re_loss2 + w3re * vi_re_loss3
                                    + (1 + w1re) * ir_re_loss + w2re * ir_re_loss2 + w3re * ir_re_loss3))
            elif (ep + 1) > (total_epoch // 2 + 2 * total_epoch // 10) and (ep + 1) <= (
                    total_epoch // 2 + 3 * total_epoch // 10):
                # total_loss = fusion_loss1 + fusion_loss2 + fusion_loss3 + fusion_loss4 + fusion_loss2_sr + fusion_loss3_sr + fusion_loss4_sr
                # w1 = fusion_loss1 / (total_loss + 1e-9)
                # w2 = fusion_loss2 / (total_loss + 1e-9)
                # w3 = fusion_loss3 / (total_loss + 1e-9)
                # w4 = fusion_loss4 / (total_loss + 1e-9)
                # train_loss = (1 + w1) * fusion_loss1 + w2 * fusion_loss2 + w3 * fusion_loss3 + w4 * fusion_loss4 + sr_weight * (fusion_loss2_sr + fusion_loss3_sr + fusion_loss4_sr) + 10 * vi_re_loss + 10 * ir_re_loss

                total_loss = fusion_loss1 + fusion_loss2_sr + fusion_loss3_sr + fusion_loss4_sr
                w1 = fusion_loss1 / (total_loss + 1e-9)
                w2 = fusion_loss2_sr / (total_loss + 1e-9)
                w3 = fusion_loss3_sr / (total_loss + 1e-9)
                w4 = fusion_loss4_sr / (total_loss + 1e-9)
                total_lossre = vi_re_loss + vi_re_loss2 + vi_re_loss3 + vi_re_loss4 + ir_re_loss + ir_re_loss2 + ir_re_loss3 + ir_re_loss4
                w1re = (vi_re_loss + ir_re_loss) / (total_lossre + 1e-9)
                w2re = (vi_re_loss2 + ir_re_loss2) / (total_lossre + 1e-9)
                w3re = (vi_re_loss3 + ir_re_loss3) / (total_lossre + 1e-9)
                w4re = (vi_re_loss4 + ir_re_loss4) / (total_lossre + 1e-9)
                train_loss = ((1 + w1) * fusion_loss1 + sr_weight * (w2 * fusion_loss2_sr + w3 * fusion_loss3_sr + w4 * fusion_loss4_sr) +
                              10 * ((1 + w1re) * vi_re_loss + w2re * vi_re_loss2 + w3re * vi_re_loss3 + w4re * vi_re_loss4
                                    + (1 + w1re) * ir_re_loss + w2re * ir_re_loss2 + w3re * ir_re_loss3 + w4re * ir_re_loss4))
            else:
                # total_loss = fusion_loss1 + fusion_loss2 + fusion_loss3 + fusion_loss4 + fusion_loss5 + fusion_loss2_sr + fusion_loss3_sr + fusion_loss4_sr + fusion_loss5_sr
                # w1 = fusion_loss1 / (total_loss + 1e-9)
                # w2 = fusion_loss2 / (total_loss + 1e-9)
                # w3 = fusion_loss3 / (total_loss + 1e-9)
                # w4 = fusion_loss4 / (total_loss + 1e-9)
                # w5 = fusion_loss5 / (total_loss + 1e-9)
                # train_loss = (1 + w1) * fusion_loss1 + w2 * fusion_loss2 + w3 * fusion_loss3 + w4 * fusion_loss4 + w5 * fusion_loss5 + sr_weight * (fusion_loss2_sr + fusion_loss3_sr + fusion_loss4_sr + fusion_loss5_sr) + 10 * vi_re_loss + 10 * ir_re_loss

                total_loss = fusion_loss1 + fusion_loss2_sr + fusion_loss3_sr + fusion_loss4_sr + fusion_loss5_sr
                w1 = fusion_loss1 / (total_loss + 1e-9)
                w2 = fusion_loss2_sr / (total_loss + 1e-9)
                w3 = fusion_loss3_sr / (total_loss + 1e-9)
                w4 = fusion_loss4_sr / (total_loss + 1e-9)
                w5 = fusion_loss5_sr / (total_loss + 1e-9)
                total_lossre = vi_re_loss + vi_re_loss2 + vi_re_loss3 + vi_re_loss4 + vi_re_loss5 + ir_re_loss + ir_re_loss2 + ir_re_loss3 + ir_re_loss4 + ir_re_loss5
                w1re = (vi_re_loss + ir_re_loss) / (total_lossre + 1e-9)
                w2re = (vi_re_loss2 + ir_re_loss2) / (total_lossre + 1e-9)
                w3re = (vi_re_loss3 + ir_re_loss3) / (total_lossre + 1e-9)
                w4re = (vi_re_loss4 + ir_re_loss4) / (total_lossre + 1e-9)
                w5re = (vi_re_loss5 + ir_re_loss5) / (total_lossre + 1e-9)
                train_loss = ((1 + w1) * fusion_loss1 + sr_weight * (w2 * fusion_loss2_sr + w3 * fusion_loss3_sr + w4 * fusion_loss4_sr + w5 * fusion_loss5_sr) +
                              10 * ((1 + w1re) * vi_re_loss + w2re * vi_re_loss2 + w3re * vi_re_loss3 + w4re * vi_re_loss4 + w5re * vi_re_loss5
                                    + (1 + w1re) * ir_re_loss + w2re * ir_re_loss2 + w3re * ir_re_loss3 + w4re * ir_re_loss4 + w5re * ir_re_loss5))
            # total_loss = fusion_loss1 + fusion_loss2_sr + fusion_loss3_sr + fusion_loss4_sr + fusion_loss5_sr
            # w1 = fusion_loss1 / (total_loss + 1e-9)
            # w2 = fusion_loss2_sr / (total_loss + 1e-9)
            # w3 = fusion_loss3_sr / (total_loss + 1e-9)
            # w4 = fusion_loss4_sr / (total_loss + 1e-9)
            # w5 = fusion_loss5_sr / (total_loss + 1e-9)
            # total_lossre = vi_re_loss + vi_re_loss2 + vi_re_loss3 + vi_re_loss4 + vi_re_loss5 + ir_re_loss + ir_re_loss2 + ir_re_loss3 + ir_re_loss4 + ir_re_loss5
            # w1re = (vi_re_loss + ir_re_loss) / (total_lossre + 1e-9)
            # w2re = (vi_re_loss2 + ir_re_loss2) / (total_lossre + 1e-9)
            # w3re = (vi_re_loss3 + ir_re_loss3) / (total_lossre + 1e-9)
            # w4re = (vi_re_loss4 + ir_re_loss4) / (total_lossre + 1e-9)
            # w5re = (vi_re_loss5 + ir_re_loss5) / (total_lossre + 1e-9)
            # train_loss = ((1 + w1) * fusion_loss1 + sr_weight * (
            #             w2 * fusion_loss2_sr + w3 * fusion_loss3_sr + w4 * fusion_loss4_sr + w5 * fusion_loss5_sr) +
            #               10 * ((1 + w1re) * vi_re_loss + w2re * vi_re_loss2 + w3re * vi_re_loss3 + w4re * vi_re_loss4 + w5re * vi_re_loss5
            #                     + (1 + w1re) * ir_re_loss + w2re * ir_re_loss2 + w3re * ir_re_loss3 + w4re * ir_re_loss4 + w5re * ir_re_loss5))
            # train_loss = 1 * fusion_loss1 + 10 * vi_re_loss + 10 * ir_re_loss
            # total_loss = fusion_loss1 + fusion_loss2_sr + fusion_loss3_sr + fusion_loss4_sr + fusion_loss5_sr
            # w1 = fusion_loss1 / (total_loss + 1e-9)
            # w2 = fusion_loss2_sr / (total_loss + 1e-9)
            # w3 = fusion_loss3_sr / (total_loss + 1e-9)
            # w4 = fusion_loss4_sr / (total_loss + 1e-9)
            # w5 = fusion_loss5_sr / (total_loss + 1e-9)
            # train_loss = (1 + w1) * fusion_loss1 + sr_weight * (
            #             w2 * fusion_loss2_sr + w3 * fusion_loss3_sr + w4 * fusion_loss4_sr + w5 * fusion_loss5_sr) + 10 * vi_re_loss + 10 * ir_re_loss

            train_loss.backward()
            optimizer.step()

            epoch_fusion_loss += fusion_loss1.item()
            epoch_vi_re_loss += vi_re_loss.item()
            epoch_ir_re_loss += ir_re_loss.item()

        avg_fusion_loss = epoch_fusion_loss / len(train_loader)
        avg_vi_re_loss = epoch_vi_re_loss / len(train_loader)
        avg_ir_re_loss = epoch_ir_re_loss / len(train_loader)

        lr = optimizer.get_lr()
        end = time()
        training_time, glob_t_intv = end - start, end - glob_st
        now_it = total_it + 1
        eta = int((total_epoch * len(train_loader) - now_it) * (glob_t_intv / (now_it)))
        eta = str(datetime.timedelta(seconds=eta))
        logger.info(
            'ep: [{}/{}], learning rate: {:.6f}, time consuming: {:.2f}s, fusion loss: {:.4f}, vi rec loss: {:.4f}, ir rec loss: {:.4f}'.format(
                ep + 1, total_epoch, lr, training_time, avg_fusion_loss, avg_vi_re_loss, avg_ir_re_loss))
        logger.info(
            'grad loss: [{:.4f}], int loss: [{:.4f}], corr loss: [{:.4f}], Eta: {}\n'.format(
                grad_loss.item(), int_loss.item(), corr_loss.item(), eta))

        start = time()

        current_weights = {}
        for module_name, module in track_modules.items():
            current_weights[module_name] = [param.clone().detach().cpu().numpy()
                                            for param in module.parameters()]

        if ep == 0:
            module_prev_weights = current_weights
            logger.info("Recorded initial weights for all modules")

            all_weights = []
            for weights in current_weights.values():
                for w in weights:
                    all_weights.append(w.flatten())
            prev_all_weights = np.concatenate(all_weights) if all_weights else np.array([])
        else:
            epoch_weight_diffs = {}

            for module_name in track_modules:
                prev = module_prev_weights[module_name]
                curr = current_weights[module_name]

                if not prev or not curr:
                    logger.warning(f"Module {module_name} has no trainable parameters")
                    continue

                if len(prev) != len(curr):
                    logger.error(f"Parameter count mismatch for {module_name}: prev={len(prev)}, curr={len(curr)}")
                    continue

                diffs = []
                for i in range(len(prev)):
                    if prev[i].shape != curr[i].shape:
                        logger.error(
                            f"Shape mismatch for {module_name} param {i}: prev={prev[i].shape}, curr={curr[i].shape}")
                        continue

                    diff_norm = np.linalg.norm(curr[i] - prev[i])
                    diffs.append(diff_norm)

                if diffs:
                    module_diff = np.mean(diffs)
                    epoch_weight_diffs[module_name] = module_diff
                    weight_changes[module_name].append(module_diff)
                else:
                    logger.warning(f"No valid parameters for {module_name}")
                    epoch_weight_diffs[module_name] = 0.0

            all_weights = []
            for weights in current_weights.values():
                for w in weights:
                    all_weights.append(w.flatten())
            curr_all_weights = np.concatenate(all_weights) if all_weights else np.array([])

            total_change = np.linalg.norm(curr_all_weights - prev_all_weights)

            module_prev_weights = current_weights
            prev_all_weights = curr_all_weights

            new_row = {
                'Epoch': ep + 1,
                'Total_Weight_Change': total_change,
                'Fusion_Loss': avg_fusion_loss,
                'VI_Recon_Loss': avg_vi_re_loss,
                'IR_Recon_Loss': avg_ir_re_loss
            }

            for name in track_modules:
                new_row[f'{name}_Weight_Change'] = epoch_weight_diffs.get(name, 0.0)

            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

            results_df.to_csv(os.path.join(result_dir, 'training_results.csv'), index=False)

            logger.info(f"Epoch {ep + 1} Weight Changes:")
            logger.info(f"  Total Change (Norm): {total_change:.6f}")
            for name, diff in epoch_weight_diffs.items():
                logger.info(f"  {name}: {diff:.6f}")

            epochs = results_df['Epoch'].values
            plt.figure(figsize=(12, 8))

            functional_modules = ['VIEncoder', 'IREncoder', 'Fusion', 'Decoder', 'Pred']
            colors = plt.cm.tab10.colors

            for i, name in enumerate(functional_modules):
                if weight_changes[name]:
                    plt.plot(epochs, weight_changes[name], label=name, color=colors[i % len(colors)])

            plt.xlabel('Epoch')
            plt.ylabel('Weight Change (Frobenius Norm)')
            plt.title('Functional Modules Weight Changes')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'functional_weight_changes.png'))
            plt.close()

            plt.figure(figsize=(12, 8))

            hierarchical_modules = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5']
            colors = plt.cm.viridis(np.linspace(0, 1, len(hierarchical_modules)))

            for i, name in enumerate(hierarchical_modules):
                if weight_changes[name]:
                    plt.plot(epochs, weight_changes[name], label=name, color=colors[i])

            plt.xlabel('Epoch')
            plt.ylabel('Weight Change (Frobenius Norm)')
            plt.title('Hierarchical Layers Weight Changes')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'hierarchical_weight_changes.png'))
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(epochs, results_df['Fusion_Loss'], label='Fusion Loss', linewidth=2)
            plt.plot(epochs, results_df['VI_Recon_Loss'], label='VI Reconstruction Loss', linewidth=2)
            plt.plot(epochs, results_df['IR_Recon_Loss'], label='IR Reconstruction Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'training_losses.png'))
            plt.close()

            fig, ax1 = plt.subplots(figsize=(12, 8))

            color = 'tab:blue'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Weight Change', color=color)

            ax1.plot(epochs, results_df['Total_Weight_Change'],
                     label='Total Change', linewidth=3, color='black', linestyle='--')

            functional_colors = plt.cm.tab10.colors
            for i, name in enumerate(functional_modules):
                if weight_changes[name]:
                    ax1.plot(epochs, weight_changes[name], label=name,
                             color=functional_colors[i % len(functional_colors)], linestyle='-')

            hierarchical_colors = plt.cm.viridis(np.linspace(0, 1, len(hierarchical_modules)))
            for i, name in enumerate(hierarchical_modules):
                if weight_changes[name]:
                    ax1.plot(epochs, weight_changes[name], label=name,
                             color=hierarchical_colors[i], linestyle=':')

            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left')
            ax1.grid(True)

            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Loss Value', color=color)
            ax2.plot(epochs, results_df['Fusion_Loss'], label='Fusion Loss', linestyle='-.', color='darkred',
                     linewidth=2)
            ax2.plot(epochs, results_df['VI_Recon_Loss'], label='VI Recon Loss', linestyle=':', color='darkorange',
                     linewidth=2)
            ax2.plot(epochs, results_df['IR_Recon_Loss'], label='IR Recon Loss', linestyle='--', color='darkgreen',
                     linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper right')

            plt.title('Weight Changes and Training Losses')
            fig.tight_layout()
            plt.savefig(os.path.join(result_dir, 'combined_metrics.png'))
            plt.close()

        if (ep + 1) % opt.img_save_freq == 0:
            input = [img_ir, img_vi, vi_Y, label]
            fused_rgb = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            vi_rgb = YCbCr2RGB(re_vi, vi_Cb, vi_Cr)
            output = [re_ir, vi_rgb, fused_img, fused_rgb]
            saver.write_img(ep, input, output)

        if (ep + 1) % opt.model_save_freq == 0:
            if (ep + 1) > (opt.n_ep - 10):
                saver.write_model(ep, total_it, multi_task_model, optimizer.optim, best_mIou, device, is_best=False)

if __name__ == '__main__':
    main()

