# ============================================================
# train_sr_pretrain.py  ——  用于训练超分辨率 (SRNet) 预训练权重
# 训练目标：让 SRNet 学习从低分辨率红外/可见光图像恢复原始高分辨率图像
# 输出：sr_pretrain.pth，可被 SRPSFusion_SR 模型加载
# ============================================================

import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# === 导入你的数据集与模型 ===
from create_dataset import MSRSData   # 你项目里的读取器
from SRPSFusion_SR import SRNet           # SR 网络结构
from utils import RGB2YCrCb              # 与 train_SR.py 一致

# ------------------------------------------------------------
# 参数定义
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SR network for SRPSFusion")
    parser.add_argument('--dataroot', type=str, default='./datasets/MSRS',
                        help='dataset root, e.g. /ai/CYF/Fusion/data/MSRS')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scale', type=int, default=2, choices=[2,4],
                        help='upscale factor for SRNet (must be power of 2)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./sr_pretrain.pth')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--sr_repeat_blocks', type=int, default=4,
                        help='number of residual blocks inside SRNet (same as fusion model)')
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()
    return args


# ------------------------------------------------------------
# 降采样函数（bicubic 模拟低分辨率输入）
# ------------------------------------------------------------
def make_lr(hr, scale):
    return F.interpolate(hr, scale_factor=1.0/scale, mode='bicubic', align_corners=False)


# ------------------------------------------------------------
# 主训练过程
# ------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # ==== 构造 MSRSData 的 opts ====
    # MSRSData 需要至少有 dataroot、phase 等属性
    opts = argparse.Namespace()
    opts.dataroot = args.dataroot
    opts.phase = 'train'
    opts.dataname = os.path.basename(args.dataroot.strip('/'))
    opts.nThreads = args.num_workers
    opts.batch_size = args.batch_size

    # ==== 构建训练集 ====
    print(f"Loading dataset from: {opts.dataroot}")
    train_dataset = MSRSData(opts, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # ==== 构建 SR 网络 ====
    srnet = SRNet(in_ch=1, feat_channels=64,
                  repeat_blocks=args.sr_repeat_blocks, up_times=args.scale).to(device)

    # ==== 损失与优化器 ====
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(srnet.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.6), int(args.epochs * 0.85)], gamma=0.2)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    best_loss = float('inf')

    # ==== 训练 ====
    for epoch in range(1, args.epochs + 1):
        srnet.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, ncols=120, desc=f"Epoch [{epoch}/{args.epochs}]")

        for step, data in enumerate(pbar):
            # === MSRSData 通常返回 (ir, vi, label) ===
            img_ir, img_vi, *_ = data
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)

            # === 可见光转 Y 通道 ===
            if img_vi.size(1) == 3:
                try:
                    vi_Y, _, _ = RGB2YCrCb(img_vi)
                except Exception:
                    r = img_vi[:, 0:1, :, :]
                    g = img_vi[:, 1:2, :, :]
                    b = img_vi[:, 2:3, :, :]
                    vi_Y = 0.299 * r + 0.587 * g + 0.114 * b
            else:
                vi_Y = img_vi[:, 0:1, :, :]

            # === 将红外和可见光Y合并成训练批次 ===
            hr = torch.cat([img_ir, vi_Y], dim=0)  # [2B,1,H,W]
            lr = make_lr(hr, args.scale)            # 下采模拟
            sr_pred = srnet(lr)
            loss = criterion(sr_pred, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (step + 1) % args.log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        scheduler.step()
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch:03d} | Avg Loss: {epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # 保存最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(srnet.state_dict(), args.save_path)
            print(f"  ✅ Saved best SRNet state_dict to {args.save_path} (loss={best_loss:.6f})")

    print("Training completed. Best loss:", best_loss)
    print(f"Final SR weights saved to: {args.save_path}")


if __name__ == '__main__':
    main()
