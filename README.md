# SRPSFusion

**SRPSFusion: SR Aligned Progressive Supervision for Infrared and Visible Image Fusion**

This repository provides the PyTorch implementation of SRPSFusion. It combines infrared and visible feature fusion with multi-level prediction, TVSGA attention, and progressively introduced SR-based supervision.

## Dependencies

- Python 3.10.14
- PyTorch 2.3.1 and torchvision 0.18.1
- CUDA 12.1
- NumPy, Pillow, tqdm, natsort, pandas, matplotlib, timm, ptflops, and kornia

## Training

Prepare matching images in `train/ir`, `train/vi`, `test/ir`, and `test/vi` under the dataset root. 

```bash
python train_SR.py 
```

The SR pretraining weights are included. Other training options are defined in `options_SR.py`. Results, checkpoints, and training curves are saved under `results/SRPSFusion/`.

## Testing

Place inference images in `/path/to/datasets` then provide a trained fusion checkpoint:

```bash
python test_SR.py 
```

Fused images are saved under `fusion_results/MSRS/`. The included `sr_pretrain.pth` contains SR weights and is not a fusion-model checkpoint.
