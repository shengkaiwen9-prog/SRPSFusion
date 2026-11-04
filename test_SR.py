from create_dataset import *
from utils import *
from SRPSFusion_SR import SRPSFusion
from options_SR import *
from saver import resume, save_img_single
from tqdm import tqdm
import torch
import os


def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()
    # define model
    device = torch.device("cuda:{}".format(opts.gpu) if torch.cuda.is_available() else "cpu")
    model = SRPSFusion(opts.class_nb).to(device)
    model = resume(model, model_save_path=opts.resume, device=device, is_train=False)
    model.eval()

    # define dataset
    test_dataset = FusionData(opts)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=False
    )

    # inference
    test_model(test_loader, model, device, opts)


def test_model(test_loader, model, device, opts):
    model.eval()
    test_bar = tqdm(test_loader)
    save_dir = os.path.join(opts.result_dir, opts.dataname)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for it, (img_ir, img_vi, img_names, widths, heights) in enumerate(test_bar):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)

            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y, vi_Cb, vi_Cr = vi_Y.to(device), vi_Cb.to(device), vi_Cr.to(device)

            outputs = model(img_vi, img_ir)
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

            # 输出主融合结果
            fused_rgb = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)

            for i in range(len(img_names)):
                save_path = os.path.join(save_dir, img_names[i])
                save_img_single(fused_rgb[i, ::], save_path, widths[i], heights[i])
                test_bar.set_description(f'Image: {img_names[i]}')


if __name__ == '__main__':
    main()
