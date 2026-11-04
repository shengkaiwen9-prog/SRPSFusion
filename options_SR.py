import argparse


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # ---------------------------
        # Data loader related
        # ---------------------------
        self.parser.add_argument('--dataroot', type=str, default='./datasets', help='path of data')  # PET-MRI  MSRS
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')  #  train
        self.parser.add_argument('--batch_size', type=int, default=6, help='batch size')
        self.parser.add_argument('--nThreads', type=int, default=1, help='# of threads for data loader')

        # ---------------------------
        # Training related
        # ---------------------------
        self.parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate for training model')
        self.parser.add_argument('--n_ep', type=int, default=100, help='number of epochs')
        self.parser.add_argument('--n_ep_decay', type=int, default=90,
                                 help='epoch start decay learning rate, set -1 if no decay')
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')

        # ---------------------------
        # Output related
        # ---------------------------
        self.parser.add_argument('--name', type=str,
                                 default='',
                                 help='folder name to save outputs')
        self.parser.add_argument('--class_nb', type=int, default=9, help='class number for segmentation model')
        self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='./results',
                                 help='path for saving result images and models')
        self.parser.add_argument('--display_freq', type=int, default=5, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=5, help='freq (epoch) of saving models')

        # ---------------------------
        # SR (Super-Resolution) module related
        # ---------------------------
        self.parser.add_argument('--sr_pretrain', type=str, default='./sr_pretrain.pth',
                                 help='path to pretrained SR model weights (None = train SR jointly)')
        self.parser.add_argument('--freeze_sr', default=True, action='store_true',
                                 help='if set, freeze SR module weights during training')
        self.parser.add_argument('--sr_repeat_blocks', type=int, default=4,
                                 help='number of residual blocks inside SRNet (default=4, smaller=faster)')
        self.parser.add_argument('--sr_supervision_weight', type=float, default=1.0,
                                 help='loss weight for SR-based high-resolution supervision')

        # 在 TrainOptions 类的 __init__ 方法中添加：
        self.parser.add_argument('--recon_weights', type=str, default='1.0,0.8,0.6,0.4,0.2',
                                 help='reconstruction loss weights for each layer (comma separated, from shallow to deep)')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print(f'{name}: {value}')
        return self.opt


class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # ---------------------------
        # Data loader related
        # ---------------------------
        self.parser.add_argument('--dataroot', type=str, default='./datasets', help='path of data')
        self.parser.add_argument('--dataname', type=str, default='detection', help='name of dataset')  # M3FD FMB MSRS Medical
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--nThreads', type=int, default=1, help='# of threads for data loader')

        # ---------------------------
        # Mode related
        # ---------------------------
        self.parser.add_argument('--class_nb', type=int, default=9, help='class number for segmentation model')
        self.parser.add_argument('--resume', type=str,
                                 default='',
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')

        # ---------------------------
        # Results related
        # ---------------------------
        self.parser.add_argument('--name', type=str, default='', help='folder name to save outputs')
        self.parser.add_argument('--result_dir', type=str,
                                 default='./MSRS_out',
                                 help='path for saving result images and models')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print(f'{name}: {value}')
        return self.opt
