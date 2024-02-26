import os.path
from data.base_dataset import BaseDataset, get_transform, get_norm, adjust2test, _resize, imread
from data.image_folder import make_dataset
import torchvision.transforms as transforms

class SingleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt, norm=False)
        self.ttensor = transforms.ToTensor()

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        # A_img = Image.open(A_path).convert('RGB')
        A_img = imread(A_path)
        A_img = self.transform(A_img)
        # if 'unete' in self.opt.netG:
        A_img = _resize(A_img, (2048,2048), False, self.opt.isTrain)
        # if 'unet' in self.opt.netG or 'resnet' in self.opt.netG:
        A_img = adjust2test(A_img)
        A_img_norm = get_norm(A_img, data_norm=self.opt.data_norm)


        A = self.ttensor(A_img_norm)
        A_origin = self.ttensor(A_img)  
        
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        # if input_nc == 1:  # RGB to gray
        #     tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #     A = tmp.unsqueeze(0)

        out_dict = {'A': A, 'A_paths': A_path, 'A_origin': A_origin}
        
            
        return out_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
