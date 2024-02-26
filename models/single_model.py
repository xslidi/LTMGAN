import torch
import itertools
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import gram_matrix, vgg_preprocess, ReviseImage
from util.vgg import Vgg16


class SingleModel(BaseModel):
    def name(self):
        return 'SingleModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        # parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_identity', type=float, default=5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # define the number of times D trained when G trained once
        self.d_iter = opt.d_iter
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'revise_B']
        visual_names_B = ['real_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_B.append('idt_B')
            self.loss_names.append('idt_B')
        if self.opt.style_weight > 0.0:
            self.loss_names.append('style')
        if self.opt.lambda_gp > 0.0:
            self.loss_names.append('GP')

        if self.opt.style_weight > 0.0:
            gpu_ids = self.opt.gpu_ids
            self.vgg = Vgg16(requires_grad=False).to(gpu_ids[0])

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G, D
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_g, 
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                      True, opt.spectral_norm)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, 
                                          opt.norm_d, use_sigmoid, opt.init_type, opt.init_gain, 
                                          self.gpu_ids, opt.spectral_norm)

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, loss=self.opt.loss).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.L1loss = torch.nn.L1Loss()
            self.mseloss = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr_g, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr_d, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.origin_A = input['A_origin'].to(self.device)


    def forward(self):
        self.fake_B = self.netG(self.real_A)       
        revise_img = ReviseImage(self.fake_B, self.origin_A)
        revise_img.saved_batches()
        self.revise_B = revise_img.recover  

    def backward_D_basic(self, netD, real, fake, loss='none'):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        # Real
        if loss == 'hinge':
            loss_D_real = nn.ReLU()(1.0 - pred_real).mean()
        elif loss == 'wasserstein':
            loss_D_real = - pred_real.mean()
        elif loss == 'relative':
            rel_t = pred_real-pred_fake
            rel_t.requires_grad_(True)
            loss_D_real = self.criterionGAN(rel_t, True)
        else:    
            loss_D_real = self.criterionGAN(pred_real, True)
        # Fake        
        if loss == 'hinge':
            loss_D_fake = nn.ReLU()(1.0 + pred_fake).mean()
        elif loss == 'wasserstein':
            loss_D_fake = pred_fake.mean()
        elif loss == 'relative':
            rel_f = pred_fake-pred_real
            rel_f.requires_grad_(True)
            loss_D_fake = self.criterionGAN(rel_f, False)
        else:
            loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        if loss == 'hinge' or loss == 'wasserstein':
            loss_D = loss_D_real + loss_D_fake
        else:
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward(retain_graph=True)
        return loss_D

    def calc_gradient_penalty(self, netD, real_images, fake_images):
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = alpha * real_images.data + (1 - alpha) * fake_images.data
        interpolated.requires_grad_(True)
        out = netD(interpolated)
        grad = torch.autograd.grad(outputs=out,
                                inputs=interpolated,
                                grad_outputs=torch.ones(out.size()).cuda(),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]  

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        d_loss_gp.backward(retain_graph=True)
        return d_loss_gp     

    def style_loss(self, img, style, same=True, loss='l2'):
        style_loss = 0
        weight = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        # radio = 127.5 / torch.min(style)
        if same:
            radio2 = 250 / torch.max(style)
            style = style * radio2
        else:
            style = vgg_preprocess(style)
        img = vgg_preprocess(img)
        style_vgg = self.vgg(style)
        img_vgg = self.vgg(img)
        if loss == 'l2':
            loss_fn = self.mseloss
        elif loss == 'l1':
            loss_fn = self.L1loss
        for i in range(len(style_vgg)):
            if self.opt.gram_matrix:
                style_vgg[i] = gram_matrix(style_vgg[i]) 
                img_vgg[i] = gram_matrix(img_vgg[i])
            style_loss += weight[i] * loss_fn(style_vgg[i], img_vgg[i]) / len(style_vgg)
        return style_loss


    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.revise_B)
        self.loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B, self.opt.loss)
        if self.opt.lambda_gp > 0.0:
            self.loss_GP = self.calc_gradient_penalty(self.netD, self.real_B, fake_B) * self.opt.lambda_gp


    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        # Identity loss
        if lambda_idt > 0.0:
            # G_A should be identity if real_B is fed.
            if self.opt.exposure:
                self.idt_B = self.netG(self.real_B, self.real_B, self.real_B)
            else:
                self.idt_B = self.netG(self.real_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B) * lambda_idt
        else:
            self.loss_idt_B = 0

        if self.opt.style_weight > 0.0:
            self.loss_style = self.opt.style_weight * self.style_loss(self.revise_B, self.real_B, same=False)
        else:
            self.loss_style = 0

        if self.opt.loss == 'relative':
            fake_B = self.fake_B_pool.query(self.revise_B)
            self.loss_G = self.backward_D_basic(self.netD, fake_B, self.real_B, self.opt.loss)
        else:
            # GAN loss D(G(A))
            self.loss_G = self.criterionGAN(self.netD(self.revise_B), True)

        # combined loss
        self.loss_G = self.loss_G + self.loss_idt_B + self.loss_style
        self.loss_G.backward()
    
    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        for _ in range(self.d_iter):
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
