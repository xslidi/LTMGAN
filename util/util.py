from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import imageio
from torch import nn


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x)
        return x

class NoneLayer(nn.Module):
    '''
    just pass the input to output for sequence
    '''
    def __init__(self, nf):
        super(NoneLayer, self).__init__()

    def forward(self, x):
        return x


def hook(module, input, output):
    setattr(module, "_value_hook", output)

def gram_matrix(image):
    (b, ch, h, w) = image.size()
    feature = image.view(b, ch, h*w)
    feature_t = feature.transpose(1, 2)
    gram = feature.bmm(feature_t) / (ch * h * w)
    return gram

def vgg_preprocess(images):
    tensortype = type(images.data)
    images = (images + 1) * 255 * 0.5
    mean = tensortype(images.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    images = images.sub(mean)
    return images 

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    imageio.imwrite(out_path, img, format='BMP-PIL')



class ReviseImage():
    def __init__(self, transfor_imgs, images):
        self.images = images
        self.transform = transfor_imgs
        self.recover = torch.zeros_like(self.images)

    def tensor2img(self, img, imtype=np.uint8, ldr=True):
        if isinstance(img, torch.Tensor):
            image_tensor = img.data
        else:
            return img
        image_numpy = image_tensor.cpu().float().numpy()
        assert len(image_numpy.shape) == 3
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if ldr:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0
            image_numpy.astype(imtype)
        return image_numpy        

    def luminance_img(self,data_in):
        is_paths = type(data_in[0]) == str
        if is_paths:
            img = imageio.imread(data_in)
        else:
            img = data_in
        luminance = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
        # luminance = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        return luminance
    
    def render_img(self, rgb_in, L_in, L_out, gamma):
        render = np.power((rgb_in / (L_in + 1e-8)), gamma) * L_out
        return render

    def render_img_mantiuk(self, rgb_in, L_in, L_out, c=0.4):
        s = (1 + 1.6774) * c ** 0.9925 / (1 + 1.6774 * c ** 0.9925)
        render = ((rgb_in / L_in - 1) * s + 1) * L_out
        return render


    def recover_img(self, transformed_img, origional_img, gamma=0.5, correct='nomantiuk'):
        # origional_img = origional_img.clip(0)
        # transformed_img = transformed_img.clip(0)
        origional_img = np.abs(origional_img)
        transformed_img = np.abs(transformed_img)
        L_in = self.luminance_img(origional_img)
        if transformed_img.shape[2] == 3:
            L_out = self.luminance_img(transformed_img)
        else:
            L_out = transformed_img
        assert origional_img.shape[2] == 3
        R_in, G_in, B_in = origional_img[:,:,0], origional_img[:,:,1], origional_img[:,:,2]
        if correct == 'mantiuk':
            R_out = self.render_img_mantiuk(R_in, L_in, L_out)
            G_out = self.render_img_mantiuk(G_in, L_in, L_out)
            B_out = self.render_img_mantiuk(B_in, L_in, L_out)  
        else:          
            R_out = self.render_img(R_in, L_in, L_out, gamma)
            G_out = self.render_img(G_in, L_in, L_out, gamma)
            B_out = self.render_img(B_in, L_in, L_out, gamma)
        img = np.dstack((R_out,G_out,B_out))
        img = np.clip(img, 0, 255).astype(np.uint8)

        # save_image(img, out_path)
        # print('already recovery!')
        return img
    
    def img2tensor(self, img):
        assert len(img.shape) == 3
        image_numpy = (np.transpose(img, (2, 0, 1)) - 127.5) / 127.5
        img_tensor = torch.from_numpy(image_numpy)
        return img_tensor 
    
    def saved_batches(self):
        assert len(self.images.size()) == 4
        for i in range(self.images.size()[0]):
            image = self.tensor2img(self.images[i,:,:,:], ldr=False)
            transfor = self.tensor2img(self.transform[i,:,:,:])
            recover_img = self.recover_img(transfor, image)
            self.recover[i,:,:,:] = self.img2tensor(recover_img)
        self.recover.cuda()

