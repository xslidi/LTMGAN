import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import math
import pyexr
import imageio


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0


# just modify the width and height to be multiple of 4
def __adjust(img):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

def _uni(data):
    a = np.min(data)
    b = np.max(data)
    out = (data - a) / (b - a)
    return out

def _normlize(data):
    mean = np.mean(data, axis=(0,1))
    std = np.std(data, axis=(0,1))
    out = (data-mean)/std
    return out

def _log_norm(data):
    # data = data.clip(0)
    data = np.abs(data)
    log_data = np.log(data + 1e-5)
    #log_data = _uni(log_data)
    out = _normlize(log_data)

    return out

def _PQcurve(Lin):
    Lin = np.abs(Lin)
    m = 1305 / 8192
    n = 2523 / 32
    L_in = Lin / 10000
    Lout = ((107 + 2413 * L_in ** m) / (128 + 2392 * L_in ** m)) ** n
    Lout = _uni(Lout)
    L_out = _normlize(Lout)
    return L_out  

def get_norm(data, data_norm='log_norm'):
    if data_norm == 'std_norm':
        data_norm = _normlize(data)
    elif data_norm == 'log_norm':
        data_norm = _log_norm(data)
    elif data_norm == 'none':
        data_norm = data 
    elif data_norm == 'pq':
        data_norm = _PQcurve(data) 
    elif data_norm == 'uni':
        data = _uni(data)
        data_norm = _normlize(data)
    else:
        raise ValueError('--data_norm %s is not a valid option.' % data_norm)
    
    return data_norm

def adjust2test(image):
    ow, oh, _ = image.shape

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 32
    if ow % mult == 0 and oh % mult == 0:
        return image
    w = (ow) // mult
    w = (w) * mult
    h = (oh) // mult
    h = (h) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)
    
    img = _center_crop(image, (w, h))
    return img   


def _resize(img, loadsize, keep_res=False, train=True):
    if not keep_res:
        shrink_h = img.shape[0] / loadsize[0]
        shrink_w = img.shape[1] / loadsize[1]
        shrink = np.minimum(shrink_h,shrink_w)
        if shrink < 1 and not train:
            osize = (img.shape[1], img.shape[0])
            print("smaller image, not resized!")
        else:
            osize = (int(img.shape[1] / shrink), int(img.shape[0] / shrink))
    else:
        osize = loadsize
    resized_img = cv2.resize(img, osize, interpolation=cv2.INTER_CUBIC)
    if len(resized_img.shape) == 2:
        resized_img = np.reshape(resized_img, [resized_img.shape[0], resized_img.shape[1], 1])
        resized_img = resized_img.astype(np.float32)
    return resized_img

def _random_crop(img, crop_size):
    h, w = img.shape[:2]
    if crop_size[0] <= h and crop_size[1] <= w:
        x = 0 if crop_size[0] == h else np.random.randint(0, h - crop_size[0])
        y = 0 if crop_size[1] == w else np.random.randint(0, w - crop_size[1])
        return img[x:(crop_size[0] + x), y:(crop_size[1] + y), :]
    else:
        print("Warning: Crop size is larger than original size")
        return img

def _center_crop(img, crop_size):
    h, w = img.shape[:2]
    if crop_size[0] <= h and crop_size[1] <= w:
        x = math.ceil(h - crop_size[0]) // 2
        y = math.ceil(w - crop_size[1]) // 2
        return img[x:(crop_size[0] + x), y:(crop_size[1] + y), :]
    else:
        print("Warning: Crop size is larger than original size")
        return img

def _horizontal_flip(im, prob=0.5):
    """Performs horizontal flip (used for training)."""
    return im[:, ::-1, :] if np.random.uniform() < prob else im  


def get_transform(opt, norm=True):
    transform_list = []
    osize = (opt.loadSize, opt.loadSize)
    fsize = (opt.fineSize, opt.fineSize)
    if opt.resize_or_crop == 'resize_and_crop':
        transform_list.append(transforms.Lambda(lambda img: _resize(img, osize, train=opt.isTrain)))
        transform_list.append(transforms.Lambda(lambda img: _random_crop(img, fsize)))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.Lambda(lambda img: _random_crop(img, fsize)))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.Lambda(lambda img: _random_crop(img, fsize)))
    elif opt.resize_or_crop == 'none':
        pass
        # transform_list.append(transforms.Lambda(
        #     lambda img: __adjust(img)))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: _horizontal_flip(img)))


    transform_list.append(transforms.Lambda(lambda img: np.ascontiguousarray(img)))

    if norm:                                     
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def imread(file):
    img = None
    if file.endswith(".exr"):
        img = pyexr.read(file)
    else:
        img = imageio.imread(file)
    if len(img.shape) == 2:
        img = img[..., None]
    return img