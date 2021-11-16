import mmcv
import numpy as np
import imgaug.augmenters as iaa
import torch
from torchvision import transforms
from imgaug.augmenters.geometric import Affine
import torchvision

from PIL import ImageFilter, Image
import random

from ..builder import PIPELINES

DEGREE = 30

_MAX_LEVEL = 10


def level_to_value(level, max_value):
    """Map from level to values based on max_value."""
    return (level / _MAX_LEVEL) * max_value

def enhance_level_to_value(level, a=1.8, b=0.1):
    """Map from level to values."""
    return (level / _MAX_LEVEL) * a + b

def bb_to_array(bbs):
    coords = []
    for bb in bbs:
        coords.append([bb[0], bb[1], bb[2], bb[3]])
    coords = np.array(coords)
    return coords

def bb_to_area(bbs):
    area = (bbs[..., 2] - bbs[..., 0]) * (
            bbs[..., 3] - bbs[..., 1])
    return area

def identity(img, level):
    return img

def adjust_color(img, level):
    level = enhance_level_to_value(level=level)
    return mmcv.adjust_color(img, level).astype(img.dtype)

def posterize(img, level):
    level = int(level_to_value(level=level, max_value=4))
    return mmcv.posterize(img, 4-level).astype(img.dtype)

def solarize(img, level):
    level = level_to_value(level=level, max_value=256)
    return mmcv.solarize(img, level).astype(img.dtype)

def adjust_contrast(img, level):
    level = enhance_level_to_value(level=level)
    return mmcv.adjust_contrast(img, level).astype(img.dtype)

def adjust_brightness(img, level):
    level = enhance_level_to_value(level=level)
    return mmcv.adjust_brightness(img, level).astype(img.dtype)

def imequalize(img, level):
    return mmcv.imequalize(img).astype(img.dtype)

def adjust_sharpness(img, level):
    level = level_to_value(level=level, max_value=1)
    return mmcv.adjust_sharpness(img, level).astype(img.dtype)

def auto_contrast(img, level):
    level = level_to_value(level=level, max_value=1)
    return mmcv.auto_contrast(img, level).astype(img.dtype)



RANDOM_COLOR_POLICY_OPS = [identity, adjust_color, adjust_contrast, adjust_brightness,
                           imequalize, adjust_sharpness, solarize, posterize,
                           auto_contrast]

# 全局转换
AFFINE_TRANSFORM = iaa.Sequential(
    [
        iaa.OneOf([
            Affine(  # TranslateX
                translate_percent={'x': (-0.1, 0.1)},
                order=[0, 1],
                cval=125,
            ),
            Affine(  # TranslateY
                translate_percent={'y': (-0.1, 0.1)},
                order=[0, 1],
                cval=125,
            ),
            Affine(  # Rotate
                rotate=(-DEGREE, DEGREE),
                order=[0, 1],
                cval=125,
            ),
            Affine(  # ShearX and ShareY
                shear=(-DEGREE, DEGREE),
                order=[0, 1],
                cval=125,
            ),
        ]),
    ],
    # do all of the above augmentations in random order
    random_order=True)

# bbox内部转换
AFFINE_TRANSFORM_WEAK = iaa.Sequential(
    [
        iaa.OneOf([
            Affine(
                translate_percent={'x': (-0.05, 0.05)},
                order=[0, 1],
                cval=125,
            ),
            Affine(
                translate_percent={'y': (-0.05, 0.05)},
                order=[0, 1],
                cval=125,
            ),
            Affine(
                rotate=(-10, 10),
                order=[0, 1],
                cval=125,
            ),
            Affine(
                shear=(-10, 10),
                order=[0, 1],
                cval=125,
            ),
        ]),
    ],
    # do all of the above augmentations in random order
    random_order=True)

CUTOUT = iaa.Cutout(nb_iterations=(1, 5), size=[0, 0.2], squared=True)


@PIPELINES.register_module()
class STACTransform:
    def __init__(self, aug_type='strong', magnitude=6, weighted_inbox_selection=False, mode='common'):
        self.affine_aug_op = AFFINE_TRANSFORM
        self.inbox_affine_aug_op = AFFINE_TRANSFORM_WEAK
        self.cutout_op = CUTOUT
        self.magnitude = magnitude
        self.weighted_inbox_selection = weighted_inbox_selection
        self.affine_ops = []
        self.aug_type = aug_type
        self.mode = mode

        self.strong_augment_fn = [[self.color_augment],
                               [self.bbox_affine_transform, self.affine_transform],
                               [self.cutout_augment]]
        # self.augment_fn = [[self.color_augment],
                               # [self.affine_transform],
                            #    [self.cutout_augment]]
        # self.mem_augment_fn = [[self.color_augment],
        #                        [self.bbox_affine_transform, self.affine_transform],
        #                        [self.cutout_augment]]
        self.default_augment_fn = []

    def cutout_augment(self, image, bounding_box=None):
        """Cutout augmentation."""
        image_aug = self.cutout_op(images=[image])[0]
        return image_aug, bounding_box

    def color_augment(self, image, bounding_box=None):
        """RandAug color augmentation."""
        op = np.random.choice(RANDOM_COLOR_POLICY_OPS, 1)[0]
        level = np.random.randint(1, self.magnitude)
        image = op(image, level)
        if bounding_box is None:
            return image, None
        return image, bounding_box

    def bbox_affine_transform(self, image, bounding_box):
        """In-box affine transformation."""
        real_box_n = len(bounding_box)
        shape = image.shape
        boxes = bb_to_array(bounding_box)
        # large area has better probability to be sampled
        if self.weighted_inbox_selection:
            area = bb_to_area(bounding_box)
            k = np.random.choice([i for i in range(real_box_n)],
                                 1,
                                 p=area / area.sum())[0]
        else:
            k = np.random.choice([i for i in range(real_box_n)], 1)[0]
        if len(boxes) > 0:
            box = boxes[k]
            im_crop = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
            im_paste = self.inbox_affine_aug_op(images=[im_crop])[0]
            # in-memory operation
            image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = im_paste
        assert shape == image.shape
        return image, bounding_box

    def affine_transform(self, image, bounding_box=None):
        """Global affine transformation."""
        shape = image.shape
        image_aug, bb_aug = self.affine_aug_op(
            images=[image], bounding_boxes=[bounding_box])
        assert shape == image_aug[0].shape
        if bounding_box is None:
            return image_aug[0], None
        else:
            return image_aug[0], bb_aug[0]

    def __call__(self, results):
        img = results['img']
        bboxes = results['ann_info']['bboxes']
        # label_type = results['label_type']
        if 1 == results['label_type']:
            augment_fn = self.strong_augment_fn
        else:
            if self.mode == 'mem':
                pass
            elif self.mode == 'strong_aug':
                augment_fn = self.strong_augment_fn
            else:
                augment_fn = self.default_augment_fn
        if len(augment_fn
               ) > 0 and augment_fn[-1][0].__name__ == 'cutout_augment':
            # put cutout in the last always
            naug = len(augment_fn)
            order = np.random.permutation(np.arange(naug - 1))
            order = np.concatenate([order, [naug - 1]], 0)
        else:
            order = np.random.permutation(np.arange(len(augment_fn)))

        for i in order:
            fns = augment_fn[i]
            fn = fns[np.random.randint(0, len(fns))]
            img, bboxes = fn(image=img, bounding_box=bboxes)

            # iimage = torch.from_numpy(strong_img.transpose([2, 0, 1]))
            # iimage = transforms.ToPILImage()(iimage).convert('RGB')
            # iimage.show()
            # iimage = torch.from_numpy(weak_img.transpose([2,0,1]))
            # iimage = transforms.ToPILImage()(iimage).convert('RGB')
            # iimage.show()
        results['img'] = img
        results['ann_info']['bboxes'] = bboxes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

@PIPELINES.register_module()
class MOCOTransform:
    def __init__(self, gray_p=0.2, gaussian_sigma=[.1, 2.], gaussian_p=0.5, color_jitter_p=0.8):
        self.random_gray_scale = torchvision.transforms.RandomGrayscale(p=gray_p)
        self.random_gaussian_blur = torchvision.transforms.RandomApply([GaussianBlur(sigma=gaussian_sigma)], p=gaussian_p)
        self.random_colorjitter = torchvision.transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=color_jitter_p),
    
    def __call__(self, results):
        img = results['img']
        img = Image.fromarray(np.uint8(img))
        img = self.random_colorjitter[0](img)
        img = self.random_gray_scale(img)
        img = self.random_gaussian_blur(img)
        img = np.asarray(img)

        # iimage = torch.from_numpy(img.transpose([2, 0, 1]))
        # iimage = transforms.ToPILImage()(iimage).convert('RGB')
        # iimage.save('./work_dirs/img_out/test.jpg')

        results['img'] = img
        return results