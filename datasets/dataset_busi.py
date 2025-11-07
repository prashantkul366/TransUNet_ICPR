import os
import random
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return image, label

class RandomGeneratorBUSI(object):
    def __init__(self, output_size):
        self.oh, self.ow = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']  # image: HxW or HxWx3, label: HxW (0/1)

        # augment
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # resize to ViT input
        image = cv2.resize(image, (self.ow, self.oh), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.ow, self.oh), interpolation=cv2.INTER_NEAREST)

        # ensure 3 channels for R50-ViT-B_16 (expects 3)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # normalize to [0,1], to float32
        image = (image.astype(np.float32) / 255.0)

        # HWC->CHW
        image = np.transpose(image, (0,1,2))  # HWC still, just explicit
        image_t = torch.from_numpy(image).permute(2,0,1).contiguous().float()
        label_t = torch.from_numpy(label.astype(np.int64))  # long for CE

        return {'image': image_t, 'label': label_t}

class BUSI_dataset(Dataset):
    """
    Loads BUSI images/masks as PNG/JPG.
    Expects:
      base_dir/
        split/ (train or val)
          images/*.png|jpg
          masks/*.png|jpg   (same stem)
    list_dir provides train.txt / val.txt with stems (without extension).
    """
    def __init__(self, base_dir, list_dir, split, transform=None, image_exts=('.png','.jpg','.jpeg'), mask_exts=('.png','.jpg')):
        assert split in ['train','test']
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        with open(os.path.join(list_dir, split+'.txt')) as f:
            self.stems = [ln.strip() for ln in f if ln.strip()]
        self.img_dir = os.path.join(base_dir, split, 'images')
        self.msk_dir = os.path.join(base_dir, split, 'masks')
        self.image_exts = image_exts
        self.mask_exts = mask_exts

        print("path:", self.img_dir)
        # print(f'BUSI_dataset: {split} set with {len(self.stems)} cases.')

    def __len__(self):
        return len(self.stems)

    def _find_with_ext(self, folder, stem, exts):
        for e in exts:
            p = os.path.join(folder, stem + e)
            if os.path.exists(p):
                return p
        # fallback: scan for any starting with stem
        for fn in os.listdir(folder):
            if os.path.splitext(fn)[0] == stem:
                return os.path.join(folder, fn)
        raise FileNotFoundError(f'File for "{stem}" not found under {folder}')

    def __getitem__(self, idx):
        stem = self.stems[idx]
        ip = self._find_with_ext(self.img_dir, stem, self.image_exts)
        mp = self._find_with_ext(self.msk_dir, stem, self.mask_exts)

        img = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'Failed to read image: {ip}')
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise RuntimeError(f'Failed to read mask: {mp}')

        # binarize mask (0/1)
        msk = (msk > 127).astype(np.uint8)

        sample = {'image': img, 'label': msk, 'case_name': stem}
        if self.transform:
            sample = self.transform(sample)
        return sample
