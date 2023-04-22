import os
from math import sqrt
from torch.utils.data import Dataset
from random import randint
from PIL import Image
import torch
from torchvision.transforms.functional import crop

class LAION22kDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.ann_dict = {}
        with open(annotations_file, 'rt') as anns_file:
            lines = anns_file.readlines()
            for i in range(1, len(lines)):
                filename = lines[i][:9]
                caption = lines[i][11:]
                self.ann_dict[filename] = caption

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(list(self.ann_dict.keys()))

    def _random_preprocess(self, image):
        image_width = image.width
        image_height = image.height

        # Random rescaling
        min_side = min(image_width, image_height)
        downscale_factor = randint(512, min_side) / min_side

        image_width = round(image_width * downscale_factor)
        image_height = round(image_height * downscale_factor)

        image = image.resize((image_width, image_height), resample=Image.LANCZOS)
        
        assert image.width >= 512 and image.height >= 512, "dim: [" + str(image.width) + ", " + str(image.height) + "]"
        
        left_x = randint(0, image_width-512)
        left_y = randint(0, image_height-512)
        
        x = left_x / image_width
        y = left_y / image_height
        res_diag = 1/sqrt(image_height*image_height+image_width*image_width)
        
        return (crop(image, left_x, left_y, 512, 512), torch.tensor([x, y, res_diag]))


    def __getitem__(self, idx):
        key = "{:09d}".format(idx)
        img_path = os.path.join(self.img_dir, key + ".jpg")
        image = Image.open(img_path).convert("RGB")
        image, res = self._random_preprocess(image)
        caption = self.ann_dict[key]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            caption = self.target_transform(caption)
        return image, caption, res
