import os
import json
from torch.utils.data import Dataset
from random import randint
from PIL import Image
import torch
from torchvision.transforms.functional import crop

class LAION50kDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.ann_dict = {}
        with open(annotations_file, 'rt') as anns_file:
            lines = anns_file.readlines()
            for i in range(1, lines):
                filename = lines[i][:9]
                caption = lines[i][11:]
                self.ann_dict[filename] = caption

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

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
        
        left_x = randint(0, image_width-self.target_dim_x)
        left_y = randint(0, image_height-self.target_dim_y)
        
        x = left_x / image_width
        y = left_y / image_height
        res_x = 1 / image_width
        res_y = 1 / image_height

        return (crop(image, left_x, left_y, self.target_dim_x, self.target_dim_y), torch.tensor([x, y, res_x, res_y]))


    def __getitem__(self, idx):
        key = "{:09d}".format(idx)
        img_path = os.path.join(self.img_dir, key + ".jpg")
        image = Image.open(img_path)
        image, res = self._random_preprocess(image)
        caption = self.ann_dict[key]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            caption = self.target_transform(caption)
        return image, caption, res
