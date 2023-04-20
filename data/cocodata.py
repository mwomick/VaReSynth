import os.path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from random import randint

import torch
from torchvision.transforms.functional import crop
from torchvision.datasets import VisionDataset
import random

class VRSCoco(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.
    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        target_dim_x: int = 512,
        target_dim_y: int = 512,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        
        # filter out lowres images
        ids = list(sorted(self.coco.imgs.keys()))
        values = [ self.coco.imgs[key] for key in ids ]
        self.ids = [ ids[i] for i in range(len(values)) if min(values[i]['width'], values[i]['height']) >= 512]

        self.target_dim_x = target_dim_x
        self.target_dim_y = target_dim_y

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

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
        
        cropped_image = crop(image, left_x, left_y, self.target_dim_x, self.target_dim_y)
        x = left_x / image_width
        y = left_y / image_height
        res_x = 1 / image_width
        res_y = 1 / image_height

        if self.transforms is not None:
            cropped_image, target = self.transforms(cropped_image, target)

        pos = torch.tensor([x, y, res_x, res_y])

        return cropped_image, target, pos

    def __len__(self) -> int:
        return len(self.ids)


class VRSCocoCaptions(VRSCoco):
    """`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.
    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.PILToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """

    def _load_target(self, id: int) -> List[str]:
        return random.choice([ann["caption"] for ann in super()._load_target(id)])