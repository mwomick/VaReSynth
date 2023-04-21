from cocodata import VRSCocoCaptions
from laion50k import LAION50kDataset

import torchvision.transforms as transforms
from torch.utils import data
import torch

COCO_ONLY = True
COCO_TRAIN_PTH = "/pine/scr/m/w/rwomick/train2017"
COCO_ANN_PTH = "/pine/scr/m/w/rwomick/annotations/captions_train2017.json"
LAION50K_PTH = "/pine/scr/m/w/rwomick/laion-high-resolution/50k"
LAION50K_ANN_PTH = "/pine/scr/m/w/rwomick/laion-high-resolution/50k/captions.csv"
BATCH_SIZE = 1

tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

coco_set = VRSCocoCaptions(root = COCO_TRAIN_PTH, annFile = COCO_ANN_PTH, transform=tf)

if  COCO_ONLY:
    TRAIN_SET = coco_set
else:
    laion50k_set = LAION50kDataset(LAION50K_ANN_PTH, LAION50K_PTH, transform=tf)
    TRAIN_SET = torch.utils.data.ConcatDataset([coco_set, laion50k_set])
    
TRAIN_LOADER = data.DataLoader(TRAIN_SET, BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)