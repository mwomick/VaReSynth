BATCH_SIZE = 16
STEPS = 500

import torch

DEVICE = torch.cuda(0)

COCO_TRAIN_PTH = "/pine/scr/m/w/rwomick/train2017"
COCO_ANN_PTH = "/pine/scr/m/w/rwomick/annotations/captions_train2017.json"