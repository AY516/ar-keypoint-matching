import os
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.WILLOW = edict()
__C.WILLOW.ROOT_DIR = "./data/WILLOW"
__C.WILLOW.CLASSES = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]
__C.WILLOW.KPT_LEN = 10
__C.WILLOW.TRAIN_NUM = 20
__C.WILLOW.TRAIN_OFFSET = 0
__C.WILLOW.VALIDATION = False

__C.PASCALVOC = edict()
__C.PASCALVOC.ROOT_DIR = "./data/PascalVOC"
__C.PASCALVOC.VALIDATION = True

__C.SPair = edict()
__C.SPair.ROOT_DIR = "./data/SPair-71k"
__C.SPair.size = "large"
__C.SPair.CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "train",
    "tvmonitor",
]


#
# Training options
#

__C.TRAIN = edict()
__C.TRAIN.difficulty_params = {}
# Iterations per epochs

__C.EVAL = edict()
__C.EVAL.difficulty_params = {}

# Mean and std to normalize images
__C.NORM_MEANS = [0.485, 0.456, 0.406]
__C.NORM_STD = [0.229, 0.224, 0.225]

# Data cache path
__C.CACHE_PATH = "data/cache"

# random seed used for data loading
__C.RANDOM_SEED = 123