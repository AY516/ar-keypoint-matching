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