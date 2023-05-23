import glob
import json
import os
import pickle
import random

import numpy as np
from PIL import Image

from utils.config import cfg

cache_path = cfg.CACHE_PATH
pair_ann_path = cfg.SPair.ROOT_DIR + "/PairAnnotation"
layout_path = cfg.SPair.ROOT_DIR + "/Layout"
image_path = cfg.SPair.ROOT_DIR + "/JPEGImages"
dataset_size = cfg.SPair.size

sets_translation_dict = dict(train="trn", test="test")
difficulty_params_dict = dict(
    trn=cfg.TRAIN.difficulty_params, val=cfg.EVAL.difficulty_params, test=cfg.EVAL.difficulty_params
)

class SPair71k:
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        self.sets = sets_translation_dict[sets]
        self.ann_files = open(os.path.join(layout_path, dataset_size, self.sets + ".txt"), "r").read().split("\n")
        self.ann_files = self.ann_files[: len(self.ann_files) - 1]
        self.difficulty_params = difficulty_params_dict[self.sets]
        self.pair_ann_path = pair_ann_path
        self.image_path = image_path
        self.classes = list(map(lambda x: os.path.basename(x), glob.glob("%s/*" % image_path)))
        self.classes.sort()
        self.obj_resize = obj_resize
        self.combine_classes = False
        self.ann_files_filtered, self.ann_files_filtered_cls_dict, self.classes = self.filter_annotations(
            self.ann_files, self.difficulty_params
        )
        self.total_size = len(self.ann_files_filtered)
        self.size_by_cls = {cls: len(ann_list) for cls, ann_list in self.ann_files_filtered_cls_dict.items()}

        def filter_annotations(self, ann_files, difficulty_params):
            if len(difficulty_params) > 0:
                basepath = os.path.join(self.pair_ann_path, "pickled", self.sets)
                if not os.path.exists(basepath):
                    os.makedirs(basepath)
                difficulty_paramas_str = self.diff_dict_to_str(difficulty_params)
                try:
                    filepath = os.path.join(basepath, difficulty_paramas_str + ".pickle")
                    ann_files_filtered = pickle.load(open(filepath, "rb"))
                    print(
                        f"Found filtered annotations for difficulty parameters {difficulty_params} and {self.sets}-set at {filepath}"
                    )
                except (OSError, IOError) as e:
                    print(
                        f"No pickled annotations found for difficulty parameters {difficulty_params} and {self.sets}-set. Filtering..."
                    )
                    ann_files_filtered_dict = {}

                    for ann_file in ann_files:
                        with open(os.path.join(self.pair_ann_path, self.sets, ann_file + ".json")) as f:
                            annotation = json.load(f)
                        diff = {key: annotation[key] for key in self.difficulty_params.keys()}
                        diff_str = self.diff_dict_to_str(diff)
                        if diff_str in ann_files_filtered_dict:
                            ann_files_filtered_dict[diff_str].append(ann_file)
                        else:
                            ann_files_filtered_dict[diff_str] = [ann_file]
                    total_l = 0
                    for diff_str, file_list in ann_files_filtered_dict.items():
                        total_l += len(file_list)
                        filepath = os.path.join(basepath, diff_str + ".pickle")
                        pickle.dump(file_list, open(filepath, "wb"))
                    assert total_l == len(ann_files)
                    print(f"Done filtering. Saved filtered annotations to {basepath}.")
                    ann_files_filtered = ann_files_filtered_dict[difficulty_paramas_str]
            else:
                print(f"No difficulty parameters for {self.sets}-set. Using all available data.")
                ann_files_filtered = ann_files

            ann_files_filtered_cls_dict = {
                cls: list(filter(lambda x: cls in x, ann_files_filtered)) for cls in self.classes
            }
            class_len = {cls: len(ann_list) for cls, ann_list in ann_files_filtered_cls_dict.items()}
            print(f"Number of annotation pairs matching the difficulty params in {self.sets}-set: {class_len}")
            if self.combine_classes:
                cls_name = "combined"
                ann_files_filtered_cls_dict = {cls_name: ann_files_filtered}
                filtered_classes = [cls_name]
                print(f"Combining {self.sets}-set classes. Total of {len(ann_files_filtered)} image pairs used.")
            else:
                filtered_classes = []
                for cls, ann_f in ann_files_filtered_cls_dict.items():
                    if len(ann_f) > 0:
                        filtered_classes.append(cls)
                    else:
                        print(f"Excluding class {cls} from {self.sets}-set.")
            return ann_files_filtered, ann_files_filtered_cls_dict, filtered_classes

        def diff_dict_to_str(self, diff):
            diff_str = ""
            keys = ["mirror", "viewpoint_variation", "scale_variation", "truncation", "occlusion"]
            for key in keys:
                if key in diff.keys():
                    diff_str += key
                    diff_str += str(diff[key])
            return diff_str