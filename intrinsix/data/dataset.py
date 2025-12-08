import copy
from enum import Enum
import os
import random
from typing import Optional, Callable, Union, Mapping, Any, List, Dict

import cv2
import numpy as np
from torchvision.datasets import VisionDataset

from batch import Batch

from .dataloading import reset_transform_params
from .cache import LoadableObjectCache
from ..io.image_io import load_image
from ..io.data_io import load_data
from ..log import init_logger


class TrainStage(Enum):
    """Definition of the different training stages"""
    Training = "train"
    Validation = "valid"
    Test = "test"

    def is_train(self):
        """
        Checks whether the stage referes to a training stage or not
        :return: True if the stage is Training or Validation
        """
        return self == self.Training

    def __str__(self):
        return self.value

    def from_str(self, val):
        return TrainStage(val)


class InteriorVerseDataset(VisionDataset):
    train_file = "train.txt"
    test_file = "test.txt"

    FEATURES = ["albedo", "normal", "depth", "material", "mask", "caption"]
    EXTENSIONS = {
        "im": ".exr",
        "albedo": ".exr",
        "normal": ".exr",
        "depth": ".exr",
        "material": ".exr",
        "mask": ".exr",
        "caption": ".txt",
    }

    def __init__(self,
                 root: str,
                 fixed_caption: Optional[str] = None,
                 caption_prefix=None,
                 extra_source: Optional[List[str]] = None,
                 stage: TrainStage = TrainStage.Training,
                 features_to_include: Optional[list] = None,
                 fixed_features_to_include: Optional[dict] = None,
                 allow_missing_scenes=True,
                 cache_size=None,
                 transform: Union[Optional[Callable], Mapping[str, Callable]] = None):
        super().__init__(root, transform=transform)
        self.module_logger = init_logger()
        self.extra_source = extra_source

        self.fixed_caption = fixed_caption
        self.caption_prefix = caption_prefix
        self.fixed_features_to_include = fixed_features_to_include

        self.stage = stage if isinstance(stage, TrainStage) else TrainStage(stage)
        self.features_to_include = features_to_include if features_to_include is not None else self.FEATURES
        self.allow_missing_scenes = allow_missing_scenes

        self.module_logger.debug(f"Loading {self.stage} dataset from {self.root}{'+' + str(self.extra_source) if self.extra_source is not None else ''}!")
        self.data = self.load_dataset()
        # assert len(self) > 0, f"Dataset {self.stage} from {self.root} is empty!"
        self.module_logger.debug(f"Dataset {self.stage} from {self.root} loaded (length={len(self)})!")

        self.samples = LoadableObjectCache(self._load_sample, auto_load=True, max_size=cache_size)

    @property
    def instance_prompt(self):
        return self.fixed_caption

    @property
    def custom_instance_prompts(self):
        return self.fixed_caption is None

    @property
    def split_file_path(self) -> str:
        if self.stage == TrainStage.Training:
            return os.path.join(self.root, self.train_file)
        elif self.stage == TrainStage.Validation:
            self.module_logger.warning(
                f"Validation split is not defined for {self.__class__.__name__}, using the test set!")
            return os.path.join(self.root, self.test_file)
        elif self.stage == TrainStage.Test:
            return os.path.join(self.root, self.test_file)
        else:
            raise ValueError(f"Invalid stage {self.stage}!")

    def load_dataset(self):
        data = Batch()

        # Collect the scene list
        with open(self.split_file_path) as f:
            lines = f.readlines()
        data['scene_list'] = [line.rstrip('\n') for line in lines]
        data['scene_list'] = list(filter(lambda x: x != "", data['scene_list']))

        # Sanity check
        scene_folders = []
        for scene_folder_path in data.scene_list:
            # Check if the scene folder exists
            if not os.path.exists(os.path.join(self.root, scene_folder_path)):
                if not self.allow_missing_scenes:
                    raise FileNotFoundError(f"Scene folder {scene_folder_path} does not exist!")
            else:
                scene_folders.append(scene_folder_path)
        data['scene_list'] = scene_folders

        # Collect the features
        self.module_logger.debug("Collecting features")
        data['samples'] = Batch(default=Batch, recursive_separator=".")
        data['sample_ids'] = []

        def collect_features(scene_folder, scene_folder_path):
            for file_name in sorted(os.listdir(scene_folder_path)):
                if "_" not in file_name or "log" in file_name:
                    continue

                view_id, feature = file_name.split('_')
                feature = feature.split(".")[0]
                sample_id = os.path.join(scene_folder, view_id)

                if sample_id not in data['sample_ids']:
                    data['sample_ids'].append(sample_id)
                    
                if feature in self.features_to_include:
                    data['samples'][sample_id][feature] = os.path.join(scene_folder_path,
                                                                           f"{view_id}_{feature}{self.EXTENSIONS[feature]}")

        for scene_folder in data['scene_list']:
            scene_folder_path = os.path.join(self.root, scene_folder)
            collect_features(scene_folder, scene_folder_path)

            if self.extra_source is not None:
                for extra_source in self.extra_source:
                    extra_scene_folder_path = os.path.join(extra_source, scene_folder)
                    collect_features(scene_folder, extra_scene_folder_path)

        # Sanity check
        lengths = [len(list(data['samples'][sample_id].keys())) for sample_id in data['samples'].keys()]
        assert all([lengths[0] == l for l in lengths]), "Missing feature!"

        return data

    def __len__(self) -> int:
        return len(self.data['sample_ids'])

    def get_sample_id(self, index: int) -> str:
        try:
            return self.data['sample_ids'][index]
        except IndexError:
            raise IndexError(f"Index {index} is out of range for dataset {self.__class__.__name__} with length {len(self)}")
        
    def _load_sample(self, index: int, features_to_include=None) -> Any:
        if features_to_include is None:
            features_to_include = copy.deepcopy(self.features_to_include)

        # Load the features
        sample = Batch()
        sample_id = self.get_sample_id(index)
        sample["idx"] = index

        if "caption" in features_to_include:
            caption_path = self.data["samples"][sample_id]["caption"]
            captions = load_data(caption_path)

            caption = random.sample(captions, 1)[0]
            if self.caption_prefix is not None:
                caption = self.caption_prefix + caption

            caption = caption.lower()
            
            # Remove the usual prefix of Florence2
            caption = caption.removeprefix("The image shows ")

            sample["caption"] = caption
            features_to_include.remove("caption")

        for feature in features_to_include:
            image_path = self.data["samples"][sample_id][feature]
            sample[feature] = load_image(image_path)

        # Transform the features
        if self.transform is not None:
            reset_transform_params(self.transform)
            # Apply different transformation to the different features
            sample = self.transform(sample)

        return sample

    def __getitem__(self, index: int) -> Any:
        batch = self.samples[index]
        return batch
