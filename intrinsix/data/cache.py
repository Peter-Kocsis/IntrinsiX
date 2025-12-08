import copy
import fnmatch
import os
import time
import warnings
from argparse import Namespace
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Mapping, List, Optional, Union

import torch
import numpy as np

from ..log import init_logger


class LoadableObject:
    def __init__(self, load_function, val=None):
        self.load_function = load_function
        self._val = val

    def reload(self):
        self._val = self.load_function()

    @property
    def val(self):
        if self._val is None:
            self._val = self.load_function()
        return self._val


class LoadableObjectList:
    def __init__(self, lodabable_objects: List[LoadableObject]):
        self.lodabable_objects = lodabable_objects

    @property
    def val(self):
        return [lodabable_object.val for lodabable_object in self.lodabable_objects]

    def __getitem__(self, k):
        return self.lodabable_objects[k]


class LoadableObjectCache:
    def __init__(self, load_function, auto_load=True, max_size=None, name=None):
        self.name = name
        self.module_logger = init_logger()
        self.load_function = load_function
        self.max_size = max_size
        self.auto_load = auto_load
        self.cache = OrderedDict()

    def __getitem__(self, index):
        if index not in self.cache:
            # self.module_logger.debug(f"Obj cache size - {self.name}: {len(self.cache)}")
            # Remove oldest sample if max size exceeded
            if self.max_size is not None and self.max_size <= len(self.cache) and len(self.cache) > 0:
                del self.cache[list(self.cache.keys())[0]]

            # Add the new element
            loadable_object = LoadableObject(partial(self.load_function, index=index))
            if self.auto_load:
                loadable_object = loadable_object.val
            if self.max_size is not None and self.max_size > 0:
                self.cache[index] = loadable_object
        else:
            loadable_object = self.cache[index]
        return loadable_object

    def clear(self):
        self.cache = OrderedDict()


def split_list(tensor, chunk_size, dim=0):
    assert dim == 0, "Only dim=0 split is supported for now"
    return [tensor[i:i + chunk_size] for i in range(0, len(tensor), chunk_size)]
