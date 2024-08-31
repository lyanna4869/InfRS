import bisect
import copy
import itertools
import logging
import numpy as np
import torch.utils.data
from tabulate import tabulate
from termcolor import colored

from fsdet.utils.comm import get_world_size
from fsdet.utils.env import seed_all_rng
from fsdet.utils.logger import log_first_n

from fsdet.data import samplers
from fsdet.data.catalog import DatasetCatalog, MetadataCatalog
from fsdet.data.common import DatasetFromList, MapDataset
from fsdet.data.dataset_mapper import DatasetMapper
from fsdet.data.detection_utils import check_metadata_consistency

from .build import *