# -*- coding: utf-8 -*-

"""Top-level package for Zephyr."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dai-lab@mit.edu'
__version__ = '0.0.3'

import os

from zephyr_ml.core import Zephyr
from zephyr_ml.entityset import create_pidata_entityset, create_scada_entityset
from zephyr_ml.labeling import DataLabeler

MLBLOCKS_PRIMITIVES = os.path.join(os.path.dirname(__file__), 'primitives', 'jsons')
MLBLOCKS_PIPELINES = os.path.join(os.path.dirname(__file__), 'pipelines')
