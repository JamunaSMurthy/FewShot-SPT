"""FewShot-SPT source code."""

__version__ = "1.0.0"
__author__ = "Implementation Team"
__description__ = "Few-Shot Spatiotemporal Perception Transformer for Video Anomaly Detection"

from . import models
from . import datasets
from . import utils
from . import training
from . import configs

__all__ = ['models', 'datasets', 'utils', 'training', 'configs']
