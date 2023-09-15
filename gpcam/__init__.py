from . import _version
from loguru import logger
import sys

from .gp_optimizer import GPOptimizer
from .gp_optimizer import fvGPOptimizer

__all__ = ['GPOptimizer', 'fvGPOptimizer']

__version__ = _version.get_versions()['version']

logger.disable('gpcam')
