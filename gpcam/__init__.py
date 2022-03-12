from . import _version
from loguru import logger
import sys

__version__ = _version.get_versions()['version']

logger.disable('gpcam')