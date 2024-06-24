try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError) as ex:
    raise RuntimeError('Running gpcam from source code requires installation. If you would like an editable source '
                       'install, use "pip install -e ." to perform and editable installation.') from ex
from loguru import logger

from .gp_optimizer import GPOptimizer
from .gp_optimizer import fvGPOptimizer
from .autonomous_experimenter import AutonomousExperimenterGP
from .autonomous_experimenter import AutonomousExperimenterFvGP

__all__ = ['GPOptimizer', 'fvGPOptimizer','AutonomousExperimenterGP','AutonomousExperimenterFvGP']

logger.disable('gpcam')
