try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError) as ex:  # pragma: no cover - needs an uninstalled source tree
    raise RuntimeError('Running gpcam from source code requires installation. If you would like an editable source '
                       'install, use "pip install -e ." to perform and editable installation.') from ex
from loguru import logger

from .gp_optimizer import GPOptimizer
from .gp_optimizer import fvGPOptimizer
from .gp_optimizer import LogGPOptimizer
from .gp_optimizer import LogitGPOptimizer
from .gp_mcmc import gpMCMC, ProposalDistribution

__all__ = ['GPOptimizer', 'fvGPOptimizer', 'LogGPOptimizer', 'LogitGPOptimizer',
           'gpMCMC', 'ProposalDistribution']

logger.disable('gpcam')
