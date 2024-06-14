from . import _version

from .gp_optimizer import GPOptimizer
from .gp_optimizer import fvGPOptimizer
from .autonomous_experimenter import AutonomousExperimenterGP
from .autonomous_experimenter import AutonomousExperimenterFvGP

__all__ = ['GPOptimizer', 'fvGPOptimizer','AutonomousExperimenterGP','AutonomousExperimenterFvGP']

__version__ = _version.get_versions()['version']

