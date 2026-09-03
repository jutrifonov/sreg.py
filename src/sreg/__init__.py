"""Stratified randomized-experiment estimation and inference.

The public API consists of :func:`sreg`, :func:`sreg_rgen`, :class:`Sreg`, and
:func:`AEJapp`. It follows the R ``sreg`` 2.1.0 estimators with idiomatic Python
names and Matplotlib output.
"""

# Import only the public functions and classes
from .core import sreg, sreg_rgen, AEJapp
from .output import Sreg

__all__ = ["sreg", "sreg_rgen", "Sreg", "AEJapp"]

__version__ = "2.1.0"

# Expose sreg_rgen and sreg directly in the sreg namespace
sreg_rgen = sreg_rgen
sreg = sreg
Sreg = Sreg
AEJapp = AEJapp
