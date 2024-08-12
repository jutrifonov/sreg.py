# sreg.py/__init__.py
print("sreg package is being imported")
"""
Your Package
============

This package provides tools for performing XYZ operations. It includes modules for A, B, and C.
"""

# Import only the public functions and classes
from .core import sreg, sreg_rgen
from .output import Sreg

__all__ = ["sreg", "sreg_rgen", "Sreg"]

# Expose sreg_rgen and sreg directly in the sreg namespace
sreg_rgen = sreg_rgen
sreg = sreg
Sreg = Sreg
