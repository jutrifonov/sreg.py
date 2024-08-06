# sreg.py/__init__.py
print("sreg package is being imported")

# Import only the public functions and classes
from .core import sreg, sreg_rgen
from .output import Sreg

__all__ = ["sreg", "sreg_rgen", "Sreg"]