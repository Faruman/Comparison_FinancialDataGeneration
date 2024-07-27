"""
**************
Linear Algebra
**************

Linear algebraic functions, calculations of important matrices.
"""

from .matrices import *

from .eigenstuff import *

from .resistance import *

from .fast_bp import *

# import helper functions for use in other places
from .matrices import _flat,_pad,_eps
from .eigenstuff import _eigs
