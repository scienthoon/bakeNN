"""Built-in portable-C op-family registrations.

Adding a family is intentionally a one-line integration change here.  Family
modules own their plan-step constants, kernels, model call, and manifest data;
the central generator contains no operation-type branches.
"""

from . import activation as _activation
from . import linear as _linear
from . import conv as _conv
from . import elementwise as _elementwise
from . import pool as _pool
from . import sequence as _sequence
from . import shape as _shape
from . import softmax as _softmax
from . import spatial as _spatial
from . import tensor as _tensor

__all__: list[str] = []
