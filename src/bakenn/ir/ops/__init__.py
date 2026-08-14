"""Built-in immutable quantized-IR operation types."""

from ..op import LinearOp
from .activation import HardSigmoidOp, HardSwishOp, SiLUOp, SigmoidOp
from .conv import Conv2DOp, DepthwiseConv2DOp
from .elementwise import AddOp, ClampOp, MulOp, RequantizeOp
from .pool import AveragePool2DOp, MaxPool2DOp
from .sequence import AveragePool1DOp, Conv1DOp, MaxPool1DOp
from .shape import ConcatenateOp, FlattenOp, ReshapeOp, SliceOp
from .softmax import SoftmaxOp
from .spatial import ConvTranspose2DOp, ResizeBilinear2DOp, ResizeNearest2DOp
from .tensor import Pad2DOp, ReduceMeanOp

__all__ = [
    "AddOp",
    "AveragePool2DOp",
    "AveragePool1DOp",
    "ClampOp",
    "ConcatenateOp",
    "Conv2DOp",
    "ConvTranspose2DOp",
    "Conv1DOp",
    "DepthwiseConv2DOp",
    "FlattenOp",
    "HardSigmoidOp",
    "HardSwishOp",
    "LinearOp",
    "MaxPool2DOp",
    "MaxPool1DOp",
    "MulOp",
    "Pad2DOp",
    "RequantizeOp",
    "ReshapeOp",
    "ReduceMeanOp",
    "ResizeBilinear2DOp",
    "ResizeNearest2DOp",
    "SiLUOp",
    "SigmoidOp",
    "SliceOp",
    "SoftmaxOp",
]
