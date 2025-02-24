import torch
import math
from tglite.gpu_mem_track import *
import nvtx

class LinearFunction_HandleZeroInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, input_num, bias):
        # ctx.save_for_backward(weight, input)
        ctx.save_for_backward(weight)
        return bias.view(1, -1).expand(input_num, -1)

    @staticmethod
    def backward(ctx, d_output):
        # weight, input = ctx.saved_tensors
        weight = ctx.saved_tensors
        # d_weight, d_input, d_bias
        return torch.zeros(weight[0].shape[0], 1, device="cuda:0"), None, None, None, None, None, None, None
        # return d_output*torch.zeros(input.shape[0], 1, device="cuda:0"), None, d_output*torch.zeros(input.shape[0], device="cuda:0"), None, None, None, None, None
        # return d_output*torch.zeros(input.shape[0], 1), d_output*weight, d_output*torch.zeros(input.shape[0]), None, None, None, None, None

class LinearHandleZeroInput(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(LinearHandleZeroInput, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, is_zero_tensor, input):
        if is_zero_tensor == True:
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            ans = LinearFunction_HandleZeroInput.apply(self.weight, input, self.bias)
            return ans
        else:
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            
            # print(f"input {input.unsqueeze(-1).size()}")
            # print(f"weight {self.weight.size()}")
            with nvtx.annotate("TODO m*1*1*n", color="green"):
                ans =  torch.nn.functional.linear(input.unsqueeze(-1), self.weight, self.bias)
            return ans

from torch.nn.modules import Module
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from typing import List, Tuple, Optional, overload
from torch import _VF

class RNNCellBase(Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    # WARNING: bias_ih and bias_hh purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.empty((num_chunks * hidden_size, input_size), **factory_kwargs))
        self.weight_hh = Parameter(torch.empty((num_chunks * hidden_size, hidden_size), **factory_kwargs))
        if bias:
            self.bias_ih = Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
            self.bias_hh = Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, hidden
        - **input** : tensor containing input features
        - **hidden** : tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** : tensor containing the next hidden state
          for each element in the batch

    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the initial hidden
          state where :math:`H_{out}` = `hidden_size`. Defaults to zero if not provided.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the next hidden state.

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        if input.dim() not in (1, 2):
            raise ValueError(f"GRUCell: Expected input to be 1D or 2D, got {input.dim()}D instead")
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(f"GRUCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead")
        is_batched = input.dim() == 2
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        if not is_batched:
            input = input.unsqueeze(0)

        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        ret = _VF.gru_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        if not is_batched:
            ret = ret.squeeze(0)
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        return ret
