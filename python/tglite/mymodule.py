import torch
import math

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
            return LinearFunction_HandleZeroInput.apply(self.weight, input, self.bias)
        else:
            return torch.nn.functional.linear(input.unsqueeze(-1), self.weight, self.bias)
