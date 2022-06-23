class Conv2d(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_126.Conv2d,
    argument_1: Tensor) -> Tensor:
    input = torch._convolution(argument_1, self.weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
    return input
