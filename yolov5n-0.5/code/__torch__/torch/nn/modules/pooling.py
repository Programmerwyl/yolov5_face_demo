class MaxPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.pooling.MaxPool2d,
    argument_1: Tensor) -> Tensor:
    stem_2p_out = torch.max_pool2d(argument_1, [2, 2], [2, 2], [0, 0], [1, 1], True)
    return stem_2p_out
