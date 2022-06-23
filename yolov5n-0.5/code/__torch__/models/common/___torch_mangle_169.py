class ShuffleV2Block(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  branch1 : __torch__.torch.nn.modules.container.___torch_mangle_159.Sequential
  branch2 : __torch__.torch.nn.modules.container.___torch_mangle_168.Sequential
  def forward(self: __torch__.models.common.___torch_mangle_169.ShuffleV2Block,
    argument_1: Tensor) -> Tensor:
    _0 = self.branch2
    _1 = (self.branch1).forward(argument_1, )
    x = torch.cat([_1, (_0).forward(argument_1, )], 1)
    batchsize = ops.prim.NumToTensor(torch.size(CONSTANTS.c13, 0))
    _2 = int(batchsize)
    _3 = int(batchsize)
    _4 = ops.prim.NumToTensor(torch.size(CONSTANTS.c13, 1))
    height = ops.prim.NumToTensor(torch.size(CONSTANTS.c13, 2))
    _5 = int(height)
    _6 = int(height)
    width = ops.prim.NumToTensor(torch.size(CONSTANTS.c13, 3))
    _7 = int(width)
    _8 = int(width)
    channels_per_group = torch.floor_divide(_4, CONSTANTS.c1)
    _9 = [_3, 2, int(channels_per_group), _6, _8]
    x0 = torch.view(x, _9)
    x1 = torch.contiguous(torch.transpose(x0, 1, 2))
    return torch.view(x1, [_2, -1, _5, _7])
