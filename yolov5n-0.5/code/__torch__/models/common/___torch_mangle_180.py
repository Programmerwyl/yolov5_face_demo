class ShuffleV2Block(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  branch1 : __torch__.torch.nn.modules.container.___torch_mangle_170.Sequential
  branch2 : __torch__.torch.nn.modules.container.___torch_mangle_179.Sequential
  def forward(self: __torch__.models.common.___torch_mangle_180.ShuffleV2Block,
    argument_1: Tensor) -> Tensor:
    _0 = self.branch2
    x1, input, = torch.chunk(argument_1, 2, 1)
    x = torch.cat([x1, (_0).forward(input, )], 1)
    batchsize = ops.prim.NumToTensor(torch.size(CONSTANTS.c14, 0))
    _1 = int(batchsize)
    _2 = int(batchsize)
    _3 = ops.prim.NumToTensor(torch.size(CONSTANTS.c14, 1))
    height = ops.prim.NumToTensor(torch.size(CONSTANTS.c14, 2))
    _4 = int(height)
    _5 = int(height)
    width = ops.prim.NumToTensor(torch.size(CONSTANTS.c14, 3))
    _6 = int(width)
    _7 = int(width)
    channels_per_group = torch.floor_divide(_3, CONSTANTS.c1)
    _8 = [_2, 2, int(channels_per_group), _5, _7]
    x0 = torch.view(x, _8)
    x2 = torch.contiguous(torch.transpose(x0, 1, 2))
    return torch.view(x2, [_1, -1, _4, _6])
