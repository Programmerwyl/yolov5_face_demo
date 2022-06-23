class StemBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  stem_1 : __torch__.models.common.Conv
  stem_2a : __torch__.models.common.___torch_mangle_3.Conv
  stem_2b : __torch__.models.common.___torch_mangle_7.Conv
  stem_2p : __torch__.torch.nn.modules.pooling.MaxPool2d
  stem_3 : __torch__.models.common.___torch_mangle_11.Conv
  def forward(self: __torch__.models.common.StemBlock,
    x: Tensor) -> Tensor:
    _0 = self.stem_3
    _1 = self.stem_2p
    _2 = self.stem_2b
    _3 = self.stem_2a
    _4 = (self.stem_1).forward(x, )
    _5 = [(_2).forward((_3).forward(_4, ), ), (_1).forward(_4, )]
    input = torch.cat(_5, 1)
    return (_0).forward(input, )
class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.Conv2d
  bn : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  act : __torch__.torch.nn.modules.activation.SiLU
  def forward(self: __torch__.models.common.Conv,
    x: Tensor) -> Tensor:
    _6 = self.act
    _7 = (self.bn).forward((self.conv).forward(x, ), )
    return (_6).forward(_7, )
class ShuffleV2Block(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  branch1 : __torch__.torch.nn.modules.container.Sequential
  branch2 : __torch__.torch.nn.modules.container.___torch_mangle_25.Sequential
  def forward(self: __torch__.models.common.ShuffleV2Block,
    argument_1: Tensor) -> Tensor:
    _8 = self.branch2
    _9 = (self.branch1).forward(argument_1, )
    x = torch.cat([_9, (_8).forward(argument_1, )], 1)
    batchsize = ops.prim.NumToTensor(torch.size(CONSTANTS.c0, 0))
    _10 = int(batchsize)
    _11 = int(batchsize)
    _12 = ops.prim.NumToTensor(torch.size(CONSTANTS.c0, 1))
    height = ops.prim.NumToTensor(torch.size(CONSTANTS.c0, 2))
    _13 = int(height)
    _14 = int(height)
    width = ops.prim.NumToTensor(torch.size(CONSTANTS.c0, 3))
    _15 = int(width)
    _16 = int(width)
    channels_per_group = torch.floor_divide(_12, CONSTANTS.c1)
    _17 = [_11, 2, int(channels_per_group), _14, _16]
    x0 = torch.view(x, _17)
    x1 = torch.contiguous(torch.transpose(x0, 1, 2))
    return torch.view(x1, [_10, -1, _13, _15])
class Concat(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.models.common.Concat,
    argument_1: Tensor,
    argument_2: Tensor) -> Tensor:
    input = torch.cat([argument_1, argument_2], 1)
    return input
class C3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_211.Conv
  cv2 : __torch__.models.common.___torch_mangle_215.Conv
  cv3 : __torch__.models.common.___torch_mangle_219.Conv
  m : __torch__.torch.nn.modules.container.___torch_mangle_228.Sequential
  def forward(self: __torch__.models.common.C3,
    argument_1: Tensor) -> Tensor:
    _18 = self.cv3
    _19 = self.cv2
    _20 = (self.m).forward((self.cv1).forward(argument_1, ), )
    input = torch.cat([_20, (_19).forward(argument_1, )], 1)
    return (_18).forward(input, )
class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_223.Conv
  cv2 : __torch__.models.common.___torch_mangle_227.Conv
  def forward(self: __torch__.models.common.Bottleneck,
    argument_1: Tensor) -> Tensor:
    _21 = (self.cv2).forward((self.cv1).forward(argument_1, ), )
    return _21
