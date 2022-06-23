class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.common.___torch_mangle_86.ShuffleV2Block
  __annotations__["1"] = __torch__.models.common.___torch_mangle_97.ShuffleV2Block
  __annotations__["2"] = __torch__.models.common.___torch_mangle_108.ShuffleV2Block
  __annotations__["3"] = __torch__.models.common.___torch_mangle_119.ShuffleV2Block
  __annotations__["4"] = __torch__.models.common.___torch_mangle_130.ShuffleV2Block
  __annotations__["5"] = __torch__.models.common.___torch_mangle_141.ShuffleV2Block
  __annotations__["6"] = __torch__.models.common.___torch_mangle_152.ShuffleV2Block
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_153.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = getattr(self, "6")
    _1 = getattr(self, "5")
    _2 = getattr(self, "4")
    _3 = getattr(self, "3")
    _4 = getattr(self, "2")
    _5 = getattr(self, "1")
    _6 = (getattr(self, "0")).forward(argument_1, )
    _7 = (_3).forward((_4).forward((_5).forward(_6, ), ), )
    _8 = (_0).forward((_1).forward((_2).forward(_7, ), ), )
    return _8
