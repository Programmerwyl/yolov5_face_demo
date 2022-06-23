class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.conv.___torch_mangle_77.Conv2d
  __annotations__["1"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_78.BatchNorm2d
  __annotations__["2"] = __torch__.torch.nn.modules.activation.___torch_mangle_79.SiLU
  __annotations__["3"] = __torch__.torch.nn.modules.conv.___torch_mangle_80.Conv2d
  __annotations__["4"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_81.BatchNorm2d
  __annotations__["5"] = __torch__.torch.nn.modules.conv.___torch_mangle_82.Conv2d
  __annotations__["6"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_83.BatchNorm2d
  __annotations__["7"] = __torch__.torch.nn.modules.activation.___torch_mangle_84.SiLU
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_85.Sequential,
    input: Tensor) -> Tensor:
    _0 = getattr(self, "7")
    _1 = getattr(self, "6")
    _2 = getattr(self, "5")
    _3 = getattr(self, "4")
    _4 = getattr(self, "3")
    _5 = getattr(self, "2")
    _6 = getattr(self, "1")
    _7 = (getattr(self, "0")).forward(input, )
    _8 = (_4).forward((_5).forward((_6).forward(_7, ), ), )
    _9 = (_1).forward((_2).forward((_3).forward(_8, ), ), )
    return (_0).forward(_9, )
