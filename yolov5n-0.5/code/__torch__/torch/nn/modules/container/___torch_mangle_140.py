class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.conv.___torch_mangle_132.Conv2d
  __annotations__["1"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_133.BatchNorm2d
  __annotations__["2"] = __torch__.torch.nn.modules.activation.___torch_mangle_134.SiLU
  __annotations__["3"] = __torch__.torch.nn.modules.conv.___torch_mangle_135.Conv2d
  __annotations__["4"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_136.BatchNorm2d
  __annotations__["5"] = __torch__.torch.nn.modules.conv.___torch_mangle_137.Conv2d
  __annotations__["6"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_138.BatchNorm2d
  __annotations__["7"] = __torch__.torch.nn.modules.activation.___torch_mangle_139.SiLU
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_140.Sequential,
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
