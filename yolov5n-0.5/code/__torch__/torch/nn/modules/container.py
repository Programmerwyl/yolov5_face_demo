class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.conv.___torch_mangle_12.Conv2d
  __annotations__["1"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_13.BatchNorm2d
  __annotations__["2"] = __torch__.torch.nn.modules.conv.___torch_mangle_14.Conv2d
  __annotations__["3"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_15.BatchNorm2d
  __annotations__["4"] = __torch__.torch.nn.modules.activation.___torch_mangle_16.SiLU
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = getattr(self, "4")
    _1 = getattr(self, "3")
    _2 = getattr(self, "2")
    _3 = getattr(self, "1")
    _4 = (getattr(self, "0")).forward(argument_1, )
    _5 = (_1).forward((_2).forward((_3).forward(_4, ), ), )
    return (_0).forward(_5, )
class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.conv.___torch_mangle_314.Conv2d
  __annotations__["1"] = __torch__.torch.nn.modules.conv.___torch_mangle_315.Conv2d
  __annotations__["2"] = __torch__.torch.nn.modules.conv.___torch_mangle_316.Conv2d
