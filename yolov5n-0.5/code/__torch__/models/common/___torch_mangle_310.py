class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_307.Conv2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_308.BatchNorm2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_309.SiLU
  def forward(self: __torch__.models.common.___torch_mangle_310.Conv,
    argument_1: Tensor) -> Tensor:
    _0 = self.act
    _1 = (self.bn).forward((self.conv).forward(argument_1, ), )
    return (_0).forward(_1, )
