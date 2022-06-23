class C3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_294.Conv
  cv2 : __torch__.models.common.___torch_mangle_298.Conv
  cv3 : __torch__.models.common.___torch_mangle_302.Conv
  m : __torch__.torch.nn.modules.container.___torch_mangle_312.Sequential
  def forward(self: __torch__.models.common.___torch_mangle_313.C3,
    argument_1: Tensor) -> Tensor:
    _0 = self.cv3
    _1 = self.cv2
    _2 = (self.m).forward((self.cv1).forward(argument_1, ), )
    input = torch.cat([_2, (_1).forward(argument_1, )], 1)
    return (_0).forward(input, )
