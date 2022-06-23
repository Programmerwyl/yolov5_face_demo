class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_278.Conv
  cv2 : __torch__.models.common.___torch_mangle_282.Conv
  def forward(self: __torch__.models.common.___torch_mangle_283.Bottleneck,
    argument_1: Tensor) -> Tensor:
    _0 = (self.cv2).forward((self.cv1).forward(argument_1, ), )
    return _0
