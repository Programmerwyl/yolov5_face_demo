class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.common.StemBlock
  __annotations__["1"] = __torch__.models.common.ShuffleV2Block
  __annotations__["2"] = __torch__.torch.nn.modules.container.___torch_mangle_59.Sequential
  __annotations__["3"] = __torch__.models.common.___torch_mangle_75.ShuffleV2Block
  __annotations__["4"] = __torch__.torch.nn.modules.container.___torch_mangle_153.Sequential
  __annotations__["5"] = __torch__.models.common.___torch_mangle_169.ShuffleV2Block
  __annotations__["6"] = __torch__.torch.nn.modules.container.___torch_mangle_203.Sequential
  __annotations__["7"] = __torch__.models.common.___torch_mangle_207.Conv
  __annotations__["8"] = __torch__.torch.nn.modules.upsampling.Upsample
  __annotations__["9"] = __torch__.models.common.Concat
  __annotations__["10"] = __torch__.models.common.C3
  __annotations__["11"] = __torch__.models.common.___torch_mangle_232.Conv
  __annotations__["12"] = __torch__.torch.nn.modules.upsampling.___torch_mangle_233.Upsample
  __annotations__["13"] = __torch__.models.common.___torch_mangle_234.Concat
  __annotations__["14"] = __torch__.models.common.___torch_mangle_257.C3
  __annotations__["15"] = __torch__.models.common.___torch_mangle_261.Conv
  __annotations__["16"] = __torch__.models.common.___torch_mangle_262.Concat
  __annotations__["17"] = __torch__.models.common.___torch_mangle_285.C3
  __annotations__["18"] = __torch__.models.common.___torch_mangle_289.Conv
  __annotations__["19"] = __torch__.models.common.___torch_mangle_290.Concat
  __annotations__["20"] = __torch__.models.common.___torch_mangle_313.C3
  __annotations__["21"] = __torch__.models.yolo.Detect
