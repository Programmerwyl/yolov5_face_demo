class Model(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  model : __torch__.torch.nn.modules.container.___torch_mangle_317.Sequential
  def forward(self: __torch__.models.yolo.Model,
    x: Tensor) -> List[Tensor]:
    _0 = getattr(self.model, "21")
    _1 = getattr(self.model, "20")
    _2 = getattr(self.model, "19")
    _3 = getattr(self.model, "18")
    _4 = getattr(self.model, "17")
    _5 = getattr(self.model, "16")
    _6 = getattr(self.model, "15")
    _7 = getattr(self.model, "14")
    _8 = getattr(self.model, "13")
    _9 = getattr(self.model, "12")
    _10 = getattr(self.model, "11")
    _11 = getattr(self.model, "10")
    _12 = getattr(self.model, "9")
    _13 = getattr(self.model, "8")
    _14 = getattr(self.model, "7")
    _15 = getattr(self.model, "6")
    _16 = getattr(self.model, "5")
    _17 = getattr(self.model, "4")
    _18 = getattr(self.model, "3")
    _19 = getattr(self.model, "2")
    _20 = getattr(self.model, "1")
    _21 = (getattr(self.model, "0")).forward(x, )
    _22 = (_19).forward((_20).forward(_21, ), )
    _23 = (_17).forward((_18).forward(_22, ), )
    _24 = (_15).forward((_16).forward(_23, ), )
    _25 = (_14).forward(_24, )
    _26 = (_12).forward((_13).forward(_25, ), _23, )
    _27 = (_10).forward((_11).forward(_26, ), )
    _28 = (_8).forward((_9).forward(_27, ), _22, )
    _29 = (_7).forward(_28, )
    _30 = (_5).forward((_6).forward(_29, ), _27, )
    _31 = (_4).forward(_30, )
    _32 = (_2).forward((_3).forward(_31, ), _25, )
    _33 = (_0).forward(_29, _31, (_1).forward(_32, ), )
    _34, _35, _36, = _33
    return [_34, _35, _36]
class Detect(Module):
  __parameters__ = []
  __buffers__ = ["anchors", "anchor_grid", ]
  anchors : Tensor
  anchor_grid : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  m : __torch__.torch.nn.modules.container.ModuleList
  def forward(self: __torch__.models.yolo.Detect,
    argument_1: Tensor,
    argument_2: Tensor,
    argument_3: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    _37 = getattr(self.m, "2")
    _38 = getattr(self.m, "1")
    _39 = (getattr(self.m, "0")).forward(argument_1, )
    bs = ops.prim.NumToTensor(torch.size(_39, 0))
    _40 = int(bs)
    ny = ops.prim.NumToTensor(torch.size(_39, 2))
    _41 = int(ny)
    nx = ops.prim.NumToTensor(torch.size(_39, 3))
    _42 = torch.view(_39, [_40, 3, 16, _41, int(nx)])
    _43 = torch.contiguous(torch.permute(_42, [0, 1, 3, 4, 2]))
    _44 = (_38).forward(argument_2, )
    bs0 = ops.prim.NumToTensor(torch.size(_44, 0))
    _45 = int(bs0)
    ny0 = ops.prim.NumToTensor(torch.size(_44, 2))
    _46 = int(ny0)
    nx0 = ops.prim.NumToTensor(torch.size(_44, 3))
    _47 = torch.view(_44, [_45, 3, 16, _46, int(nx0)])
    _48 = torch.contiguous(torch.permute(_47, [0, 1, 3, 4, 2]))
    _49 = (_37).forward(argument_3, )
    bs1 = ops.prim.NumToTensor(torch.size(_49, 0))
    _50 = int(bs1)
    ny1 = ops.prim.NumToTensor(torch.size(_49, 2))
    _51 = int(ny1)
    nx1 = ops.prim.NumToTensor(torch.size(_49, 3))
    _52 = torch.view(_49, [_50, 3, 16, _51, int(nx1)])
    _53 = torch.contiguous(torch.permute(_52, [0, 1, 3, 4, 2]))
    return (_43, _48, _53)
