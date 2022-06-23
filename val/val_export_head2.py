import torch
import torch.nn as nn
import numpy as np

class Detect(nn.Module):
    stride = torch.tensor([8.0,16.0,32.0])  # strides computed during build
    export_cat = True  # onnx export cat output

    def __init__(self, nc=1, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = 1  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        self.no = nc + 5 + 10  # number of outputs per anchor

        self.nl = 3  # number of detection layers
        self.na = 3  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid

        self.anchor_grid = [torch.zeros(1)] * self.nl  # init grid

        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        # self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        if self.export_cat:
            for i in range(self.nl):
                # x[i] = self.m[i](x[i])  # conv
                bs, number, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

                # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                for j in range(number//16):
                # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                #     feature = x[i][:,16*j:16*(j+1),:,:].view(bs, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()
                    feature = x[i][:,16*j:16*(j+1),:,:]

                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        # self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                        self.grid[i], self.anchor_grid[i] = self._make_grid_new(nx, ny,i)

                    # y = torch.full_like(feature, 0)
                    # feature_temp1 = torch.cat((feature[ :, 5:15, :,: ],feature[:, 15:15 + self.nc, :,: ].sigmoid()),1)
                    # y = y + torch.cat((feature_temp1,feature[ :, 0:5, :,: ].sigmoid()),dim=1)


                    y = torch.full_like(feature, 0)
                    y = y + torch.cat((feature[ :, 0:5, :,:].sigmoid(),
                                       torch.cat((feature[ :, 5:15, :,: ], feature[ :, 15:15 + self.nc, :, :].sigmoid()), 1)),
                                      1)

                    y = torch.permute(y, [0, 3, 2, 1])
                    # y = y + torch.cat((feature[ :, :, :, 0:5].sigmoid(), torch.cat((feature[ :, :, :, 5:15], feature[:, :, :, 15:15+self.nc].sigmoid()), 3)), 3)
                    # box_xy = (y[:, :, :, 0:2] * 2. - 0.5 + self.grid[i].to(feature.device)) * self.stride[i] # xy
                    box_xy = (y[:, :, :, 0:2] * 2.+(- 0.5) + self.grid[i][j:j+1,:,:,:].to(feature.device)) * self.stride[i] # xy
                    # box_wh = (y[:, :, :, :, 2:4] * 2) ** 2 * self.anchor_grid[i] # wh
                    box_wh = (y[:, :, :, 2:4] * 2) *(y[ :, :, :, 2:4] * 2) * self.anchor_grid[i][j:j+1,:,:,:]  # wh
                    # box_conf = torch.cat((box_xy, torch.cat((box_wh, y[:, :, :, :, 4:5]), 4)), 4)
                    landm1 = y[ :, :, :, 5:7] * self.anchor_grid[i][j:j+1,:,:,:] + self.grid[i][j:j+1,:,:,:].to(feature.device) * self.stride[i]  # landmark x1 y1
                    landm2 = y[ :, :, :, 7:9] * self.anchor_grid[i][j:j+1,:,:,:] + self.grid[i][j:j+1,:,:,:].to(feature.device) * self.stride[i]  # landmark x2 y2
                    landm3 = y[ :, :, :, 9:11] * self.anchor_grid[i][j:j+1,:,:,:] + self.grid[i][j:j+1,:,:,:].to(feature.device) * self.stride[i]  # landmark x3 y3
                    landm4 = y[ :, :, :, 11:13] * self.anchor_grid[i][j:j+1,:,:,:] + self.grid[i][j:j+1,:,:,:].to(feature.device) * self.stride[i]  # landmark x4 y4
                    landm5 = y[ :, :, :, 13:15] * self.anchor_grid[i][j:j+1,:,:,:] + self.grid[i][j:j+1,:,:,:].to(feature.device) * self.stride[i]  # landmark x5 y5
                    # landm = torch.cat((landm1, torch.cat((landm2, torch.cat((landm3, torch.cat((landm4, landm5), 4)), 4)), 4)), 4)
                    # y = torch.cat((box_conf, torch.cat((landm, y[:, :, :, :, 15:15+self.nc]), 4)), 4)

                    y1 = y[:, :, :, 4:5]
                    y2 = y[:, :, :, 15:15 + self.nc]
                    box_xy = torch.permute(box_xy,[0,3,2,1])
                    box_wh = torch.permute(box_wh,[0,3,2,1])
                    y1 = torch.permute(y1,[0,3,2,1])
                    y2 = torch.permute(y2, [0, 3, 2, 1])
                    landm1 = torch.permute(landm1,[0,3,2,1])
                    landm2 = torch.permute(landm2,[0,3,2,1])
                    landm3 = torch.permute(landm3,[0,3,2,1])
                    landm4 = torch.permute(landm4,[0,3,2,1])
                    landm5 = torch.permute(landm5,[0,3,2,1])

                    y = torch.cat([box_xy,box_wh,y1,landm1,landm2,landm3,landm4,landm5,y2],dim=1)

                    y = torch.permute(y,[0,3,2,1]).contiguous()
                    # y = torch.cat([box_xy, box_wh, y[ :, :, :, 4:5], landm1, landm2, landm3, landm4, landm5, y[ :, :, :, 15:15+self.nc]], -1)

                    # z.append(y.view(1,bs, -1, self.no))
                    z.append(y.view(1, -1, bs, self.no))
            # return torch.cat(z, 1)
            # z  = torch.permute(z,[0,2,1,3])
            # for z_data in z
            z = torch.cat(z,dim=1)
            z = torch.permute(z,[0,2,1,3])
            # return torch.cat(z, 2)
            return z

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = torch.full_like(x[i], 0)
                class_range = list(range(5)) + list(range(15, 15 + self.nc))
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 5:15] = x[i][..., 5:15]
                # y = x[i].sigmoid()

                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                # y[..., 5:15] = y[..., 5:15] * 8 - 4
                y[..., 5:7] = y[..., 5:7] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x1 y1
                y[..., 7:9] = y[..., 7:9] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x2 y2
                y[..., 9:11] = y[..., 9:11] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x3 y3
                y[..., 11:13] = y[..., 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x4 y4
                y[..., 13:15] = y[..., 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x5 y5

                # y[..., 5:7] = (y[..., 5:7] * 2 -1) * self.anchor_grid[i]  # landmark x1 y1
                # y[..., 7:9] = (y[..., 7:9] * 2 -1) * self.anchor_grid[i]  # landmark x2 y2
                # y[..., 9:11] = (y[..., 9:11] * 2 -1) * self.anchor_grid[i]  # landmark x3 y3
                # y[..., 11:13] = (y[..., 11:13] * 2 -1) * self.anchor_grid[i]  # landmark x4 y4
                # y[..., 13:15] = (y[..., 13:15] * 2 -1) * self.anchor_grid[i]  # landmark x5 y5

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _make_grid_new(self,nx=20, ny=20,i=0):
        d = self.anchors[i].device
        if '1.10.0' in torch.__version__: # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((self.na, ny, nx, 2)).float()

        # anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view(( self.na, 1, 1, 2)).expand(( self.na, ny, nx, 2)).float()
        return grid, anchor_grid


img = [torch.ones(1, 48,80,80)*0.5163,torch.ones(1, 48,40,40)*0.756,torch.ones(1, 48,20,20)*12.36]


anchors = [[4, 5, 8, 10, 13, 16], [23, 29, 43, 55, 73, 105], [146, 217, 231, 300, 335, 433]]
ch = [64, 64, 64]
model = Detect(anchors=anchors,ch=ch)

y = model(img)

np.save("y2.npy",y.detach().cpu().numpy())

# print(" y ",y.shape)
# print(" y ",y.detach().cpu().numpy()[0,0,0,:])