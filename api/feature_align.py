import sys
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as OP
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.common.tensor import Tensor
import mindspore.context as context

context.set_context(device_target="GPU")

class feature_align(nn.Cell):
    def __init__(self):
        super(feature_align, self).__init__()
        self.exd = OP.ExpandDims()
        self.cc0 = OP.Concat(0)
        self.cc1 = OP.Concat(1)
        self.zrs = OP.Zeros()
        self.flr = OP.Floor()

    def construct(self, raw_feature: Tensor, P: Tensor, ns_t: Tensor, ori_size: tuple, device=None):
        """
        Perform feature align from the raw feature map.
        :param raw_feature: raw feature map
        :param P: point set containing point coordinates
        :param ns_t: number of exact points in the point set
        :param ori_size: size of the original image
        :param device: device. If not specified, it will be the same as the input
        :return: F
        """

        batch_num = raw_feature.shape[0]
        channel_num = raw_feature.shape[1]
        n_max = P.shape[1]

        ori_size = Tensor(ori_size, dtype=mindspore.float32)
        F = list()
        for idx, feature in enumerate(raw_feature):
            n = ns_t[idx]
            n = n.asnumpy().item()
            feat_size = Tensor(feature.shape[1:3])
            _P = P[idx, 0:n]
            f = self.interp_2d(feature, _P, ori_size, feat_size)
            if n_max > n:
                f = self.cc1((f, self.zrs((channel_num, n_max-n), mindspore.float32)))
            f = self.exd(f, 0)
            F.append(f)
        F1 = self.cc0(F)

        return F1


    def interp_2d(self, z: Tensor, P: Tensor, ori_size: Tensor, feat_size: Tensor, out=None, device=None):
        """
        Interpolate in 2d grid space. z can be 3-dimensional where the 3rd dimension is feature vector.
        :param z: 2d/3d feature map
        :param P: input point set
        :param ori_size: size of the original image
        :param feat_size: size of the feature map
        :param out: optional output tensor
        :param device: device. If not specified, it will be the same as the input
        :return: F
        """

        step = ori_size / feat_size

        out = list()

        for i, p in enumerate(P):
            p = (p - step / 2) / ori_size * feat_size
            ex = self.exd(self.bilinear_interpolate_torch(z, p[0], p[1]), 1)
            out.append(ex)

        out1 = self.cc1(out)
        return out1


    def bilinear_interpolate_torch(self, im: Tensor, x: Tensor, y: Tensor, out=None, device=None):
        """
        Bi-linear interpolate 3d feature map im to 2d plane (x, y)
        :param im: 3d feature map
        :param x: x coordinate
        :param y: y coordinate
        :param out: optional output tensor
        :param device: device. If not specified, it will be the same as the input
        :return: interpolated feature vector
        """

        x = x.astype(mindspore.float32)
        y = y.astype(mindspore.float32)
        x0 = self.flr(x)
        x1 = x0 + 1
        y0 = self.flr(y)
        y1 = y0 + 1

        x0 = ops.composite.clip_by_value(x0, 0, im.shape[2] - 1)
        x1 = ops.composite.clip_by_value(x1, 0, im.shape[2] - 1)
        y0 = ops.composite.clip_by_value(y0, 0, im.shape[1] - 1)
        y1 = ops.composite.clip_by_value(y1, 0, im.shape[1] - 1)

        x0 = x0.astype(mindspore.int32)
        x1 = x1.astype(mindspore.int32)
        y0 = y0.astype(mindspore.int32)
        y1 = y1.astype(mindspore.int32)

        Ia = im[:, y0, x0]
        Ib = im[:, y1, x0]
        Ic = im[:, y0, x1]
        Id = im[:, y1, x1]

        # to perform nearest neighbor interpolation if out of bounds
        if x0 == x1:
            if x0 == 0:
                x0 -= 1
            else:
                x1 += 1
        if y0 == y1:
            if y0 == 0:
                y0 -= 1
            else:
                y1 += 1

        x0 = x0.astype(mindspore.float32)
        x1 = x1.astype(mindspore.float32)
        y0 = y0.astype(mindspore.float32)
        y1 = y1.astype(mindspore.float32)

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        out = Ia * wa + Ib * wb + Ic * wc + Id * wd
        return out
