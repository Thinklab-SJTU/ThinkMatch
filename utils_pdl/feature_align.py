import paddle
from paddle import Tensor
from .pdl_device_trans import place2int 


def feature_align(raw_feature: Tensor, P: Tensor, ns_t: Tensor, ori_size: tuple, device=None):
    """
    Perform feature align from the raw feature map.
    :param raw_feature: raw feature map
    :param P: point set containing point coordinates
    :param ns_t: number of exact points in the point set
    :param ori_size: size of the original image
    :param device: device. If not specified, it will be the same as the input
    :return: F
    """
    if device is None:
        device = raw_feature.place

    batch_num = raw_feature.shape[0]
    channel_num = raw_feature.shape[1]
    n_max = P.shape[1]
    #n_max = 0
    #for idx in range(batch_num):
    #    n_max = max(ns_t[idx], n_max)

    ori_size = paddle.to_tensor(ori_size, dtype='float32', place=device)
    F = paddle.zeros([batch_num, channel_num, n_max], dtype='float32').cuda(place2int(device))
    for idx, feature in enumerate(raw_feature):
        n = ns_t[idx]
        feat_size = paddle.to_tensor(feature.shape[1:3], dtype='float32', place=device)
        _P = P[idx, 0:n]
        interp_2d(feature, _P, ori_size, feat_size, out=F[idx, :, 0:n])
    return F


def interp_2d(z: Tensor, P: Tensor, ori_size: Tensor, feat_size: Tensor, out=None, device=None):
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
    if device is None:
        device = z.place

    step = ori_size / feat_size
    if out is None:
        out = paddle.zeros([z.shape[0], P.shape[0]], dtype='float32').cuda(place2int(device))
    for i, p in enumerate(P):
        p = (p - step / 2) / ori_size * feat_size
        out[:, i] = bilinear_interpolate_paddle(z, p[0], p[1])

    return out


def bilinear_interpolate_paddle(im: Tensor, x: Tensor, y: Tensor, out=None, device=None):
    """
    Bi-linear interpolate 3d feature map im to 2d plane (x, y)
    :param im: 3d feature map
    :param x: x coordinate
    :param y: y coordinate
    :param out: optional output tensor
    :param device: device. If not specified, it will be the same as the input
    :return: interpolated feature vector
    """
    if device is None:
        device = im.place
    x = paddle.to_tensor(x, dtype='float32', place=device)#paddle.tensor(x, dtype=paddle.float32, device=device)
    y = paddle.to_tensor(y, dtype='float32', place=device)#paddle.tensor(y, dtype=paddle.float32, device=device)

    x0 = paddle.floor(x)
    x1 = x0 + 1
    y0 = paddle.floor(y)
    y1 = y0 + 1

    x0 = paddle.clip(x0, 0, im.shape[2] - 1)
    x1 = paddle.clip(x1, 0, im.shape[2] - 1)
    y0 = paddle.clip(y0, 0, im.shape[1] - 1)
    y1 = paddle.clip(y1, 0, im.shape[1] - 1)

    x0 = paddle.to_tensor(x0, dtype='int32', place=device)#paddle.tensor(x0, dtype=paddle.int32, device=device)
    x1 = paddle.to_tensor(x1, dtype='int32', place=device)#paddle.tensor(x1, dtype=paddle.int32, device=device)
    y0 = paddle.to_tensor(y0, dtype='int32', place=device)#paddle.tensor(y0, dtype=paddle.int32, device=device)
    y1 = paddle.to_tensor(y1, dtype='int32', place=device)#paddle.tensor(y1, dtype=paddle.int32, device=device)

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

    x0 = paddle.to_tensor(x0, dtype='float32', place=device)#paddle.tensor(x0, dtype=paddle.float32, device=device)
    x1 = paddle.to_tensor(x1, dtype='float32', place=device)#paddle.tensor(x1, dtype=paddle.float32, device=device)
    y0 = paddle.to_tensor(y0, dtype='float32', place=device)#paddle.tensor(y0, dtype=paddle.float32, device=device)
    y1 = paddle.to_tensor(y1, dtype='float32', place=device)#paddle.tensor(y1, dtype=paddle.float32, device=device)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out