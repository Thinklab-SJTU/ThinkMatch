import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor, Parameter
import mindspore.ops.operations as P
#from mindspore.common.tensor import Tensor
import mindspore.context as context

context.set_context(device_target="GPU")

class SpecialLRN(nn.Cell):

    def __init__(self):
        super(SpecialLRN, self).__init__()

    def construct(self, X):
        assert len(X.shape) == 4

        # Y = np.zeros_like(X)
        # for a in range(X.shape[0]):
        #     for c in range(X.shape[2]):
        #         for d in range(X.shape[3]):
        #             l2 = pow(P.ReduceSum()(X[a,:,c,d]**2), 1/2)
        #             Y[a, :, c, d] = X[a, :, c, d] / l2

        l2 = P.ReduceSum(keep_dims=True)(X**2,1)
        Y = X / (l2**(1/2))

        return Y

if __name__ == '__main__':
    import torch
    lrn_t = torch.nn.LocalResponseNorm(6, alpha=6, beta=0.5, k=0)
    lrn_m = SpecialLRN()
    input_t = torch.rand(3,3,3,3)
    input_m = Tensor(input_t.detach().cpu().numpy())
    output_t = lrn_t(input_t)
    output_m = lrn_m(input_m)
    print(input_t)
    print(output_t)
    print(output_m)
