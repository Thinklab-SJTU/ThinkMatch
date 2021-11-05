import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.common.tensor import Tensor
import mindspore_hub as mshub
from mindspore import Parameter
import mindspore.context as context

context.set_context(device_target="GPU")

class FinalMaxPool(nn.Cell):
    """
        This function equals to:
        nn.AdaptiveMaxPool2d((1, 1), return_indices=False)
    """
    def __init__(self):
        super(FinalMaxPool, self).__init__()

    def construct(self, input):
        return P.ReduceMax()(input, (2,3))

class VGG16_base(nn.Cell):
    def __init__(self, batch_norm=True, final_layers=False):
        super(VGG16_base, self).__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone(batch_norm)
        if not final_layers: self.final_layers = None
        self.backbone_params = list(self.trainable_params())

    def construct(self, *input):
        raise NotImplementedError

    @staticmethod
    def get_backbone(batch_norm):
        """
        Get pretrained VGG16 models for feature extraction.
        :return: feature sequence
        """
        if batch_norm:
            model = mshub.load("mindspore/ascend/1.2/vgg16_v1.2_cifar10", num_classes=10)
        else:
            print("Warning: Mindspore doesn't support vgg16 without BN. Using vgg16 with BN.")
            model = mshub.load("mindspore/ascend/1.2/vgg16_v1.2_cifar10", num_classes=10)

        conv_layers = nn.SequentialCell()
        for layer in model.cells():
            if isinstance(layer, nn.SequentialCell):
                for sublayer in layer:
                    conv_layers.append(sublayer)
            else:
                conv_layers.append(layer)

        for l in conv_layers:
            if isinstance(l,type(conv_layers[0])):
                # conv layer
                l.has_bias = True
                biasname = l.trainable_params()[0].name[:-6] + 'bias'
                l.bias = Parameter(np.zeros(l.out_channels), name=biasname)


        conv_list = node_list = edge_list = []

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(conv_layers):
            if isinstance(module, nn.Conv2d):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]

            #if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
            if cnt_m == 4 and cnt_r == 3 and isinstance(module, nn.Conv2d):
                node_list = conv_list
                conv_list = []
            #elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
            elif cnt_m == 5 and cnt_r == 2 and isinstance(module, nn.Conv2d):
                edge_list = conv_list
                conv_list = []

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.SequentialCell([*node_list])
        edge_layers = nn.SequentialCell([*edge_list])
        final_layers = nn.SequentialCell([*conv_list, FinalMaxPool()]) # this final layer follows Rolink et al. ECCV20

        return node_layers, edge_layers, final_layers
    

class VGG16_bn_final(VGG16_base):
    def __init__(self):
        super(VGG16_bn_final, self).__init__(True, True)


class VGG16_bn(VGG16_base):
    def __init__(self):
        super(VGG16_bn, self).__init__(True, False)


class VGG16_final(VGG16_base):
    def __init__(self):
        super(VGG16_final, self).__init__(False, True)


class VGG16(VGG16_base):
    def __init__(self):
        super(VGG16, self).__init__(False, False)


class NoBackbone(nn.Cell):
    def __init__(self, *args, **kwargs):
        super(NoBackbone, self).__init__()
        self.node_layers, self.edge_layers = None, None

    def construct(self, *input):
        raise NotImplementedError
