import torch
import torch.nn as nn
from torchvision import models


class VGG16_base(nn.Module):
    r"""
    The base class of VGG16. It downloads the pretrained weight by torchvision API, and maintain the layers needed for
    deep graph matching models.
    """
    def __init__(self, batch_norm=True, final_layers=False):
        super(VGG16_base, self).__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone(batch_norm)
        if not final_layers: self.final_layers = None
        self.backbone_params = list(self.parameters())

    def forward(self, *input):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def get_backbone(batch_norm):
        """
        Get pretrained VGG16 models for feature extraction.

        :return: feature sequence
        """
        if batch_norm:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16(pretrained=True)

        conv_layers = nn.Sequential(*list(model.features.children()))

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
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)
        final_layers = nn.Sequential(*conv_list, nn.AdaptiveMaxPool2d((1, 1), return_indices=False)) # this final layer follows Rolink et al. ECCV20

        return node_layers, edge_layers, final_layers
    

class VGG16_bn_final(VGG16_base):
    r"""
    VGG16 with batch normalization and final layers.
    """
    def __init__(self):
        super(VGG16_bn_final, self).__init__(True, True)


class VGG16_bn(VGG16_base):
    r"""
    VGG16 with batch normalization, without final layers.
    """
    def __init__(self):
        super(VGG16_bn, self).__init__(True, False)


class VGG16_final(VGG16_base):
    r"""
    VGG16 without batch normalization, with final layers.
    """
    def __init__(self):
        super(VGG16_final, self).__init__(False, True)


class VGG16(VGG16_base):
    r"""
    VGG16 without batch normalization or final layers.
    """
    def __init__(self):
        super(VGG16, self).__init__(False, False)


class NoBackbone(nn.Module):
    r"""
    A model with no CNN backbone for non-image data.
    """
    def __init__(self, *args, **kwargs):
        super(NoBackbone, self).__init__()
        self.node_layers, self.edge_layers = None, None

    def forward(self, *input):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device
