import paddle
import paddle.nn as nn
from paddle.vision import models
from src.utils_pdl.model_sl import load_model

class VGG16_base(nn.Layer):
    def __init__(self, batch_norm=True):
        super(VGG16_base, self).__init__()
        self.node_layers, self.edge_layers = self.get_backbone(batch_norm)

    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    def get_backbone(batch_norm):
        """
        Get pretrained VGG16 models for feature extraction.
        :return: feature sequence
        """
        model = models.vgg16(pretrained=False, batch_norm=batch_norm)
        load_model(model,'src/utils_pdl/vgg16_bn.pdparams')

        conv_list = node_list = edge_list = []

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(model.sublayers()):
            if( layer == 0 ): continue 
            if isinstance(module, nn.Conv2D):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2D):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]

            #if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
            if cnt_m == 4 and cnt_r == 3 and isinstance(module, nn.Conv2D):
                node_list = conv_list
                conv_list = []
            #elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
            elif cnt_m == 5 and cnt_r == 2 and isinstance(module, nn.Conv2D):
                edge_list = conv_list
                conv_list = []
                #break

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)

        return node_layers, edge_layers
    

class VGG16_bn(VGG16_base):
    def __init__(self):
        super(VGG16_bn, self).__init__(True)


class VGG16(VGG16_base):
    def __init__(self):
        super(VGG16, self).__init__(False)


class NoBackbone(nn.Layer):
    def __init__(self, batch_norm=True):
        super(NoBackbone, self).__init__()
        self.node_layers, self.edge_layers = None, None

    def forward(self, *input):
        raise NotImplementedError


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
