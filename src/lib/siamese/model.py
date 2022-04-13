import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from collections import OrderedDict, defaultdict

from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT
import timm


class ResNet50Conv4(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Conv4, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.features(x)
        return x


class VGG16Pool5(nn.Module):
    def __init__(self):
        super(VGG16Pool5, self).__init__()
        self.net = torchvision.models.vgg16(pretrained=True).features

    def forward(self, x):
        x = self.net(x)
        return x


class VGG16FC6(nn.Module):
    def __init__(self):
        super(VGG16FC6, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.avgpool = torchvision.models.vgg16(pretrained=True).avgpool
        self.classifier = torchvision.models.vgg16(pretrained=True).classifier[0]
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class VGG16FC7(nn.Module):
    def __init__(self):
        super(VGG16FC7, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.avgpool = torchvision.models.vgg16(pretrained=True).avgpool
        self.classifier = torchvision.models.vgg16(pretrained=True).classifier[:4]
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def load_siamese_checkpoint(name, checkpoint_file):
    if name == "resnet50":
        print('--------------------------------------------------------------')
        print('used model: zoo_resnet50')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet50(pretrained=True)
        # model.eval()
        return model

    elif name == "resnet18":
        print('--------------------------------------------------------------')
        print('used model: zoo_resnet18')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet18(pretrained=True)
        model.eval()
        return model

    elif name == "resnet34":
        print('--------------------------------------------------------------')
        print('used model: zoo_resnet34')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet34(pretrained=True)
        model.eval()
        return model

    elif name == "multigrain_resnet50":
        print('--------------------------------------------------------------')
        print('used model: multigrain_resnet50')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet50(pretrained=False)
        st = torch.load(checkpoint_file)
        state_dict = OrderedDict([
            (name[9:], v)
            for name, v in st["model_state"].items() if name.startswith("features.")
        ])
        model.fc
        model.fc = None
        model.load_state_dict(state_dict)
        # model.eval()
        return model

    elif name == "vgg":
        print('--------------------------------------------------------------')
        print('used model: VGG16')
        print('--------------------------------------------------------------')
        model = torchvision.models.vgg16(pretrained=True)
        # model.eval()
        return model

    elif name == "vgg_pool5":
        print('--------------------------------------------------------------')
        print('used model: VGG16')
        print('--------------------------------------------------------------')
        model = VGG16Pool5()
        # model.eval()
        return model

    elif name == "vgg_fc6":
        print('--------------------------------------------------------------')
        print('used model: VGG16')
        print('--------------------------------------------------------------')
        model = VGG16FC6()
        # model.eval()
        return model

    elif name == "vgg_fc7":
        print('--------------------------------------------------------------')
        print('used model: VGG16_fc7')
        print('--------------------------------------------------------------')
        model = VGG16FC7()
        # model.eval()
        return model

    elif name == "vgg_fc7":
        print('--------------------------------------------------------------')
        print('used model: VGG16')
        print('--------------------------------------------------------------')
        model = VGG16FC7()
        # model.eval()
        return model

    elif name == "resnet152":
        print('--------------------------------------------------------------')
        print('used model: ResNet152')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet152(pretrained=True)
        model.eval()
        return model

    elif name == "efficientnetb1":
        print('--------------------------------------------------------------')
        print('used model: EfficientNet-b1')
        print('--------------------------------------------------------------')
        model = EfficientNet.from_pretrained('efficientnet-b1')
        model.eval()
        return model

    elif name == "efficientnetb7":
        print('--------------------------------------------------------------')
        print('used model: EfficientNet-b7')
        print('--------------------------------------------------------------')
        model = EfficientNet.from_pretrained('efficientnet-b7')
        model.eval()
        return model

    elif name == "transformer":
        print('--------------------------------------------------------------')
        print('used model: ViT')
        print('--------------------------------------------------------------')
        model = ViT('B_16_imagenet1k', pretrained=True)
        model.eval()
        return model

    elif name == "visformer":
        print('--------------------------------------------------------------')
        print('used model: vit_large_patch16_384')
        print('--------------------------------------------------------------')
        model = timm.create_model('vit_large_patch16_384', pretrained=True)
        # model.eval()
        return model

    elif name == "resnet50_conv4":
        print('--------------------------------------------------------------')
        print('used model: resnet50_conv4')
        print('--------------------------------------------------------------')
        resnet = torchvision.models.resnet50(pretrained=True)
        model = ResNet50Conv4(resnet)
        # model.eval()
        return model

    # TODO: Train from scratch if the network weights are not available
    else:
        print('--------------------------------------------------------------')
        print('used model: resnet50')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet50(pretrained=False)
        model.eval()
        return model



class ContrastiveSiameseNetwork(nn.Module):
    def __init__(self, model, checkpoint='vgg'):
        super(ContrastiveSiameseNetwork, self).__init__()
        self.head = load_siamese_checkpoint(model, checkpoint)
        # for p in self.parameters():
        #     p.requires_grad = False
        if model == "zoo_resnet50" or model == "multigrain_resnet50" or model == "resnet152":
            self.map = True
        else:
            self.map = False
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     # nn.Linear(2048 * 16 * 16, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.2),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 256)
        # )
        #
        # self.fc2 = nn.Sequential(
        #     nn.Linear(1000, 512),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(512, 256)
        # )

        self.score = nn.PairwiseDistance(p=2)

    def forward_once(self, x):
        if self.map:
            x = self.head.conv1(x)
            x = self.head.bn1(x)
            x = self.head.relu(x)
            x = self.head.maxpool(x)

            x = self.head.layer1(x)
            x = self.head.layer2(x)
            x = self.head.layer3(x)
            x = self.head.layer4(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            output = self.flatten(x)
            # output = self.fc1(x)
        else:
            output = self.head(x)
            # output = self.fc2(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        score = self.score(output1, output2)
        return score, output1, output2


class TripletSiameseNetwork(nn.Module):
    def __init__(self, model, method, checkpoint='/cluster/yinan/isc2021/data/multigrain_joint_3B_0.5.pth'):
        super(TripletSiameseNetwork, self).__init__()
        self.head = load_siamese_checkpoint(model, checkpoint)
        self.flatten = nn.Flatten()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.method = method

        # self.fc = nn.Sequential(
        #     nn.Linear(1000, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 256)
        # )

    def gem(self, x, p=3, eps=1e-6):
        x = torch.clamp(x, eps, np.inf)
        x = x ** p
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x ** (1. / p)

    def forward_once(self, x):
        x = self.head(x)
        if self.method == 'center_extraction' or self.method == 'warp_extraction':
            x = F.normalize(x)
        elif self.method == 'max_pool':
            x = F.adaptive_max_pool2d(x, (1, 1))
            x = self.flatten(x)
        elif self.method == 'sum_pool':
            x = x.size()[2] * x.size()[3] * F.adaptive_avg_pool2d(x, (1, 1))
            x = self.flatten(x)
        elif self.method == 'sum_pool_2x2':
            x = x.size()[2] * x.size()[3] * 0.25 * F.adaptive_avg_pool2d(x, (2, 2))
            x = self.flatten(x)
        elif self.method == 'feature_map':
            pass
        else:
            x = self.gem(x)
            x = self.flatten(x)
        return x

    def forward(self, input1, input2, input3):
        # score_positive_1 = 1 - (torch.sum(self.cos(out1, out2), axis=(1, 2)) / (out1.shape[2] * out1.shape[3]))
        # score_negative_1 = 1 - (torch.sum(self.cos(out1, out3), axis=(1, 2)) / (out1.shape[2] * out1.shape[3]))
        x1 = self.forward_once(input1)
        x2 = self.forward_once(input2)
        x3 = self.forward_once(input3)
        score_positive = 1 - self.cos(x1, x2)
        score_negative = 1 - self.cos(x1, x3)

        return score_positive, score_negative


class TripletSiameseNetwork_custom(nn.Module):
    def __init__(self, model, checkpoint='/cluster/yinan/isc2021/data/multigrain_joint_3B_0.5.pth'):
        super(TripletSiameseNetwork_custom, self).__init__()
        self.model = model
        self.head = load_siamese_checkpoint(model, checkpoint)
        self.flatten = nn.Flatten()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.clamp(x, eps, np.inf)
        x = x ** p
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x ** (1. / p)

    def forward_once(self, x):
        if self.model == 'resnet50':
            x = self.head.conv1(x)
            x = self.head.bn1(x)
            x = self.head.relu(x)
            x = self.head.maxpool(x)

            x1 = self.head.layer1(x)
            x2 = self.head.layer2(x1)
            x3 = self.head.layer3(x2)
            x4 = self.head.layer4(x3)
            x1 = self.gem(x1)
            x1 = self.flatten(x1)
            x2 = self.gem(x2)
            x2 = self.flatten(x2)
            x3 = self.gem(x3)
            x3 = self.flatten(x3)
            x4 = self.gem(x4)
            x4 = self.flatten(x4)
            return x1, x2, x3, x4

        elif self.model == 'vgg_fc7':
            '''relu1_2'''
            x1 = self.head.features[:3](x)
            '''relu2_2'''
            x2 = self.head.features[4:9](x1)
            '''relu3_3'''
            x3 = self.head.features[9:16](x2)
            '''relu4_3'''
            x4 = self.head.features[16:23](x3)
            '''linear classifier'''
            x_pool5 = self.head.features[23:](x4)
            x5 = self.head.avgpool(x_pool5)
            x5 = self.head.classifier[0](x5)
            x5 = self.head.classifier[1:](x5)
            x5 = F.normalize(x5)
            x6 = x_pool5.size()[2] * x_pool5.size()[3] * 0.25 * F.adaptive_avg_pool2d(x_pool5, (2, 2))
            x6 = self.flatten(x6)
            x6 = F.normalize(x6)


            x1 = F.adaptive_max_pool2d(x1, (1, 1))
            x1 = self.flatten(x1)
            x1 = F.normalize(x1)
            x2 = F.adaptive_max_pool2d(x2, (1, 1))
            x2 = self.flatten(x2)
            x2 = F.normalize(x2)
            x3 = F.adaptive_max_pool2d(x3, (1, 1))
            x3 = self.flatten(x3)
            x3 = F.normalize(x3)
            x4 = F.adaptive_max_pool2d(x4, (1, 1))
            x4 = self.flatten(x4)
            x4 = F.normalize(x4)
            return x1, x2, x3, x4, x5, x6

    def forward(self, input1, input2, input3):
        if self.model == 'resnet50':
            x1_1, x1_2, x1_3, x1_4 = self.forward_once(input1)
            x2_1, x2_2, x2_3, x2_4 = self.forward_once(input2)
            x3_1, x3_2, x3_3, x3_4 = self.forward_once(input3)

            return x1_1, x1_2, x1_3, x1_4, x2_1, x2_2, x2_3, x2_4, x3_1, x3_2, x3_3, x3_4

        elif self.model == 'vgg' or self.model == 'vgg_fc7':
            x1_1, x1_2, x1_3, x1_4, x1_5, x1_6 = self.forward_once(input1)
            x2_1, x2_2, x2_3, x2_4, x2_5, x2_6 = self.forward_once(input2)
            x3_1, x3_2, x3_3, x3_4, x3_5, x3_6 = self.forward_once(input3)

            return x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x3_1, x3_2, x3_3, x3_4, x3_5, x3_6





