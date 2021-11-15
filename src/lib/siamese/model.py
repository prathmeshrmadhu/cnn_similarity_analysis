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


def load_siamese_checkpoint(name, checkpoint_file):
    if name == "resnet50":
        print('--------------------------------------------------------------')
        print('used model: zoo_resnet50')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet50(pretrained=True)
        model.eval()
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
        model.eval()
        return model

    elif name == "vgg":
        print('--------------------------------------------------------------')
        print('used model: VGG16')
        print('--------------------------------------------------------------')
        model = torchvision.models.vgg16(pretrained=True)
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
        model.eval()
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
    def __init__(self, model, checkpoint='/cluster/yinan/isc2021/data/multigrain_joint_3B_0.5.pth'):
        super(TripletSiameseNetwork, self).__init__()
        self.head = load_siamese_checkpoint(model, checkpoint)
        # for p in self.parameters():
        #     p.requires_grad = False
        if model == "zoo_resnet50" or model == "multigrain_resnet50":
            self.map = True
        else:
            self.map = False
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            # nn.Linear(2048 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

        self.fc2 = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, 128)
        )

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.clamp(x, eps, np.inf)
        x = x ** p
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x ** (1. / p)

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
            x = self.gem(x)
            x = self.flatten(x)
            # x = self.fc1(x)
        else:
            x = self.head(x)
            # x = self.fc2(x)
        return x

    def calculate_distance(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        score = 1 - self.cos(output1, output2)
        return score

    def forward(self, input1, input2, input3):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        out3 = self.forward_once(input3)
        score_positive = 1 - (torch.sum(self.cos(out1, out2), axis=(1, 2)) / (out1.shape[2] * out1.shape[3]))
        score_negative = 1 - (torch.sum(self.cos(out1, out3), axis=(1, 2)) / (out1.shape[2] * out1.shape[3]))
        # score_positive = 1 - self.cos(out1, out3)
        # score_negative = 1 - self.cos(out1, out3)
        return score_positive, score_negative




