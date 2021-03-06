import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pdb


class Img2Vec():

    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img) == list:
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name == 'alexnet':
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name == 'alexnet':
                    return my_embedding.numpy()[:, :]
                else:
                    print(my_embedding.numpy()[:, :, 0, 0].shape)
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

            if self.model_name == 'alexnet' or self.model_name == 'vgg-16' \
                    or self.model_name == 'inception-v3' or self.model_name == 'googlenet' \
                    or 'densenet' in self.model_name:
                my_embedding = torch.zeros(1, self.layer_output_size)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name == 'alexnet' or self.model_name == 'vgg-16'\
                        or self.model_name == 'inception-v3' or self.model_name == 'googlenet'\
                        or 'densenet' in self.model_name:
                    return my_embedding.numpy()[0, :]
                else:
                    return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'resnet-34':
            model = models.resnet34(pretrained=True)
        elif model_name == 'resnet-50':
            model = models.resnet50(pretrained=True)
        elif model_name == 'resnet-101':
            model = models.resnet101(pretrained=True)
        elif model_name == 'resnet-152':
            model = models.resnet152(pretrained=True)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif model_name == 'vgg-16':
            model=models.vgg16(pretrained=True)
        elif model_name == 'inception-v3':
            model=models.inception_v3(pretrained=True)
        elif model_name == 'googlenet':
            model=models.googlenet(pretrained=True)
        elif model_name == 'densenet121':
            model=models.densenet121(pretrained=True)
        elif model_name == 'densenet169':
            model=models.densenet169(pretrained=True)

        if 'resnet' in model_name:
            if layer == 'default':
                layer = model._modules.get('avgpool')
                if model_name == 'resnet-18' or model_name == 'resnet-34':
                    self.layer_output_size = 512
                else:
                    self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'vgg-16':
            if layer == 'default':
                # pdb.set_trace()
                layer = model.classifier[-3]
                self.layer_output_size = 4096
            else:
                layer=model.classifier[-layer]

            return model, layer

        elif model_name == 'inception-v3':
            if layer == 'default':
                layer = model.fc
                self.layer_output_size = 1000
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'googlenet':
            if layer == 'default':
                layer = model.fc
                self.layer_output_size = 1000
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif 'densenet' in model_name:
            if layer == 'default':
                layer = model.classifier
                self.layer_output_size = 1000
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)