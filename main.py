import numpy as np
import argparse, pdb
import matplotlib.pyplot as plt

from PIL import Image
from utils import get_model_distance

parser = argparse.ArgumentParser(description='Similarity finder and analyzer')
parser.add_argument('--img1', type=str, required=True, help='path to input image 1')
parser.add_argument('--img2', type=str, required=True, help='path to input image 2')
parser.add_argument('--size', type=int, default=256, help='image size for similarity comparison')
args = parser.parse_args()

## Loading Image and pre-processing
image1 = Image.open(args.img1).convert('RGB')
image2 = Image.open(args.img2).convert('RGB')

image1 = image1.resize((args.size, args.size), Image.ANTIALIAS)
image2 = image2.resize((args.size, args.size), Image.ANTIALIAS)

image1_array = np.array(image1) / 255.0
image2_array = np.array(image2) / 255.0

pixel_distance = np.linalg.norm(image1_array - image2_array)
alexnet_distance = get_model_distance(image1, image2, model_name='alexnet')
vgg16_distance = get_model_distance(image1, image2, model_name='vgg-16')
inceptionv3_distance = get_model_distance(image1, image2, model_name='inception-v3')
googlenet_distance = get_model_distance(image1, image2, model_name='googlenet')
densenet121_distance = get_model_distance(image1, image2, model_name='densenet121')
densenet169_distance = get_model_distance(image1, image2, model_name='densenet169')
resnet18_distance = get_model_distance(image1, image2, model_name='resnet-18')
resnet34_distance = get_model_distance(image1, image2, model_name='resnet-34')
resnet50_distance = get_model_distance(image1, image2, model_name='resnet-50')
resnet101_distance = get_model_distance(image1, image2, model_name='resnet-101')
resnet152_distance = get_model_distance(image1, image2, model_name='resnet-152')


print("Average pixel-wise distance : {}".format(pixel_distance))
print("AlexNet Cosine Distance: {}".format(alexnet_distance))
print("VGG16 Cosine Distance: {}".format(vgg16_distance))
print("inception-v3 Cosine Distance: {}".format(inceptionv3_distance))
print("googlenet Cosine Distance: {}".format(googlenet_distance))
print("densenet121 Cosine Distance: {}".format(densenet121_distance))
print("densenet169 Cosine Distance: {}".format(densenet169_distance))
print("Resnet18 Cosine Distance: {}".format(resnet18_distance))
print("Resnet34 Cosine Distance: {}".format(resnet34_distance))
print("Resnet50 Cosine Distance: {}".format(resnet50_distance))
print("Resnet101 Cosine Distance: {}".format(resnet101_distance))
print("Resnet152 Cosine Distance: {}".format(resnet152_distance))


plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.imshow(image1)
plt.title('Image 1')
plt.axis('off')
plt.subplot(122)
plt.imshow(image2)
plt.title('Image 2')
plt.axis('off')
plt.show()

