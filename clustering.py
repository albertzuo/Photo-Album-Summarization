import torch
import torchvision
from torchvision import models
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
from vgg import Vgg16
import utils

imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

pretrained_model = torchvision.models.vgg16(pretrained=True)
modified_pretrained = torch.nn.Sequential(*list(pretrained_model.features.children())[:-1])

def image_loader(image_path):
    """load image, returns cuda tensor"""
    image = utils.load_image(image_path)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    # return image.cuda()  #assumes that you're using GPU
    return image

basePath = './Set1/image'
fv = []
for i in range(59):
    path = basePath + str(i+1) + '.jpg'
    img1 = image_loader(path)
    fv.append(modified_pretrained(img1))


# v = Vgg16()
