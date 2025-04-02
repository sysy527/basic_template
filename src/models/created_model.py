from src.models.layers import *

MODEL_CLASSES = {"VGG16" : VGG16, 
                 "VGG19" : VGG19,
                 "ResNet18": Model_ResNet().resnet18,
                 "ResNet34": Model_ResNet().resnet34,
                 "ResNet50": Model_ResNet().resnet50,
                 "ResNet101": Model_ResNet().resnet101,
                 "ResNet152": Model_ResNet().resnet152,
                 "Plain34_wRes":Model_PlainNet().plain34,
                 "UNet" : UNet,
                 "UNet_preEncoder" : UNet_preEncoder,
                 "Generator" : Generator,
                 "Discriminator" : Discriminator,
                 "CycleGan" : CycleGAN
                 }