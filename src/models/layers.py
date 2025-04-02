import torch
import torch.nn as nn
from torch.nn import init


class VGG16(nn.Module):
    def __init__(self, nch_in, nch_out):  # task - seg
        super(VGG16, self).__init__()
        self.nch_in = nch_in
        self.nch_out = nch_out

        # Convolutional layers                            
        self.features = nn.Sequential(                                          # [1,  32, 32] input size
            nn.Conv2d(self.nch_in, 64, kernel_size=3, padding=1), nn.ReLU(),    # [64, 32, 32] # bn 추가
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2, stride=2),                              # [64, 16, 16]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),            # [128, 16, 16]
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # [128, 8, 8]
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),           # [256, 8, 8]
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # [256, 4, 4]
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),           # [512, 4, 4]
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),   
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # [512, 2, 2]
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),           # [512, 2, 2]
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                               # [512, 1, 1] -> deconv
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), nn.ReLU(), nn.Dropout(),                      # [4096]
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),                     # [4096]
            nn.Linear(4096, self.nch_out)                                       # [10]
        )
        
        #self.segmenter = conv2dTranspose unpooling 

    def forward(self, x):
        x = self.features(x)        # Convolutional layers
        #self.segmenter
        x = x.view(x.size(0), -1)   # Flatten the tensor
        x = self.classifier(x)      # Fully connected layers

        return x
    
class VGG19(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(VGG19, self).__init__()
        self.nch_in = nch_in
        self.nch_out = nch_out

        # Convolutional layers                            
        self.features = nn.Sequential(                                          
            nn.Conv2d(self.nch_in, 64, kernel_size=3, padding=1), nn.ReLU(),    
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2, stride=2),                              
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),          
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                             
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),          
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                              
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),         
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),   
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                              
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),           
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                          
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(), nn.Dropout(),                     
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),              
            nn.Linear(4096, self.nch_out)                       
        )

    def forward(self, x):
        x = self.features(x)        # Convolutional layers
        x = x.view(x.size(0), -1)   # Flatten the tensor
        x = self.classifier(x)      # Fully connected layers

        return x

class BasicBlock(nn.Module):
    expansion_factor = 1
    def __init__(self, nch_in, nch_out, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(nch_in, nch_out, kernel_size=3, stride = stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(nch_out)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(nch_out, nch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nch_out)
        self.relu2 = nn.ReLU()
        
        self.residual = nn.Sequential()
        if stride != 1 or nch_in != nch_out*self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(nch_in, nch_out*self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(nch_out*self.expansion_factor))
    
    def forward(self, x):
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.residual(out)
        x = self.relu2(x)
        
        return x

class BottleNeck(nn.Module):
    expansion_factor = 4
    def __init__(self, nch_in, nch_out, stride =1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(nch_in, nch_out, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nch_out)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(nch_out, nch_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nch_out)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(nch_out, nch_out*self.expansion_factor, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nch_out*self.expansion_factor)
        self.relu3 = nn.ReLU()
        self.residual = nn.Sequential()
        
        if stride != 1 or nch_in != nch_out * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(nch_in, nch_out*self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(nch_out*self.expansion_factor))
    
    def forward(self,x):
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = x + self.residual(out)
        x = self.relu3(x)
        
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.nch_in = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.conv2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512*block.expansion_factor, num_classes)
        
        self._init_layer()
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        
        layers = []
        for stride in strides:
            layers.append(block(self.nch_in, out_channels, stride))
            self.nch_in = out_channels * block.expansion_factor
        
        return nn.Sequential(*layers)
    
    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
class Model_ResNet:
    def resnet18(self):
        return ResNet(BasicBlock, [2,2,2,2])
    
    def resnet34(self):
        return ResNet(BasicBlock, [3,4,6,3])    

    def resnet50(self):
        return ResNet(BottleNeck, [3,4,6,3])
    
    def resnet101(self):
        return ResNet(BottleNeck, [3,4,23,3])
    
    def resnet152(self):
        return ResNet(BottleNeck, [3,8,36,3])
    

class PlainBlock(nn.Module):
    """
    Plain Block: ResNet의 Basic Block에서 skip connection이 없는 버전
    """
    def __init__(self, nch_in, nch_out, stride):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(nch_in, nch_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nch_out)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(nch_out, nch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nch_out)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class PlainNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PlainNet, self).__init__()
        self.nch_in = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.nch_in, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._init_layer()
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.nch_in, out_channels, stride))
            self.nch_in = out_channels
        return nn.Sequential(*layers)
    
    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class Model_PlainNet:
    def plain34(self):
        return PlainNet(PlainBlock, [3, 4, 6, 3]) 

    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=21, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        
        return x
    


import torch
import torch.nn as nn
from torchvision.models import vgg16_bn

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )


class UNet_preEncoder(nn.Module):
    def __init__(self, pretrained=True, out_channels=21):
        super().__init__()


        self.encoder = vgg16_bn(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)


        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)


        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)


        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)


        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)


        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)


        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)


        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)


        x = self.conv11(x)

        return x

class DECBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        # layers += [nn.ReflectionPad2d(padding=padding)]
        layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                                      bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflection', bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []

        if padding_mode == 'reflection':
            layers += [nn.ReflectionPad2d(padding)]
        elif padding_mode == 'replication':
            layers += [nn.ReplicationPad2d(padding)]
        elif padding_mode == 'constant':
            value = 0
            layers += [nn.ConstantPad2d(padding, value)]
        elif padding_mode == 'zeros':
            layers += [nn.ZeroPad2d(padding)]

        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=0,
                             bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []

        # 1st conv
        layers += [CBR2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias, norm=norm, relu=relu)]

        # 2nd conv
        layers += [CBR2d(in_channels=out_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias, norm=norm, relu=None)]

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(Generator, self).__init__()

        self.dec1 = DECBR2d(1 * in_channels, 4 * nker, kernel_size=4, stride=1,
                            padding=0, norm=norm, relu=0.0, bias=False)

        self.dec2 = DECBR2d(4 * nker, 2 * nker, kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)

        self.dec3 = DECBR2d(2 * nker, 1 * nker, kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)
        
        self.dec4 = DECBR2d(1 * nker, out_channels , kernel_size=4, stride=2,
                            padding=1, norm=None, relu=None, bias=False)
        '''
        self.dec4 = DECBR2d(2 * nker, 1 * nker , kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)

        self.dec5 = DECBR2d(1 * nker,out_channels, kernel_size=4, stride=2,
                            padding=1, norm=None, relu=None, bias=False)
        '''
    def forward(self, x):

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        #x = self.dec5(x)
        x = torch.tanh(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=0.2, bias=False)

        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc5 = CBR2d(8 * nker, out_channels, kernel_size=4, stride=1,
                          padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):

        x = self.enc1(x) # torch.Size([4, 64, 128, 128])
        x = self.enc2(x) # torch.Size([4, 128, 64, 64])
        x = self.enc3(x) # torch.Size([4, 256, 32, 32])
        x = self.enc4(x) # torch.Size([4, 512, 16, 16])
        x = self.enc5(x) # torch.Size([4, 1, 15, 15]) 

        x = torch.sigmoid(x)

        return x
    
class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm', nblk=6):
        super(CycleGAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nker = nker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CBR2d(self.in_channels, 1 * self.nker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
        self.enc2 = CBR2d(1 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc3 = CBR2d(2 * self.nker, 4 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)

        if self.nblk:
            res = []

            for i in range(self.nblk):
                res += [ResBlock(4 * self.nker, 4 * self.nker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0)]

            self.res = nn.Sequential(*res)

        self.dec3 = DECBR2d(4 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.dec2 = DECBR2d(2 * self.nker, 1 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.dec1 = CBR2d(1 * self.nker, self.out_channels, kernel_size=7, stride=1, padding=3, norm=None, relu=None)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x
