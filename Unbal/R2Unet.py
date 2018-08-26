import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import OrderedDict

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        a = self.conv2
        
        stack1 = self.conv1(x)
        stack2 = self.batch_norm(stack1)
        stack3 = F.relu(stack2)
        
        stack4 = a(stack3)
        stack5 = stack1+stack4
        stack6 = self.batch_norm(stack5)
        stack7 = F.relu(stack6)
        
        stack8 = a(stack7)
        stack9 = stack8+stack1
        stack10 = self.batch_norm(stack9)
        stack11 = F.relu(stack10)
        
        stack12 = a(stack11)
        stack13 = stack1+stack12
        stack14 = self.batch_norm(stack13)
        stack15 = F.relu(stack14)
        
        return stack15
    
class Get_Sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Get_Sample, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.batch_norm(self.conv(x))
        return x
    
class Up_Block(nn.Module):
    def __init__(self, inplanes):
        super(Up_Block, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size =1, stride=1)
        self.deconv = nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(inplanes)  
        
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = F.relu(self.bn(self.deconv(out)))
        out = F.relu(self.bn(self.conv(out)))
        return out
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.encoder_1 = ConvBlock(3,64)
        self.upsample1 = Get_Sample(3,64)
        
        self.encoder_2 = ConvBlock(64,128)
        self.upsample2 = Get_Sample(64,128)
        
        self.encoder_3 = ConvBlock(128,256)
        self.upsample3 = Get_Sample(128,256)
        
        self.encoder_4 = ConvBlock(256,512)
        self.upsample4 = Get_Sample(256,512)
        
        self.encoder_5 = ConvBlock(512,1024)

        self.decoder_1 = ConvBlock(1536,512)
        self.downsample = Get_Sample(1536,512)
        
        self.decoder_2 = ConvBlock(768,256)
        self.downsample2 = Get_Sample(768,256)
        
        self.decoder_3 = ConvBlock(384,128)
        self.downsample3 = Get_Sample(384,128)
        
        self.decoder_4 = ConvBlock(192,64)
        self.downsample4 = Get_Sample(192,64)
        
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))       

        self.up1 = Up_Block(1024)
        self.up2 = Up_Block(512)
        self.up3 = Up_Block(256)
        self.up4 = Up_Block(128)
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )  

    def forward(self, x):       
        enc1 = self.encoder_1(x)
        temp = self.upsample1(x)
        enc1+=temp
        x = self.maxpool(enc1)

        enc2 = self.encoder_2(x)
        temp = self.upsample2(x)
        enc2+=temp
        x = self.maxpool(enc2)  
        
        enc3 = self.encoder_3(x)
        temp = self.upsample3(x)
        enc3+=temp
        x = self.maxpool(enc3)  
        
        enc4 = self.encoder_4(x)
        temp = self.upsample4(x)
        enc4+=temp
        x = self.maxpool(enc4)  
        
        center = self.encoder_5(x)
        
        
        temp = torch.cat([enc4, self.up1(center)], 1)
        dec1 = self.decoder_1(temp) 
        dec1 += self.downsample(temp)
        
        temp = torch.cat([enc3, self.up2(dec1)], 1)
        dec2 = self.decoder_2(temp)
        dec2 += self.downsample2(temp)

        temp = torch.cat([enc2, self.up3(dec2)], 1)
        dec3 = self.decoder_3(temp)
        dec3 += self.downsample3(temp)
        
        temp = torch.cat([enc1, self.up4(dec3)], 1)
        dec4 = self.decoder_4(temp)
        dec4 += self.downsample4(temp)
        
        final = self.final(dec4)
        return final

if __name__ == "__main__":
    from segData import DataS
    import torch.optim as optim
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import torchvision.transforms as t

    train_dataset = DataS('train')
    train_loader= DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8)
    
    # model = torch.load('model2_epoch50.pt')
    model = Unet()
    if torch.cuda.is_available(): 
        model = model.cuda()
        model = nn.DataParallel(model)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    model = model.train()
    max_epochs = 20
    for i in range(max_epochs):
        running_loss = 0.0
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs = Variable(inputs)
            labels = Variable(labels)                       
            model.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if j % 1000 == 999:
                print('[epoch: %d, j: %5d] average loss: %.8f' % (i + 1, j + 1, running_loss / 1000))
                running_loss = 0.0
    torch.save(model, 'model3.pt')