import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import OrderedDict


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm(x)
        return x

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = self.batch_norm(x)
        x = F.relu(self.deconv2(x))
        x = self.batch_norm(x)
        return x
    
class trans_block(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(trans_block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size =1, stride=1)
        self.bn1 = nn.BatchNorm2d(inplanes)        
        self.conv2 = nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.conv3 = nn.Conv2d(inplanes, outplanes, kernel_size =1, stride=1)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder_1 = ConvBlock(3,64)
        self.encoder_2 = ConvBlock(64,128)
        self.encoder_3 = ConvBlock(128,256)
        self.encoder_4 = ConvBlock(256,512)
        
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )  
        
        self.decoder_1 = DeConvBlock(1536,512)
        self.decoder_2 = DeConvBlock(768,256)
        self.decoder_3 = DeConvBlock(384,128)
        self.decoder_4 = DeConvBlock(192,64)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )  

        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)       

        self.up1 = trans_block(1024,1024)
        self.up2 = trans_block(512,512)
        self.up3 = trans_block(256,256)
        self.up4 = trans_block(128,128)

        self.unpool = nn.MaxUnpool2d(2, stride=2)  # get masks
        
        
        #self.classifier = nn.Softmax()

    def forward(self, x):       
        
        size_1 = x.size()
        enc1 = self.encoder_1(x)
        x,indices1 = self.maxpool(enc1) 
        
        size_2 = x.size()
        enc2 = self.encoder_2(x)
        x,indices2 = self.maxpool(enc2)  
        
        size_3 = x.size()
        enc3 = self.encoder_3(x)
        x,indices3 = self.maxpool(enc3)  
        
        size_4 = x.size()
        enc4 = self.encoder_4(x)
        x,indices4 = self.maxpool(enc4)  
        
        center = self.center(x)
        
        dec1 = self.decoder_1(torch.cat([enc4, self.up1(center)], 1)) #여기서 upsample안하고 maxunpool로하려니깐 center와 indices4의 채널갯수가 맞지않음
        dec2 = self.decoder_2(torch.cat([enc3, self.up2(dec1)], 1))
        dec3 = self.decoder_3(torch.cat([enc2, self.up3(dec2)], 1))
        dec4 = self.decoder_4(torch.cat([enc1, self.up4(dec3)], 1))
        final = self.final(dec4)
        #x = self.classifier(x)
        return final

if __name__ == "__main__":
    from segData import DataS
    import torch.optim as optim
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import torchvision.transforms as t

    train_dataset = DataS('train')
    train_loader= DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=8)
    test_dataset = DataS('val')
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True, num_workers=8)
    
    model = Unet()
    if torch.cuda.is_available(): 
        model = model.cuda()
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    model = model.train()
    max_epochs = 1
    flag = False
    for i in range(max_epochs):
        running_loss = 0.0
        if flag: break
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
            
    torch.save(model, 'model2.pt')
            
    model = model.eval()
    total_loss = 0.0
    
    for data in test_loader:
        inputs, labels = data
        if torch.cuda.is_available(): 
                inputs = inputs.cuda()
                labels = labels.cuda()
        inputs = Variable(inputs)
        labels = Variable(labels)                       

        outputs = model(inputs)
        total_loss += len(data) * criterion(outputs, labels).data

    print("Total Loss: %.8f" % (total_loss / len(test_dataset)))