import torch
import torch.nn as nn


class CNN_Model(nn.Module):
    
    def __init__(self,channel_size,label_size,img_size):
        super(CNN_Model,self).__init__()
        
        self.channel_size=channel_size
        self.label_size=label_size
        
        self.CNNblock=nn.Sequential(nn.Conv2d(in_channels=self.channel_size,out_channels=64,kernel_size=3,padding=1,padding_mode="reflect"),
                                    nn.BatchNorm2d((64)),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.MaxPool2d(kernel_size=2,stride=2),# img_s/2
                                    
                                    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2,padding=1,padding_mode="reflect"),
                                    nn.BatchNorm2d((128)),
                                    nn.Dropout(0.2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2),# img_size/4
                                    
                                    nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,padding_mode="reflect"),
                                    nn.BatchNorm2d((256)),
                                    nn.Dropout(0.2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2), # img_size/8
                                    
                                    nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3), # (img_size/8)-2
                                    nn.BatchNorm2d((512)),
                                    nn.Dropout(0.2),
                                    nn.ReLU()
                                    ) 
        
        self.conv_size=int(img_size/8)-2

        
        self.LinearBlock=nn.Sequential(nn.Linear(in_features=self.conv_size*self.conv_size*512,out_features=1024),
                                       nn.ReLU(),
                                       nn.Linear(in_features=1024,out_features=512),
                                       nn.ReLU(),
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Linear(in_features=128,out_features=label_size))
       
    def forward(self,data):
        x=self.CNNblock(data)
        x=x.view(-1,self.conv_size*self.conv_size*512)
        out=self.LinearBlock(x)
        return out
            
