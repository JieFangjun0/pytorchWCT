from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,fineSize,device):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in listdir(contentPath) if is_image_file(x)]
        self.image_list.sort()
        self.stylePath = stylePath
        self.fineSize = fineSize
        self.styleImg = default_loader(self.stylePath)
        self.se5=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_5_e.pt'),map_location=device)
        self.sv5=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_5_v.pt'),map_location=device)
        self.se4=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_4_e.pt'),map_location=device)
        self.sv4=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_4_v.pt'),map_location=device)
        self.se3=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_3_e.pt'),map_location=device)
        self.sv3=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_3_v.pt'),map_location=device)
        self.se2=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_2_e.pt'),map_location=device)
        self.sv2=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_2_v.pt'),map_location=device)
        self.se1=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_1_e.pt'),map_location=device)
        self.sv1=torch.load(os.path.join('styles_svd',stylePath.split('/')[-1].split('.')[0]+'_1_v.pt'),map_location=device)
        
        
        self.prep = transforms.Compose([
                    transforms.Resize(fineSize),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                    ])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        
        contentImg = default_loader(contentImgPath)
        styleImg = self.styleImg.copy()
        # resize
        if(self.fineSize != 0):
            w,h = contentImg.size
            sw,sh=styleImg.size
            if(w > h):
                if(w != self.fineSize or sw!= self.fineSize):
                    neww = self.fineSize
                    newh = int(h*neww/w)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
            else:
                if(h != self.fineSize or sh!= self.fineSize):
                    newh = self.fineSize
                    neww = int(w*newh/h)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))


        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        return contentImg.squeeze(0),styleImg.squeeze(0),self.se5,self.sv5,self.se4,self.sv4,self.se3,self.sv3,self.se2,self.sv2,self.se1,self.sv1

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
