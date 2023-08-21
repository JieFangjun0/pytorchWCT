import torch
import argparse
from vLoader import Dataset

from vgg19_decoders import VGG19Decoder1, VGG19Decoder2, VGG19Decoder3, VGG19Decoder4, VGG19Decoder5 
from vgg19_normalized import VGG19_normalized
import torch.nn as nn

import os

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath',default='/home/jiefangjun/Pictures/LDV2_test15/003',help='path to train')
parser.add_argument('--stylePath',default='/home/sam/Pictures/styles/Viento.jpg',help='path to train')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--encoder', default='models/vgg19_normalized.pth.tar', help='Path to the VGG conv1_1')
parser.add_argument('--decoder5', default='models/vgg19_normalized_decoder5.pth.tar', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/vgg19_normalized_decoder4.pth.tar', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/vgg19_normalized_decoder3.pth.tar', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/vgg19_normalized_decoder2.pth.tar', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/vgg19_normalized_decoder1.pth.tar', help='Path to the decoder1')
parser.add_argument('--cuda', action='store_true', default=True,help='enables cuda')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")
args = parser.parse_args()

class WCT(nn.Module):
    def __init__(self,args):
        super(WCT, self).__init__()
        # load pre-trained network
        self.encoder = VGG19_normalized()
        self.encoder.load_state_dict(torch.load(args.encoder))

        self.d1 = VGG19Decoder1()
        self.d1.load_state_dict(torch.load(args.decoder1))
        self.d2 = VGG19Decoder2()
        self.d2.load_state_dict(torch.load(args.decoder2))
        self.d3 = VGG19Decoder3()
        self.d3.load_state_dict(torch.load(args.decoder3))
        self.d4 = VGG19Decoder4()
        self.d4.load_state_dict(torch.load(args.decoder4))
        self.d5 = VGG19Decoder5()
        self.d5.load_state_dict(torch.load(args.decoder5))

    def whiten_and_color(self,sF,file_name):
        # assuming input shape is batch_size x c x h x w
        batch_size, sFSize = sF.size(0), sF.size()[1:]
    
        sF = sF.view(batch_size, sFSize[0], -1)
        s_mean = torch.mean(sF, 2)
        sF = sF - s_mean.unsqueeze(2).expand_as(sF)
        styleConv = torch.bmm(sF, sF.transpose(1, 2)).div(sFSize[1] - 1)
        # s_u, s_e, s_v = torch.linalg.svd(styleConv)
        s_u, s_e, s_v = torch.linalg.svd(styleConv)
        s_u=s_u[0]
        s_e=s_e[0]
        s_v=s_v[0]
        torch.save(s_u,os.path.join('styles_svd',file_name+'_u.pt'))
        torch.save(s_e,os.path.join('styles_svd',file_name+'_e.pt'))
        torch.save(s_v,os.path.join('styles_svd',file_name+'_v.pt'))
        return 

    def transform(self, sF,file_name):
        batch_size, C,W1, H1 = sF.size()
        sF = sF.double().view(batch_size, C, -1)

        targetFeature = self.whiten_and_color(sF,file_name)
        
        return

def styleTransfer(wct, styleImg,style_name):
    
    sF5 = wct.encoder(styleImg, 'relu5_1')
    file_name=style_name+'_5'
    csF5 = wct.transform(sF5,file_name)

    sF4 = wct.encoder(styleImg, 'relu4_1')
    file_name=style_name+'_4'
    csF4 = wct.transform(sF4,file_name)
    
    sF3 = wct.encoder(styleImg, 'relu3_1')
    file_name=style_name+'_3'
    csF3 = wct.transform(sF3,file_name)

    sF2 = wct.encoder(styleImg, 'relu2_1')
    file_name=style_name+'_2'
    csF2 = wct.transform(sF2,file_name)

    sF1 = wct.encoder(styleImg, 'relu1_1')
    file_name=style_name+'_1'
    csF1 = wct.transform(sF1,file_name)
    
    return 

def main():
    import os
    basePath=os.path.join(os.getenv('HOME'),"Pictures")
    stylesPath="styles"
    styles=[file for file in os.listdir(os.path.join(basePath,stylesPath)) \
    if file.endswith('.jpg')]
    styles.sort()
    wct = WCT(args)
    if(args.cuda):
        wct.cuda(args.gpu)
    for style in styles:
        stylePath=os.path.join(basePath,stylesPath,style)
        style_name=style.split('.')[0]
        with torch.no_grad():
            # Data loading code
            dataset = Dataset(args.contentPath,stylePath,args.fineSize,'cuda:0')
            loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False)
            for i,(contentImg,styleImg) in enumerate(loader):
                if(args.cuda):
                    styleImg = (styleImg.cuda(args.gpu))
                styleTransfer(wct,styleImg,style_name)
                break
if __name__ == "__main__":
    main()
    
