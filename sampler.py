

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


class opt:
    dataset = "cifar10"
    dataroot = "data"
    workers = 2
    batchSize = 64
    imageSize = 64
    nz = 100
    ngf = 64
    ndf = 64
    niter=500
    lr = 0.0002
    beta1 = 0.5
    cuda= False
    ngpu = 1
    netG = 'results2/netG_epoch_20.pth'
    netD =''
    outf= 'results2'
    manualSeed = 7053


opt = opt()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc_g = 3
nc_d = 13


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc_g, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc_g) x 64 x 64
        )

    def forward(self, input ,noise_condition):
        combined = torch.cat((input, noise_condition), 1)
        output = self.main(combined)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)



def get_one_hot( labels, batchSize):
    if opt.cuda:
        y_onehot = torch.FloatTensor(batchSize, 10).cuda()
        labels = labels.cuda()
    else:
        y_onehot = torch.FloatTensor(batchSize, 10)
    #print (label.unsqueeze(1).size())
    try:
        y_onehot.zero_().scatter_(1, labels.unsqueeze(1), 1)
    except:
        y_onehot.zero_().scatter_(1, labels, 1)

    return y_onehot



fixed_noise = torch.FloatTensor(opt.batchSize, nz - nc_d + nc_g, 1, 1).normal_(0, 1)
fixed_noise = Variable(fixed_noise)

fixed_labels = torch.LongTensor(opt.batchSize,1).zero_()


dic = {
	'airplane' : 0,
	'automobile':1,
	'bird':2,
	'cat':3,
	'deer':4,
	'dog':5,
	'frog':6,
	'horse':7,
	'ship':8,
	'truck':9
}



class_label = input('please enter a category to generate images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck :- ')

while class_label not in dic.keys():
	print('error: class doesnt exist')
	class_label  =input('Enter again:-')

class_value = dic[class_label]


for i in range(opt.batchSize):
    fixed_labels[i] = class_value

fixed_labels = Variable(fixed_labels)



fake = netG(fixed_noise, Variable(get_one_hot(fixed_labels.data, opt.batchSize)))
vutils.save_image(fake.data,'fake_samples_epoch_'+class_label+'.png',normalize=True)
