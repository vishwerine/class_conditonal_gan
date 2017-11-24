
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

"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)
"""


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
    lr = 1e-4
    beta1 = 0.5
    cuda= True
    ngpu = 1
    netG = ''
    netD = ''
    outf = 'results_wgan'
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

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(opt.dataroot, train=True, download=True,
                   transform=transforms.Compose([
                        transforms.Scale(opt.imageSize),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

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


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc_d) x 64 x 64
            nn.Conv2d(nc_d, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8 , 1, 4, 1, 0, bias=False)
            )

    def forward(self, input, image_condition):
        combined = torch.cat((input, image_condition), 1)
        output = self.main(combined)
        return output.view(-1, 1).squeeze(1)



netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

noise_condition = torch.FloatTensor(opt.batchSize, 10)
image_condition = torch.FloatTensor(opt.batchSize, 10, opt.imageSize, opt.imageSize)

noise = torch.FloatTensor(opt.batchSize, nz - nc_d + nc_g, 1, 1)

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0



fixed_noise = torch.FloatTensor(opt.batchSize, nz - nc_d + nc_g, 1, 1).normal_(0, 1)
fixed_noise = Variable(fixed_noise)

fixed_labels = torch.LongTensor(opt.batchSize,1)

for i in range(opt.batchSize):
    fixed_labels[i] = i%8

fixed_labels = Variable(fixed_labels)



if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    fixed_labels = fixed_labels.cuda()
    noise_condition , image_condition = noise_condition.cuda() , image_condition.cuda()


# setup optimizer
optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)



def get_one_hot_image(labels, batchSize, imageSize):
    if opt.cuda:
        y_onehot = torch.FloatTensor(batchSize, 10).cuda()
        labels = labels.cuda()
    else:
        y_onehot = torch.FloatTensor(batchSize, 10)
    
    try:
        y_onehot.zero_().scatter_(1, labels.unsqueeze(1), 1)
    except:
        y_onehot.zero_().scatter_(1, labels, 1)
    
    y_onehot = torch.unsqueeze(torch.unsqueeze(y_onehot, 2),3).expand(batchSize, 10, imageSize, imageSize)

    return y_onehot



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


  


def newconcat(images, labels, batchSize, imageSize):
    labels = labels.cuda()
    y_onehot = get_one_hot_image(labels,batchSize, imageSize)
    images = images.cuda()
    
    if opt.cuda:
        return torch.cat((images, y_onehot),1).cuda()
    else:
        return torch.cat((images, y_onehot),1).cpu()



# In[192]:

def newconcat_noise(images, labels, batchSize):
    labels = labels.cuda()
    y_onehot = get_one_hot(labels, batchSize)
    images = images.cuda()

    if opt.cuda:
        return torch.cat((images, y_onehot),1).cuda()
    else:
        return torch.cat((images, y_onehot),1).cpu()






logFile = open(opt.outf+'/statistics.txt','w')

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        for _ in range(5):    
            netD.zero_grad()
            images, classes = data


            real_cpu = images
            batch_size = images.size(0)

            wrong_classes = torch.LongTensor(batch_size)
            for j in range(batch_size):
                wrong_classes[j] = (classes[j]+4)%10


            cvec = get_one_hot(classes, batch_size)
            ivec = get_one_hot_image(classes, batch_size, opt.imageSize)
            cvec_w = get_one_hot(wrong_classes,batch_size)
            ivec_w = get_one_hot_image(wrong_classes, batch_size, opt.imageSize)

            noise_condition.resize_as_(cvec).copy_(cvec)
            image_condition.resize_as_(ivec).copy_(ivec)

            #mlist = []
            #for m in netD.modules():
            #    mlist.append(m)

            #print(mlist[13].weight)
        
        

            if opt.cuda:
                real_cpu = real_cpu.cuda()
                classes = classes.cuda()

            input.resize_as_(real_cpu).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
        
            labelv = Variable(label)
            noise_conditionv = Variable(noise_condition)
            image_conditionv = Variable(image_condition)
        
            output1 = netD(inputv, image_conditionv)
            #print(output)
            #errD_real = criterion(output, labelv)
            #errD_real.backward()
            D_x = output1.data.mean()

            # train with fake
            noise.resize_(batch_size, nz -nc_d + nc_g , 1, 1).normal_(0, 1)
            noisev = Variable(noise)
        
            fake = netG(noisev, noise_conditionv)
            #fake = Variable(newconcat(fake.data,labels,batch_size))
        
            labelv = Variable(label.fill_(fake_label))
            output2 = netD(fake.detach(), image_conditionv)
            #errD_fake = criterion(output, labelv)
            #errD_fake.backward()
            D_G_z1 = output2.data.mean()

            # train with wrong classes
            noise_condition.resize_as_(cvec_w).copy_(cvec_w)
            image_condition.resize_as_(ivec_w).copy_(ivec_w)

            noise_conditionv = Variable(noise_condition)
            image_conditionv = Variable(image_condition)
            output3 = netD(inputv, image_conditionv)
            #errD_wrong = criterion(output, labelv)
            #errD_wrong.backward()
            D_x_w = output3.data.mean()


            #errD = errD_real + errD_fake + errD_wrong
            D_loss = -(torch.mean(output1) - torch.mean(output2) - torch.mean(output3))
            D_loss.backward()        
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        image_condition.resize_as_(ivec).copy_(ivec)
        image_conditionv = Variable(image_condition)

        output4 = netD(fake, image_conditionv)
        #errG = criterion(output, labelv)
        #errG.backward()
        D_G_z2 = output4.data.mean()
        G_loss = -torch.mean(output4)
        G_loss.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 D_loss.data[0], G_loss.data[0], D_x, D_G_z1, D_G_z2))
        logFile.write('LossD '+str(D_loss.data[0])+' lossG '+str(G_loss.data[0])+ ' Dx '+str(D_x)+' D(G(z)) '+str(D_G_z1)+' '+str(D_G_z2)+'\n')
        if i % 100 == 0:
            vutils.save_image(images,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            #Variable(newconcat_noise(fixed_noise.data,fixed_labels,opt.batchSize))
            fake = netG(fixed_noise, Variable(get_one_hot(fixed_labels.data, opt.batchSize)))
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

logFile.close()






