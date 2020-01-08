


import sys
print(sys.version) # python 3.6
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
print(torch.__version__) 
import subprocess

import matplotlib.pyplot as plt
import os, time

import itertools
import pickle

import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm

# You can use whatever display function you want. This is a really simple one that makes decent visualizations
def show_imgs(x,epochs,iterations,G, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1) # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())
    plt.text(0, -20, 'Epoch: ' + str(epochs) + ', ' + 'Iteration: ' + str(iterations), fontsize=20)
    torch.save(G.state_dict(),'/home/ubuntu/model')
    with open('/home/ubuntu/image_' + str(epochs) + '_' + str(iterations) + '.png',mode='w'):
      plt.savefig('/home/ubuntu/image_' + str(epochs) + '_' + str(iterations) + '.png')
    subprocess.run(['aws','s3','cp','/home/ubuntu/model','s3://thleats-bucket/models/' + 'etsy_GAN_model'])
    subprocess.run(['aws','s3','cp','/home/ubuntu/image_' + str(epochs) + '_' + str(iterations) + '.png','s3://thleats-bucket/etsy_GAN_out/' + 'image_' + str(epochs) + '_' + str(iterations) + '.png'])



# helper function to initialize the weights using a normal distribution. 
# this was done in the original work (instead of xavier) and has been shown
# to help GAN performance
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    # initializers
    def __init__(self, d=128 ):
        super().__init__()
        
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        #print("G " + str(x.size()))
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        #print("G " + str(x.size()))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        #print("G " + str(x.size()))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        #print("G " + str(x.size()))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        #print("G " + str(x.size()))
        x = torch.tanh(self.deconv5(x))
        #print("G " + str(x.size()))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super().__init__()
        #self.conv0 = nn.Conv2d(3, d , 2, 4 ,2 )
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        #print("D " + str(x.size()))
        #x = F.leaky_relu(self.conv0(x), 0.2)
        #print("D " + str(x.size()))
        x = F.leaky_relu(self.conv1(x), 0.2)
        #print("D " + str(x.size()))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        #print("D " + str(x.size()))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        #print("D " + str(x.size()))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #print("D " + str(x.size()))
        x = torch.sigmoid(self.conv5(x))
        #print("D " + str(x.size()))

        return x

#####
# instantiate a Generator and Discriminator according to their class definition.
#####
D=Discriminator()
G=Generator()

batch_size = 128
lr = 0.0002
train_epoch = 3

import urllib.request
from zipfile import ZipFile
from torch.utils import data
from os import path

img_size = 64

#download the data, and change the filepath
url='https://thleats-bucket.s3.us-east-2.amazonaws.com/jpgs.zip'
location = '/home/ubuntu/jpgs.zip'


if path.exists(location):
  print('already downloaded!')
else:
  print('downloading')
  urllib.request.urlretrieve(url,location)
# Create a ZipFile Object and load sample.zip in it
  with ZipFile(location, 'r') as zipObj:
    # Extract all the contents of zip file in current directory
    zipObj.extractall('/home/ubuntu/LLD_favicons_clean_png/')



dataset=datasets.ImageFolder(root='/home/ubuntu/LLD_favicons_clean_png/',
                                      transform=transforms.Compose([transforms.Resize(img_size),
                                      transforms.CenterCrop(img_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), 
                                      (0.5, 0.5, 0.5)),]))


##### Create the dataloader #####
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,dataset):
    'Initialization'
    self.dataset=dataset
  def __len__(self):
    'Denotes the total number of samples'
    return len(self.dataset)
    #return 1000
  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    x,_ = self.dataset[index] 
    Y = index
    return x, Y

thing=Dataset(dataset)
params={'batch_size':batch_size,'shuffle':True}
training_generator=data.DataLoader(thing,**params)

xbatch, _ = iter(training_generator).next()
xbatch.shape
D(xbatch)
D(xbatch).shape


G = Generator(128)
D = Discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G = G.cuda()
D = D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

num_iter = 0
fixed_z_ = torch.randn(128,100,1,1) 
collect_x_gen = []
train_epoch=1000
import pdb
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for x_, _ in tqdm(training_generator):
        ######################### train discriminator D ###############################
        ###############################################################################
        if x_.size()[0]==128:
          D.zero_grad()
          
          mini_batch = x_.size()[0]
          ##Set optimizer grads to zero
          D_optimizer.zero_grad()
          G_optimizer.zero_grad()
          #create a random noise
          z = torch.randn(mini_batch,100,1,1)
          #create the zeros and ones vector for real and fake
          y_real=torch.ones(x_.size(0)).cuda()
          y_fake=torch.zeros(x_.size(0)).cuda()
          #Pass through discriminiator - train it to recognize real images
          D_result=D(x_.cuda()).squeeze(-1).squeeze(-1)
          #find the real loss for the discriminator
          D_real_loss=BCE_loss(D_result.squeeze(-1),y_real)
          #pass the noise through the generator
          #pdb.set_trace()
          G_result=G(z.cuda())
          #pass the Generated data through the discriminator
          D_result_G=D(G_result)
          #calculate how well the discriminator does at recognizing fake images
          D_fake_loss=BCE_loss(D_result_G.squeeze(-1).squeeze(-1).squeeze(-1),y_fake)
          #calculate the total loss (real + fake) - basically - how good is the discriminator at seeing real from fake
          D_train_loss=D_real_loss+D_fake_loss
          #backpropagation on teh network
          D_train_loss.backward()
          #treain the network
          D_optimizer.step()
          #record the losses
          D_losses.append(D_train_loss.item())
          #rezero the optimizers
          D_optimizer.zero_grad()
          G_optimizer.zero_grad()

          ######################### train generator G ###############################
          ###############################################################################
          G.zero_grad()
          #create more noise
          z_new = torch.randn(128,100,1,1)
          #pass the noise through the generator
          G_result_G=G(z_new.cuda())
          #pass the generated data through the discriminator
          D_result_2=D(G_result_G)
          #find how good the generator is at generating fakes
          G_train_loss=BCE_loss(D_result_2.squeeze(-1).squeeze(-1).squeeze(-1),y_real)
          #calculate the gradients
          G_train_loss.backward()
          #train the network
          G_optimizer.step()    
          #record the stuff
          G_losses.append(G_train_loss.item())
          
          num_iter += 1

      # generate a fixed_z_ image and save
          if num_iter%100==0:
            x_gen = G(fixed_z_.cuda())
            collect_x_gen.append(x_gen.detach().clone())
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # print out statistics
            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                      torch.mean(torch.FloatTensor(G_losses))))
            
            show_imgs(x_gen[:4],epoch,num_iter,G)
