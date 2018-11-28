import glob
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFont, ImageDraw
import warnings
import torch.optim as optim
from torchvision import transforms
import getopt
import sys

from torch.autograd import Variable
from skimage import io#, transform

#https://drive.google.com/drive/folders/1XaFM8BJFligrqeQdE-_5Id0V_SubJAZe?usp=sharing
from torch.utils.data.sampler import SubsetRandomSampler




class ConvNet(nn.Module):

    #Classifying RGB images, therefore number of input channels = 3
    #We want to apply 32 feature detectors (filters), so out channels is 32
    #3x3 filter moves 1 pixel at a time
    #ReLU" all negative values become 0, all positive values remain


    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3_drop = nn.Dropout2d(0.5)


        self.fc1 = torch.nn.Linear(32*32*4, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        torch.nn.init.xavier_uniform(self.conv1.weight) #initialize weights


    def forward(self, x):
        print('Begin forward pass. X shape:',x.shape)        
        x = F.relu(self.conv1(x.cuda()))
        x = self.pool1(x)
        print('Conv1 layer: X shape:',x.shape)
        x = F.relu(self.conv2(x.cuda()))
        x = self.pool2(x)
        print('Conv2 layer: X shape:',x.shape)        
        x = F.relu(self.conv3(x.cuda()))
        x = self.pool3(x)
        print('Conv3 layer: X shape:',x.shape)    
        x = F.dropout(x, training=self.training)
        x = x.view(1, 32*32*4)  #Rectify 
        print('Fully connected layer, shape:',x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
    
        return F.sigmoid(x)


def transform_file(imgfile,transform):
    sample = Image.open(imgfile)
    sample = sample.convert('RGB')
    sample = sample.resize((64,64)) #Resizing images to universal size
    return transform(sample)     

def check_existence(FILENAME):
    if os.path.isfile(FILENAME) == False:
        print('File', FILENAME, 'not found')
        sys.exit(2)
    else:
        return True



def main(argv):

    image_file=''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print("acesim.py <filename.wav>")
        sys.exit(2)
    for opt,arg in opts:
        if opt == "-h":
            print("Passes a *jpg file into the cadoAI. Output: 0 - dogs, 1 - cats. Usage: cadoai.py -i <file.jpg>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            image_file = arg
            check_existence(image_file)


    warnings.filterwarnings("ignore")  #not to dlood the output
    torch.set_printoptions(precision=10)   #to get a nice output
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #working on cuda, not on the CPU

    dtype=torch.cuda.FloatTensor

    transformer = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #normalize the data         

    cnn = ConvNet() #Create the instanse of net 
    #cnn.to(device)  #Send it to GPU
    cnn = cnn.cuda()
    cnn = torch.load('cadoAI.pt')
    cnn.eval()
    inputs = transform_file(image_file,transformer)
    inputs = Variable(inputs.type(dtype))
    inputs = inputs.unsqueeze(0)
    




    print(cnn(inputs.cuda()))
 
    #outputs = cnn(inputs)
           
 
    print('Done.')




if __name__ == "__main__":
   main(sys.argv[1:])



