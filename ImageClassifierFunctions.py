

# Imports here

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


data_dir = './'
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'
# Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(30),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(30), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
}

#Load the datasets with ImageFolder
image_datasets ={x: datasets.ImageFolder(os.path.join(data_dir, x),
                                      data_transforms[x])
              for x in [train_dir, valid_dir, test_dir]}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
           for x in [train_dir,valid_dir, test_dir]} 
           #by mistake the process(img) function call had come here
           #removed it based on the review

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

#Train the classifier layers using backpropagation using 
#the pre-trained network to get the features


def train_model(model, criterion, optimizer, scheduler, epochs=10, gpu= False):
    type_changer = torch.FloatTensor
    if gpu:
        type_changer = torch.cuda.FloatTensor
    for x in model.features:
        if type(x) == nn.modules.conv.Conv2d:
            x.weight.data = x.weight.data.type(type_changer)
            x.bias.data = x.bias.data.type(type_changer)

    for x in model.classifier:
        if type(x) == nn.modules.linear.Linear:
            x.weight.data = x.weight.data.type(type_changer)
            x.bias.data = x.bias.data.type(type_changer) 

    model_weights = copy.deepcopy(model.state_dict())
    classi_accuracy = 0.0
    
    #Iterate over the data, with number of times = epochs
    for ep in range(epochs):
        
        print('Epoch {}/{}'.format(ep, epochs - 1))
        print(40*"=")
        
        # In each epoch, we have training and validation phases
        for p in ['train', 'valid']:
            if p == 'train':
                scheduler.step()
                #Training
                model.train()  
            else:
                #Validation
                model.eval()   

            inter_loss = 0.0
            inter_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[p]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Set the parameter weights to zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(p == 'train'):
                    #inputs = Variable(inputs)
                    #labels = Variable(labels)
                    outputs = model.forward(inputs)
                    ps = torch.exp(outputs).data
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if p == 'train':
                        loss.backward()
                        optimizer.step()
                    elif p == 'valid':
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        outputs = model.forward(inputs)
                        ps = torch.exp(outputs).data
                        loss = criterion(outputs, labels)

                # Intermediate loss
                eq = (labels.data == ps.max(1)[1])
                inter_loss += loss * inputs.size(0)
                
                #Intermediate corrects
                inter_corrects += eq.type_as(torch.FloatTensor()).sum()

            
            epoch_loss = inter_loss / float(dataset_sizes[p])
            epoch_accuracy = inter_corrects.double() / float(dataset_sizes[p])
            
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(p, epoch_loss, epoch_accuracy))

            if p == 'valid' and epoch_accuracy > classi_accuracy:
                classi_accuracy = epoch_accuracy
                model_weights = copy.deepcopy(model.state_dict())


    print('Best validation Classification Accuracy: {:4f}'.format(classi_accuracy))

    model.load_state_dict(model_weights)
    return model



# Do validation on the test set
def testing(model):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in iter(dataloaders[test_dir]):
            inputs = inputs.cuda()
            labels = inputs.cuds()
            output = model(inputs)
            
            # Class prediction 
            pred = output.max(1, keepdim= True)[1]
            correct_classification += pred.eq(labels.view_as(pred)).sum().item()
    print("Testing Classification Accuracy: {}"          .format(correct*100.0/dataset_sizes['test']))


# Write a function that loads a checkpoint and rebuilds the model
def loads_checkpoint(path = "./checkpoint.pth"):
    ckpoint = torch.load(path)
    pretrained_model = ckpoint['model_name']
    
    #implemented the two model selection requirement and the review

    if pretrained_model == "VGG16":
        model = models.vgg16(pretrained=True)
    elif pretrained_model == "VGG19":
        model = models.vgg19(pretrained=True)

    #Ensure that the optimiser skips training the VGG parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = ckpoint['classifier'] #changed the classifier part of vgg with saved classifier
    model.class_to_idx = ckpoint['class_to_idx']
    model.load_state_dict(ckpoint['state_dict'])
    
    #Change type of weights
    for x in model.features:
        if type(x) == nn.modules.conv.Conv2d:
            x.weight.data = x.weight.data.type(torch.DoubleTensor)
            x.bias.data = x.bias.data.type(torch.DoubleTensor)

    for x in model.classifier:
        if type(x) == nn.modules.linear.Linear:
            x.weight.data = x.weight.data.type(torch.DoubleTensor)
            x.bias.data = x.bias.data.type(torch.DoubleTensor)  

    return(model)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #Process a PIL image for use in a PyTorch model
    #scaling
    y_to_x_ratio = image.size[1] / image.size[0]
    trans = transforms.ToTensor()
    x = 256
    y = int(y_to_x_ratio * x)
    image = image.resize((x, y))
    
    #cropping from center
    center_width = image.size[0] / 2
    center_height = image.size[1] / 2
    
    cropped_image = image.crop(
        (
        center_width - 112,
        center_height - 112,
        center_width + 112,
        center_height + 112
        )
                        )
    #Normalize image
    np_image = np.array(cropped_image)
    np_image = np.array(np_image)/255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (np_image - mean) / std
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)


#Visualize the image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5, gpu="No"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #Implement the code to predict the class from an image file
    #image = None
    print("Prediction function entered!")
    print(40*"=")
    with Image.open(image_path) as img:
        image = process_image(img)
    print("Image loaded from path: "+str(image_path))
    print(40*"=")
    image = Variable(image.unsqueeze(0))
    #print(gpu)

    if gpu == "Yes":
        print("GPU=True and Tommy Lee Johnes")
        print(40*"=")
        #image = image.to(torch.device("cuda:0"))

        model = model.cuda()
        image = image.type_as(torch.cuda.DoubleTensor())

        oput = model.forward(image)
        ps = torch.exp(oput)
        probs, ind = ps.topk(topk)
        
        classes=[]
        probabilities = []
        
        probs=([probs.data.cpu()[0,i].numpy() for i in range(len(probs.data.cpu()[0]))])
        
        ind = [ind.data.cpu()[0,i].numpy() for i in  range(len(ind.data.cpu()[0]))]
        
        probabilities = np.hstack(probs)
        ind= np.hstack(ind)
        
        idx_to_classes=dict(model.idx_to_class)
        
        for c in ind:
                classes.append(idx_to_classes[c])
        print(40*"=")
        return classes, probabilities

    elif gpu == "No":
        print("GPU=False and Snadra Bullock")
        print(40*"=")
        image = image.type_as(torch.DoubleTensor())
        oput = model.forward(image)
        ps = torch.exp(oput)
        probs, ind = ps.topk(topk)
        
        classes=[]
        probabilities = []
        
        probs=([probs.data[0,i].numpy() for i in range(len(probs.data[0]))])
        
        ind = [ind.data[0,i].numpy() for i in  range(len(ind.data[0]))]
        
        probabilities = np.hstack(probs)
        ind= np.hstack(ind)
        
        idx_to_classes=dict(model.idx_to_class)
        
        for c in ind:
                classes.append(idx_to_classes[c])
        print(40*"=")
        return classes, probabilities
