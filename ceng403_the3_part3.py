# Thanks to Pytorch datasets, this is very easy:

OrigResNet18 = None
###########################################################
# @TODO: Look at PyTorch's documentation for downloading  #
# and loading a pretrained ResNet18 model.                #
#                                                         #
# Hint: This should be a single function call.            #
###########################################################

import torch
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

OrigResNet18 = models.resnet18(pretrained=True)

###########################################################
#                         END OF YOUR CODE                #
###########################################################
print(OrigResNet18)
"""
### 3.1.2 Get ImageNet Labels

Before we finetune ResNet18, let us see what it predicts on CIFAR10. To be able to do so, we will require the list of ImageNet labels.
"""

# Download ImageNet labels
#!wget -nc https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
"""
### 3.1.3 Load CIFAR10

As we mentioned above, ResNet18 expects images with resolution 224x224 and normalized with a mean & std. Therefore, while loading CIFAR10, 
we will apply certain transformations to handle these requirements. (Note that this cell is slightly different than 2.3)
"""

batch_size = 8

TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=TF)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=TF)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

CIFAR10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
"""
### 3.1.4 Apply Pre-trained ResNet18 on CIFAR10

Let us look at what ResNet18 predicts on a batch of CIFAR10. For a set of samples, we will visualize the images and look at ResNet18's top predictions.
"""

# Get a batch
dataiter = iter(testloader)
images, labels = next(dataiter)

# Get scores for classes on the batch
with torch.no_grad():
    output = OrigResNet18(images)

# Convert them to probabilities (shape: batch_size x 1000)
probabilities = torch.nn.functional.softmax(output, dim=1)

# Show results on a 2x2 grid
S = 2
for i in range(S):
    for j in range(S):
        X = images[i * S + j]
        X = np.transpose((X.numpy() / 2 + 0.5), (1, 2, 0))
        top1_prob, top1_catid = torch.topk(probabilities[i * S + j], 1)
        title = "{} p:{:1.2f}".format(categories[top1_catid], top1_prob.item())

        plt.subplot(S, S, i * S + j + 1)
        plt.imshow(X)
        plt.axis('off')
        plt.title(title)
        plt.subplots_adjust(hspace=0.5)
"""
We see that the predictions are way off and we will hopefully get better results with some finetuning.

## 3.2 Adapt ResNet18 for CIFAR10

We will "freeze" the parameters of ResNet18 and replace the last layer of ResNet18 with a new layer which we will finetune.
"""

# Copy ResNet18
NewResNet18 = copy.deepcopy(OrigResNet18)
"""
### 3.2.1 Freeze Parameters of ResNet18

We "freeze" a parameter by setting its `requires_grad` member variable to `False`.
"""

###########################################################
# @TODO: Go over the parameters of NewResNet18 and set    #
# requires_grad for all parameters to False.              #
#                                                         #
# Hint: Check parameters() member function of NewResNet18.#
###########################################################

for param in NewResNet18.parameters():
    param.requires_grad = False

###########################################################
#                         END OF YOUR CODE                #
###########################################################
"""
### 3.2.2 Add a New Learnable FC Layer to ResNet18

If you look at the summary of ResNet18 shown above, you will see that the last layer is:

    `(fc): Linear(in_features=512, out_features=1000, bias=True)`

In our case, we should just replace this with a new FC layer (its dimensions should be straight-forward to figure out).
"""

###########################################################
# @TODO: Create a new layer and save it into              #
# NewResNet18.fc. This new layer will map the activations #
# of the previous layer to the outputs for the CIFAR10    #
# classes.                                                #
###########################################################

num_ftrs = NewResNet18.fc.in_features
NewResNet18.fc = nn.Linear(in_features=num_ftrs, out_features=10)  # CIFAR10 has 10 classes

###########################################################
#                         END OF YOUR CODE                #
###########################################################
"""
### 3.2.3 Visualize the Model

Now, let us see whether the new fc layer is correct for CIFAR10.
"""

print(NewResNet18.fc)
"""
## 3.3 Finetune ResNet18

While finetuning ResNet18, we will just update the last layer.

### 3.3.1 Training Method

This is the same training method from Task 2.
"""


def train(model, criterion, optimizer, epochs, dataloader, verbose=True):
    """
    Define the trainer function. We can use this for training any model.
    The parameter names are self-explanatory.

    Returns: the loss history.
  """
    loss_history = []
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):

            # Our batch:
            inputs, labels = data
            inputs = inputs.to(torch.device("cpu"))
            labels = labels.to(torch.device("cpu"))

            # zero the gradients as PyTorch accumulates them
            optimizer.zero_grad()

            # Obtain the scores
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs.to(torch.device("cpu")), labels)

            # Backpropagate
            loss.backward()

            # Update the weights
            optimizer.step()

            loss_history.append(loss.item())

        if verbose:
            print(
                f'Epoch {epoch} / {epochs}: avg. loss of last 5 iterations {np.sum(loss_history[:-6:-1])/5}')

    return loss_history


"""
### 3.3.2 Finetune the Adapted ResNet18 on CIFAR10

We will only provide the learnable parameters to the optimizer.
"""

# For reproducibility, let us recreate the FC layer here with a fixed seed:
torch.manual_seed(403)
random.seed(403)
np.random.seed(403)

###########################################################
# @TODO: Repeat what you did in 3.2.2 here                #
###########################################################

num_ftrs = NewResNet18.fc.in_features
NewResNet18.fc = nn.Linear(in_features=num_ftrs, out_features=10)

###########################################################
#                         END OF YOUR CODE                #
###########################################################


def get_learnable_parameters(model):
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update


parameters_to_update = get_learnable_parameters(NewResNet18)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(parameters_to_update, lr=0.0001, momentum=0.95)

NewResNet18 = NewResNet18.to(torch.device("cpu"))
epochs = 2
loss_history = train(NewResNet18, criterion, optimizer, epochs, trainloader)
"""
### 3.3.3 The Loss Curve

You will see that the loss curve is very noisy, which suggests that we should finetune our hyper-parameters. Though, we will see that we get already reasonably well performance on test data.
"""

plt.plot(loss_history)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
"""
### 3.3.4 Quantitative Results

We can analyze the accuracy of the predictions as follows. You should see around 69\% accuracies. We can finetune the hyperparameters to obtain better results.

*Disclaimer: This code piece is taken from PyTorch examples.*
"""

correct = 0
total = 0

NewResNet18 = NewResNet18.to(torch.device("cpu"))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(torch.device("cpu"))
        labels = labels.to(torch.device("cpu"))
        outputs = NewResNet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
"""
### 3.3.5 Visual Results

We see that with just two epochs of training a single FC layer, we can get decent results.
"""

# Get a batch
dataiter = iter(testloader)
images, labels = next(dataiter)
images = images.to(torch.device("cpu"))
labels = labels.to(torch.device("cpu"))

# Get scores for classes on the batch
with torch.no_grad():
    output = NewResNet18(images)

# Convert them to probabilities (shape: batch_size x 1000)
probabilities = torch.nn.functional.softmax(output, dim=1)

# Show results on a 2x2 grid
S = 2
for i in range(S):
    for j in range(S):
        X = images[i * S + j]
        X = np.transpose((X.to("cpu").numpy() / 2 + 0.5), (1, 2, 0))
        top1_prob, top1_catid = torch.topk(probabilities[i * S + j], 1)
        title = "{} p:{:1.2f}".format(CIFAR10_classes[top1_catid], top1_prob.item())

        plt.subplot(S, S, i * S + j + 1)
        plt.imshow(X)
        plt.axis('off')
        plt.title(title)
        plt.subplots_adjust(hspace=0.5)
