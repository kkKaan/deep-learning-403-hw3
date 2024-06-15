import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    print("Cuda (GPU support) is available and enabled!")
    device = torch.device("cuda")
else:
    print("Cuda (GPU support) is not available :(")
    device = torch.device("cpu")
"""
## 2.3 The Dataset

We will use torchvision.datasets to download the CIFAR10 dataset.
"""

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
"""### 2.3.1 Visualize Samples"""

dataiter = iter(trainloader)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
S = 4

for i in range(S):
    for j in range(S):
        images, labels = next(dataiter)
        X = np.transpose(images[0].numpy() / 2 + 0.5, (1, 2, 0))
        y = labels[0]

        plt.subplot(S, S, i * S + j + 1)
        plt.imshow(X)
        plt.axis('off')
        plt.title(classes[y])
        plt.subplots_adjust(hspace=0.5)

plt.show()
"""
## 2.4 Define and Train a Small CNN Model

Now, all the pieces are ready and we can define a model.

### 2.4.1 Model Definition

Create a three-layer CNN with the following layers:

| Layer Name | Input HxW | Filter size | Stride | Pad | # of in channels | Out HxW | # of out channels |
| ----| -----| ----| ---| ---| ----| -----|---------- |
| Conv1   | 32x32 | ? | ? | ? | 3  | 28x28 | 16 |
| Conv2   | 28x28 | ? | ? | ? | 16 | 26x26 | 32 |
| Maxpool | 26x26 | 4 | 2 | 0 | 32 | 12x12 | 32 |
| Conv3   | 12x12 | ? | ? | ? | 32 | 10x10 | 32 |

and the fully-connected layers:

| Layer Name | Input Size | Output size |
| ----| -----| ----|
| FC1 | 3200 | 1500 |
| FC2 | 1500 | 10 |

You should choose suitable values for variables marked with `?' in the table and make sure that receptive fields can be properly placed in all layers.

While creating your model, pay attention to the following aspects:
* Each Conv layer and FC layer will be followed by ReLU, except for the last one.
* You should keep all other parameters (dilation, bias, group-mode, ..) as their default values in Pytorch.

You will need to read the following pages from Pytorch regarding the layers that you will use:
* [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
* [MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d)
"""


class SmallCNN(nn.Module):

    def __init__(self):
        super(SmallCNN, self).__init__()
        torch.manual_seed(403)
        random.seed(403)
        np.random.seed(403)
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.maxpool = None

        self.fc1 = None
        self.fc2 = None

        ###########################################################
        # @TODO: Create the convolutional and FC layers as        #
        #  described above.                                       #
        ###########################################################

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(in_features=3200, out_features=1500)
        self.fc2 = nn.Linear(in_features=1500, out_features=10)

        ###########################################################
        #                         END OF YOUR CODE                #
        ###########################################################

    def forward(self, x):
        ###########################################################
        # @TODO: Feedforward x through the layers. Note that x    #
        # needs to be reshaped to (batchsize, 3200) before        #
        # the FC layers.                                          #
        ###########################################################

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        ###########################################################
        #                         END OF YOUR CODE                #
        ###########################################################
        return x


"""
### 2.4.2 Trainer for the Model

Let us define our training function, which will use the cuda device for training the model.
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
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the gradients as PyTorch accumulates them
            optimizer.zero_grad()

            # Obtain the scores
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs.to(device), labels)

            # Backpropagate
            loss.backward()

            # Update the weights
            optimizer.step()

            loss_history.append(loss.item())

        if verbose:
            print(
                f'Epoch {epoch} / {epochs}: avg. loss of last 5 iterations {np.sum(loss_history[:-6:-1])/5}')

    return loss_history


"""### 2.4.3 Create and visualize the model"""

model = SmallCNN()
summary(model, input_size=(batch_size, 3, 32, 32))
"""
### 2.4.4 Train the Small Model

We will create an instance of our model and "define" which loss function we want to use. We will also state our choice for the optimizer here.

For more information, check the PyTorch docs: [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 
and [SGD](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD).
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model = model.to(device)
epochs = 10
loss_history = train(model, criterion, optimizer, epochs, trainloader)
"""
### 2.4.5 The Loss Curve

Let us visualize the loss curve.
"""

plt.plot(loss_history)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
"""
### 2.4.6 Quantitative Analysis

We can analyze the accuracy of the predictions as follows. You should see around 54\% accuracies. 
We can finetune the hyperparameters to obtain better results. But we will skip that and go for a bigger model.

*Disclaimer: This code piece is taken from PyTorch examples.*
"""

correct = 0
total = 0
model = model.to("cpu")
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
"""
## 2.5 Your CNN

Now, create your own CNN. It should have at least 5 convolutional layers. Other than that, there is no restriction 
on what you can use in your CNN or how you can structure it.

### 2.5.1 Model Definition
"""


class YourCNN(nn.Module):

    def __init__(self):
        super(YourCNN, self).__init__()
        torch.manual_seed(403)
        random.seed(403)
        np.random.seed(403)

        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.conv5 = None

        self.pool = None
        self.dropout = None

        self.fc1 = None
        self.fc2 = None

        ###########################################################
        # @TODO: Create your layers here.                         #
        ###########################################################

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 1 * 1, 256)  # Adjusted fully connected layer size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

        ###########################################################
        #                         END OF YOUR CODE                #
        ###########################################################

    def forward(self, x):
        ###########################################################
        # @TODO: Feedforward x through the layers.                #
        ###########################################################

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 512 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        ###########################################################
        #                         END OF YOUR CODE                #
        ###########################################################

        return x


"""### 2.5.2 Create and visuale your model"""

model = YourCNN()
summary(model, input_size=(batch_size, 3, 32, 32))
"""### 2.5.3 Train the Model"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model = model.to(device)
epochs = 10
loss_history = train(model, criterion, optimizer, epochs, trainloader)
"""
### 2.5.4 Loss Curve

Let us visualize the loss curve.
"""

plt.plot(loss_history)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
"""### 2.5.5 Quantitative Analysis

Analyze your model quantitatively.

*Disclaimer: This code piece is taken from PyTorch examples.*
"""

correct = 0
total = 0
model = model.to("cpu")
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
