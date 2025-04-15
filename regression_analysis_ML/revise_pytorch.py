import torch

lst = [1,2,3][2,3,5]

tns = torch.tensor(lst)

torch.shape
torch.device
torch.dtype

#neural network
# linear layer -  each linear layer has a weight and a bias
#networks with only linear layers are called fully connected networks
import torch.nn as nn
input_tensor = torch.Tensor([[1,2,3]])
linear_layer = nn.Linear(in_features=3, out_features=2)
output = linear_layer(input_tensor)
print(output)

#three linear layer
model = nn.Sequential(
    nn.Linear(10, 18),
    nn.Linear(18, 20),
    nn.Linear(20, 5)
)

# sigmoid for binary classifiation -- either 1 or 0
model = nn.Sequential(
    nn.Linear(10, 18),
    nn.Linear(18, 20),
    nn.Sigmoid()
)
# softmax for multi-class classification problem; dim=-1 indicates sofmax is applied to input tensor's last dimension

input = torch.Tensor([1, 2, 3])

probabilties = nn.Softmax(dim=-1)
output = probabilties(input)
print(output)

'''
Three parts to pytorch:
pytorch dataset (and dataloaders)
model
training loop
'''

"""
STEPS IN TRAININ A NEAURAL NETWORK:

1. DEFINE YOUR PROBLEM AND DATA - IMAGE CLASSIFICATION, TEXT GENERATION ETC
- GATHER, PREPROCESS, AND SPLIT DATA INTO TRAINING, VALIDATION AND TEST SET
2. CHOOSE YOUR MODEL ARCHITECTURE - E.G CNN FOR IMAGES, RNN FOR SEQUENCES
- USE PYTORCH MODULES TO BUILD YOUR LAYERS nn.Linear, nn.Conv2d, nn.LSTM
3. DEFINE LOSS FUNCTION - measures how well model prediction matches the actual label (MSE, CrossEntropy)
4. Choose an optimizer - this algorithm updates your model to minimize the loss during trainin
5. SET UP A TRAINING LOOP
- FORWARD PASS; CALCULATE LOSS; BACKWARD PASS; UPDATE MODEL PARAMTERS
6. Evaluate your model AND 
7. Improve and refine
"""
import torch, torchvision
import torch.nn as nn
import tqdm.notebook as tqdm
from torch.utils.data import Dataset, DataLoader
import torch.transforms as transforms

class MyDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

#define loss function
loss_fn = nn.MSELoss()

#define optimizer 
optim = nn.optim.Adam(model.parameters(), lr=0.001) #lr = learning rate

# set up dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
train_folder = '../path/for...'
test_folder = '../path/for...'
val_folder = '../path/'

train_dataset = MyDataSet(train_folder, transform=transform)
test_dataset = MyDataSet(test_folder, transform=transform)
val_dataset = MyDataSet(val_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_dataset = DataLoader(val_dataset, batch_size=32, shuffle=False)

#train the network
num_epoch = 3

for epoch in range(num_epoch):
    for data, label in train_loader:
        #forward pass
        output = model(data)
        loss = loss_fn(output, label)
        #backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Validation step (optional)
        with torch.no_grad():  # Disable gradient calculation for validation
            val_loss = 0
            for val_data, val_labels in val_loader:
                val_outputs = model(val_data)
                val_loss += loss_fn(val_outputs, val_labels).item()
            val_loss /= len(val_loader)

            print(f"Epoch: {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

#save your model
torch.save(model.state_dict(), 'model.pt')

#load saved model 
model.load_state_dict(torch.load('model.pt'))
# Create test dataset and loader
test_dataset = MyDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32)

# Test the model
test_loss = 0
with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        test_loss += loss_fn(outputs, labels).item()
test_loss /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}") 












"""ANOTHER WAY FROM SENDEX"""
#1. IMPORTING AND PREPARING OUR DATA
import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
#DATA LOADER
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

#CHECK CONTENT
"""for data in trainset:
    print(data)
    break

X, y = data[0][0], data[1][0]

print(data[1])

import matplotlib.pyplot as plt  # pip install matplotlib

plt.imshow(data[0][0].view(28,28))
plt.show()
"""

#BALANCING
total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}


for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1

print(counter_dict)

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100.0}%")

#2. BUILDING OUR NEAURAL NETWORK 
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()
print(net)

#3. TRAINING OUR NEAURAL NETWORK
import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(3): # 3 full passes over the data
    for data in trainset:  # `data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
        output = net(X.view(-1,784))  # pass in the reshaped batch (recall they are 28x28 atm)
        loss = F.nll_loss(output, y)  # calc and grab the loss value
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 

correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1,784))
        #print(output)
        for idx, i in enumerate(output):
            #print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3)) #Accuracy:  0.968

import matplotlib.pyplot as plt

plt.imshow(X[0].view(28,28))
plt.show()

