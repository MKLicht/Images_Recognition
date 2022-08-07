import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 64
num_epoch = 100

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ]
)

data_train = datasets.MNIST(
    root = './data/',
    transform = transform,
    train = True,
    download = True
)

dataloader_train = torch.utils.data.DataLoader(
    dataset = data_train, batch_size = batch_size, shuffle = True
)

data_test = datasets.MNIST(
    root = './data/',
    transform = transform,
    train = False,
    download = True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset = data_test, batch_size = batch_size, shuffle = True
)

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
           
        self.dense = nn.Sequential(
            nn.Linear(14*14*128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x

MN = MnistNet()
MN = MN.to(device)
optimizer = torch.optim.Adam(MN.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

if not os.path.exists('./models'):
    os.mkdir('./models')

loss_log = []
accuracy_log = []
for epoch in range(num_epoch):
    train_loss = 0.0
    train_accuracy = 0.0
    for idx, data in enumerate(dataloader_train):
        image, label = data
        image_train, label_train = Variable(image), Variable(label)
        image_train, label_train = image_train.to(device), label_train.to(device)
        output = MN(image_train)
        _, pred = torch.max(output, 1)

        optimizer.zero_grad()
        loss = criterion(output, label_train)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_accuracy += torch.sum(label_train.data == pred)

    torch.save(MN, './models/mnist_net_{}.pth'.format(epoch+1))    
    print('Epoch:{}/{}, loss = {:.6f}, accuracy = {:.6f}'.format(
            (epoch+1), num_epoch, loss.item(), train_accuracy/len(data_train)))
    
    loss_log.append(loss.item())
    accuracy_log.append(100*train_accuracy/len(data_train))
        
test_accuracy = 0.0
for data in dataloader_test:
    image_t, label_t = data
    image_test, label_test = Variable(image_t), Variable(label_t)
    image_test, label_test = image_test.to(device), label_test.to(device)
    output_t = MN(image_test)
    _, pred_t = torch.max(output_t, 1)

    test_accuracy += torch.sum(label_test.data==pred_t)
print('test_accuracy = {:.6f}'.format(test_accuracy/len(data_test)))

figure_path = 'train_figure'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
x = [i for i in range(num_epoch)]
plt.figure(figsize=(8, 8))
plt.title('Train Loss')
plt.plot(x, loss_log)
plt.savefig(figure_path + '/loss.jpg')
plt.clf()
plt.title('Train Accuracy')
plt.plot(x, accuracy_log)
plt.savefig(figure_path + '/accuracy.jpg')
