import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.autograd import Variable

batch_size = 64
num_epoch = 1000

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
        self.fc1 = nn.Sequential(
            nn.Linear(28*28, 28),
            nn.ReLU(),
        )        
        self.fc2 = nn.Linear(28, 10)   
    def forward(self, x):
        x = x.view(-1, 28*28*1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

MN = MnistNet()
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
