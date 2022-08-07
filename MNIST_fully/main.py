import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from torchvision.utils import save_image

test_root = 'test_images'
images = os.listdir(test_root)
test_data = []
for x in images:
    if os.path.splitext(x)[1] == '.png':
        test_data.append(x)

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
#model = MnistNet()

for data in test_data:
    image_path = os.path.join(test_root, data)
    image_raw = Image.open(image_path)
    image = image_raw.convert('1')
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ]
    )
    image = transform(image)
    model = torch.load('models/mnist_net_1000.pth', map_location=torch.device('cpu'))
    image = torch.reshape(image, (1, 1, 28, 28))
    save_image(image, './data/change-{}.png'.format(data))
    model.eval()
    with torch.no_grad():
        output = model(image)
    _, pred = torch.max(output, 1)
    print('the recognition result of the image {} is {}'.format(data, pred))

    

