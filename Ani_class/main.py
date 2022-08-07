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

data_class = ['cat', 'dog', 'wild']

class AniNet(nn.Module):
    def __init__(self):
        super(AniNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
           
        self.dense = nn.Sequential(
            nn.Linear(1024*256, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 3)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.dense(x)
        return x

for data in test_data:
    image_path = os.path.join(test_root, data)
    image = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ]
    )
    image_raw = transform(image)
    image = image_raw.convert('3')
    model = torch.load('models/aninet_net_1000.pth')
    image = torch.reshape(image, (1, 1, 64, 64))
    #save_image(image, './data/change-{}.png'.format(data))
    model.eval()
    with torch.no_grad():
        output = model(image)
    pred = data_class[int(output.argmax(1))]
    print('the recognition result of the image {} is {}'.format(data, pred))

    

