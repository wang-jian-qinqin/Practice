import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28
num_class = 10
num_epoch = 3
batch_size = 64

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

tests_dataset = datasets.MNIST(
    root='./data',
    download=False,
    transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


if __name__=='__main__':
    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(num_epoch):
        train_rights = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            net.train()
            output = net.forward(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = accuracy(output, target)
            train_rights.append(right)

            if batch_idx % 100 == 0:
                net.eval()
                val_rights = []
                for (data, target) in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = net(data)
                    right = accuracy(output, target)
                    val_rights.append(right)

                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                print("Now epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}\ttrain tights: {:.2f}%\ttest rights: {:.2f}%".format(
                    epoch,batch_idx*batch_size,len(train_loader.dataset),
                    100.*batch_idx/len(train_loader),
                    loss.data,
                    100.*train_r[0].item()/train_r[1],
                    100.*val_r[0].item()/val_r[1]
                ))

    print("***Test***")
    x=[]
    y=[]
    for i in range(10):
        a,b=datasets[i]
        x.append(a)
        y.append(b)
    x_values_tensor = torch.stack(x)
    test_output=net(x_values_tensor)
    pred_y=torch.max(test_output,1)[1].data.numpy()
    print(pred_y,'prediction')
    print(y,'real numbers')
