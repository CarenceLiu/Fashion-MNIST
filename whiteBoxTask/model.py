'''
Wenrui Liu
2024-4-11

CNN Model for Fashion-MNIST classification task
'''

import torch
from torch import nn

class WhiteBoxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 24, 5, 1, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 5, 2, 2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, 5, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5*5*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
    
    def forward(self, x):
        x = x.reshape((x.size(0), 1, 28, 28))
        return self.layers(x)

if __name__ == "__main__":
    from fmnist_dataset import load_fashion_mnist
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    train, dev, test = load_fashion_mnist("../data")
    batch_size = 128
    train_dataloader = DataLoader(train, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size)

    model = WhiteBoxModel()
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("GPU")
    else:
        print("CPU")
    model = model.to(dev)
    criterion = nn.CrossEntropyLoss().to(dev)
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_epoch = 70
    train_loss = []
    for epoch in range(total_epoch):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.float().to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            results = model(images)
            loss = criterion(results, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().data.item()
        train_loss.append(epoch_loss/(i+1))

        print("Epoch: %d"%(epoch))
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_loss)
    plt.show()
    
    torch.save(model.state_dict(), "./cnn.pth")

    model.eval()
    correct = 0
    total = 0
    for images, labels in train_dataloader:
        images = images.float().to(dev)
        labels = labels.to(dev)

        results = model(images)
        _, predict = torch.max(results.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum()
    print("train acc: %f %%"%(100*correct/total))

    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images = images.float().to(dev)
        labels = labels.to(dev)

        results = model(images)
        _, predict = torch.max(results.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum()
    print("test acc: %f %%"%(100*correct/total))
