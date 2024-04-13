'''
Wenrui Liu
2024-4-13

use train dataset to generate new data and update model
'''
from model import WhiteBoxModel
import torch
from torch import nn
from fmnist_dataset import load_fashion_mnist
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import os
import random

# load original model
train, device, test = load_fashion_mnist("../data")
batch_size = 128
train_dataloader = DataLoader(train, batch_size=1)
test_dataloader = DataLoader(test, batch_size=batch_size)

model = WhiteBoxModel()
model.load_state_dict(torch.load('../whiteBoxTask/cnn.pth'))
dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("GPU")
else:
    print("CPU")
model = model.to(dev)
criterion = nn.CrossEntropyLoss().to(dev)

train_images = []
train_labels = []

new_train_images = []
new_train_labels = []



# get attack data
print("get attack data")
if os.path.exists("./result/attack_train.pkl"):
    with open("./result/attack_train.pkl", "rb") as f:
        data = pickle.load(f)
        train_images = data[0]
        train_labels = data[1]
    print("read from file")
else:
    total = 0
    # get 1000 correct train data
    for images,labels in train_dataloader:
        images = images.float().to(dev)
        labels = labels.to(dev)

        results = model(images)
        _, predict = torch.max(results.data, 1)
        if (predict == labels).sum() == 1:
            train_images.append(images.tolist())
            train_labels.append(labels.tolist())
            total += 1
        if total == 1000:
            break
    print("finish find 1000 correct train data")
    lr = 0.1
    max_epoch = 500
    total = 0
    correct = 0
    for i in range(len(train_images)):
        total += 1
        image = torch.tensor(train_images[i]).float().to(dev)
        image.requires_grad_(True)
        label = train_labels[i]
        print("image %d, label %d, attack label %d" % (i, label[0], (label[0]+1) % 10))
        label[0] = (label[0]+1) % 10
        label = torch.tensor(label).to(dev)
        

        for epoch in range(max_epoch):
            model.zero_grad()
            if image.grad is not None:
                image.grad.zero_()
            result = model(image)
            loss = criterion(result, label)
            loss.backward()
            image.data -= lr*image.grad

            result = model(image)
            _, predict = torch.max(result.data, 1)
            if (predict == label).sum() == 1:
                correct += 1 
                print("image %d, attack successfully" % i)
                new_train_images.append(image.tolist())
                new_train_labels.append(label.tolist())
                break
        print("finish train image %d"%i)
    with open("./result/attack_train.pkl", "wb") as f:
        pickle.dump([new_train_images, new_train_labels], f)
    print("attack rate: %f %%" % (100*correct/total))


# put train data into new train list
print("add train data into list")
for image, label in train_dataloader:
    new_train_images.append(image.tolist())
    new_train_labels.append(label.tolist())

new_train_labels = [i[0] for i in new_train_labels]

# train new model and store
print("begin train new model")
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
    # data shuffle
    combined = list(zip(new_train_images, new_train_labels))
    random.shuffle(combined)
    new_train_images, new_train_labels = zip(*combined)

    epoch_loss = 0
    total_len = len(new_train_images)
    for i in range(0, total_len, batch_size):
        images = torch.tensor(new_train_images[i: min(total_len, i+batch_size)])
        labels = torch.tensor(new_train_labels[i: min(total_len, i+batch_size)])
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
torch.save(model.state_dict(), "./update_cnn.pth")

#test new model
model.eval()
correct = 0
total = 0
for i in range(0, total_len, batch_size):
    images = torch.tensor(new_train_images[i: min(total_len, i+batch_size)])
    labels = torch.tensor(new_train_labels[i: min(total_len, i+batch_size)])
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