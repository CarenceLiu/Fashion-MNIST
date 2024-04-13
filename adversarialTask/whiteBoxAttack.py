'''
Wenrui Liu
2024-4-13

white box attack
'''
from model import WhiteBoxModel
import torch
from torch import nn
from fmnist_dataset import load_fashion_mnist
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import os


train, dev, test = load_fashion_mnist("../data")
test_dataloader = DataLoader(test, batch_size=1)

model = WhiteBoxModel()
model.load_state_dict(torch.load('./update_cnn.pth'))
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

# get correct classification data
if os.path.exists("./result/correct_test.pkl"):
    with open("./result/correct_test.pkl", "rb") as f:
        data = pickle.load(f)
        train_images = data[0]
        train_labels = data[1]
    print("read from file")
else:
    total = 0
    for images,labels in test_dataloader:
        images = images.float().to(dev)
        labels = labels.to(dev)

        results = model(images)
        _, predict = torch.max(results.data, 1)
        if (predict == labels).sum() == 1:
            train_images.append(images.tolist())
            train_labels.append(labels.tolist())
            total += 1
        if total == 1000:
            with open("./result/correct_test.pkl", "wb") as f:
                pickle.dump([train_images,train_labels], f)
            break
    print("create file")


lr = 0.1
max_epoch = 500
total = 0
correct = 0
attack_images = []
attack_labels = []
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
            if correct < 10:
                plt.imsave("./result/white_fmnist_%d_origin_label_%d.jpg" % (correct, train_labels[i][0]), torch.tensor(train_images[i]).float().reshape([28, 28]))
                plt.imsave("./result/white_fmnist_%d_attack_label_%d.jpg" % (correct, (train_labels[i][0]+1)%10), image.detach().cpu().reshape([28, 28]))
            correct += 1 
            print("image %d, attack successfully" % i)
            attack_images.append(image.tolist())
            attack_labels.append(label.tolist())
            break
    print("finish train image %d"%i)

with open("./result/attack_test.pkl", "wb") as f:
    pickle.dump([attack_images,attack_labels], f)
print("attack rate: %f %%" % (100*correct/total))


    