'''
Wenrui Liu
2024-4-13

use attack data to attack black model
'''
from model import BlackBoxModel
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
import os
import random

model = BlackBoxModel()
model.load_state_dict(torch.load('../model/cnn.ckpt'))
dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("GPU")
else:
    print("CPU")
model = model.to(dev)

train_images = []
train_labels = []

# get correct classification data
if os.path.exists("../attack_data/correct_1k.pkl"):
    with open("../attack_data/correct_1k.pkl", "rb") as f:
        data = pickle.load(f)
        train_images = data[0]
        train_labels = data[1]
    print("read from file")
else: 
    print("Error path")
    exit(1)


total = len(train_images)
correct = 0
success_images = []
success_labels = []
max_epoch = 2500
delta_max = 0.5
sigma = 0.1

# markov model
model.eval()
for i in range(len(train_images)):
    image = torch.tensor([train_images[i]]).float().to(dev)
    label = [train_labels[i].item()]
    aim_label = [(label[0]+1)%10]
    label = torch.tensor(label).to(dev)
    aim_label = torch.tensor(aim_label).to(dev)
    new_prob = 0
    for epoch in range(max_epoch):
        image_bias = torch.randn_like(image).to(dev)*sigma
        new_image = image+image_bias
        if torch.max(image_bias.data) > 0.5:
            continue

        result = model(new_image)
        result = torch.nn.functional.softmax(result, dim = 1)
        if result[0, aim_label] >= new_prob:
            image = new_image
            new_prob = result[0, aim_label]
        _, predict = torch.max(result.data, 1)
        if (predict == aim_label).sum() == 1:
            correct += 1
            if correct < 10:
                plt.imsave("./result/fmnist_%d_origin_label_%d.jpg" % (correct, train_labels[i]), torch.tensor(train_images[i]).float().reshape([28, 28]))
                plt.imsave("./result/fmnist_%d_attack_label_%d.jpg" % (correct, (train_labels[i]+1)%10), new_image.detach().cpu().reshape([28, 28]))
            success_images.append([train_images[i]])
            success_labels.append([(train_labels[i]+1)%10])
            print("successfully attack %d, label %d, aim label %d"%(correct, train_labels[i], (train_labels[i]+1)%10))
            break
    print("finish image %d"%(i))
with open("./result/attack_success_test.pkl", "wb") as f:
    pickle.dump([success_images, success_labels], f)
print("attack rate: %f %%" % (100*correct/total))