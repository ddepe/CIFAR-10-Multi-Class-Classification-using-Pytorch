"""
Convolutional Neural Net approach to CIFAR-10 Multi-class classification
Using Pytorch and Cross Entropy Loss
Greater than 85% accuracy with the constraint of no Maxpooling layers
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as td
import random, time
import torchvision


class SevenConvolutionalNet(nn.Module):
    def __init__(self):
        super(SevenConvolutionalNet, self).__init__()
        # (num_2d arrays, num_filters, kernel size)
        self.conv1 = nn.Conv2d(3, 66, 5)  # in: 3x32x32 out: 66x28x28
        self.conv2 = nn.Conv2d(66, 128, 5, stride=2, padding=1)  # in: 66x28x28 out:128x13x13
        self.conv3 = nn.Conv2d(128, 192, 5)  # in: 128x13x13 out:192x9x9
        self.conv4 = nn.Conv2d(192, 256, 5, stride=2, padding=1)  # in: 192x9x9 out:256x4x4
        self.fc1 = nn.Linear(4096, 2734)  # equal spacing between in/out variables for FC layers
        self.fc2 = nn.Linear(2734, 1372)  # equal spacing between in/out variables for FC layers
        self.fc3 = nn.Linear(1372, 10)  # equal spacing between in/out variables for FC layers

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.fc2(x)
        """
        x = self.fc3(x)
        return x


def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, 4),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    test = datasets.CIFAR10('./', train=False,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


batch_size = 64
test_batch_size = 64

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

lr_list = [1e-2]
num_epochs = 100
for learning_rate in lr_list:

    # ----------------------------------------------------------------------------------------------------------------------
    # neural net initialization
    # ----------------------------------------------------------------------------------------------------------------------

    net = SevenConvolutionalNet()

    # ----------------------------------------------------------------------------------------------------------------------
    # Loss function and optimizer
    # ----------------------------------------------------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1, verbose=True, cooldown=10)

    # ----------------------------------------------------------------------------------------------------------------------
    # Train the network
    # ----------------------------------------------------------------------------------------------------------------------

    print('Run Start Time: ', time.ctime())
    begin_time = time.time()
    filename = 'Results_' + str(learning_rate) + '_CNN_' + str(time.time()) + '.txt'
    f = open(filename, 'w')
    f.write('Run Start Time: ' + str(time.ctime()))
    print('Learning Rate: %f' % learning_rate)
    f.write('Learning Rate\t%f\n' % learning_rate)
    max_accuracy = 0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0
        # scheduler.step()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass + Optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('Epoch[% d/% d]: Loss: %.4f' % (epoch + 1, num_epochs, running_loss / (i + 1)))
        f.write("Epoch\t%d\tLoss\t%f\t" % (epoch + 1, running_loss / (i + 1)))
        end_time = time.time()
        print("Epoch[%d] total time taken: %f" % (epoch + 1, end_time - start_time))

        # ------------------------------------------------------------------------------------------------------------------
        # Test the network
        # ------------------------------------------------------------------------------------------------------------------

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct_matrix = (predicted == labels)
                c = correct_matrix.squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                total += labels.size(0)
                correct += correct_matrix.sum().item()
        max_accuracy = max(max_accuracy, int(100 * correct / total))
        scheduler.step(max_accuracy)
        for i in range(10):
            print('Accuracy of %5s [%d/%d]: %2f %%' % (classes[i], class_correct[i], class_total[i],
                                                       100 * class_correct[i] / class_total[i]))
            # f.write('Accuracy of %5s [%d/%d]\t%2f %%\n' % (classes[i], class_correct[i], class_total[i],
            #                                                100 * class_correct[i] / class_total[i]))

        print('Accuracy of the network [%d/%d]: %f %%' % (correct, total, 100 * correct / total))
        f.write('Accuracy of the network [%d/%d]\t%f %%\n' % (correct, total, 100 * correct / total))
    print('Finished Training: ', time.ctime())
    f.write('Finished Training: ' + str(time.ctime())+'\n')
    run_time = time.time() - begin_time
    print('Total Runtime: %f' % run_time)
    f.write('Total Runtime\t%f\n' % run_time)

