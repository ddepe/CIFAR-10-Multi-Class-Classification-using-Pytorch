"""
Multilayer Perceptron approach to CIFAR-10 Multi-class classification
Using Pytorch and Cross Entropy Loss
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

class SevenFullyConnectedNet(nn.Module):
    def __init__(self):
        super(SevenFullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 2634)  # equal spacing between in/out variables
        self.fc2 = nn.Linear(2634, 2196)  # equal spacing between in/out variables
        self.fc3 = nn.Linear(2196, 1758)  # equal spacing between in/out variables
        self.fc4 = nn.Linear(1758, 1320)  # equal spacing between in/out variables
        self.fc5 = nn.Linear(1320, 882)  # equal spacing between in/out variables
        self.fc6 = nn.Linear(882, 444)  # equal spacing between in/out variables
        self.fc7 = nn.Linear(444, 10)  # equal spacing between in/out variables

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        """
        x = self.fc7(x)
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ----------------------------------------------------------------------------------------------------------------------
# neural net initialization
# ----------------------------------------------------------------------------------------------------------------------

learning_rate = 1e-2
num_epochs = 1

net = SevenFullyConnectedNet()
net.to(device)

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
filename = 'Results_' + str(learning_rate) + '_MLP_' + str(time.time()) + '.txt'
f = open(filename, 'w')
f.write('Run Start Time: ' + str(time.ctime()))
print('Learning Rate: %f' % learning_rate)
f.write('Learning Rate\t%f\n' % learning_rate)
max_accuracy = 0
for epoch in range(num_epochs):  # loop over the dataset multiple times
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

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
    print("Epoch[%d] total time taken: %f" % (epoch+1, end_time - start_time))

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
            print('===============================================')
            print('out ', outputs.data)
            print('pred ', predicted)
            print('labels ', labels)
            print('correct', correct_matrix)

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
        #                                              100 * class_correct[i] / class_total[i]))

    print('Accuracy of the network [%d/%d]: %f %%' % (correct, total, 100 * correct / total))
    f.write('Accuracy of the network [%d/%d]\t%f %%\n' % (correct, total, 100 * correct / total))
print('Finished Training: ', time.ctime())
f.write('Finished Training: ' + str(time.ctime()) + '\n')
run_time = time.time() - begin_time
print('Total Runtime: %f' % run_time)
f.write('Total Runtime\t%f\n' % run_time)
