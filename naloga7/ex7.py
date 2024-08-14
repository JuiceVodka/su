import math

import torch
import numpy as np
from fashion_mnist_master.utils.mnist_reader import load_mnist
import matplotlib.pyplot as plt


#A) data acquirement and initial visualization

mnist_path = 'fashion_mnist_master/data/fashion'

images_train, labels_train = load_mnist(mnist_path, kind='train')

#normalize the training sample values to the range [0, 1]
images_train = images_train / 255

#visualise one random member from each of the 10 classes on a subplot with 2 rows and 5 columns

for i in range(10):
    #get the first image of class i
    img = images_train[labels_train == i][0]
    #reshape it to 28x28
    img = img.reshape(28, 28)
    #plot it
    plt.subplot(2, 5, i+1)
    plt.imshow(img)

    #title the plot with the class name
    plt.title('Class ' + str(i))

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()


#B) data preparation
#split the training data into training and validation sets, where the validation sets has 20% of the data
#Reshape the sample data in both subsets such that each sample corresponds to a two-dimensional image with a single channel.
#The dimensions of each subset should be equal to [N, 1, 28, 28], where N denotes the total number of samples in the subset.
# Convert both the samples and the labels of both subsets to PyTorch Tensors, where you additionally transform the class labels to one-hot encodings.

#random shuffle the data
np.random.seed(42)
random_indices = np.random.permutation(len(images_train))
images_train = images_train[random_indices]
labels_train = labels_train[random_indices]

#train, validation split
train_size = int(0.8 * len(images_train))
validation_size = len(images_train) - train_size

#split the data
images_validation = images_train[train_size:]
images_train = images_train[:train_size]
labels_validation = labels_train[train_size:]
labels_train = labels_train[:train_size]

#reshape the data
images_train = images_train.reshape(-1, 1, 28, 28)
images_validation = images_validation.reshape(-1, 1, 28, 28)

#convert to tensors
images_train = torch.from_numpy(images_train).float()
images_validation = torch.from_numpy(images_validation)

#one-hot encoding
labels_train = torch.nn.functional.one_hot(torch.from_numpy(labels_train).long(), num_classes=10).float()
labels_validation = torch.nn.functional.one_hot(torch.from_numpy(labels_validation).long(), num_classes=10).float()

#Create the dateset class
#The Dataset class will serve batched, randomly shuffled samples during training and validation

class Dataset():
    def __init__(self, samples, labels, batch_size):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.arange(samples.shape[0])

    def __getitem__(self, index):
        start_ix = index * self.batch_size
        end_ix = min((index + 1) * self.batch_size, self.samples.shape[0])
        batch_indices = self.indices[start_ix:end_ix]
        batch_samples = self.samples[batch_indices]
        batch_labels = self.labels[batch_indices]
        return batch_samples, batch_labels

    def __len__(self):
        return int(math.ceil(self.samples.shape[0]/self.batch_size))

    def shuffle(self):
        np.random.shuffle(self.indices)


#training and validation datasets
train_dataset = Dataset(images_train, labels_train, batch_size=16)
validation_dataset = Dataset(images_validation, labels_validation, batch_size=16)

#Visualize 16 samples from a batch belonging to the training dataset and their corresponding class labels.
training_samples, training_labels = train_dataset[0]
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(training_samples[i][0])
    plt.title('Class ' + str(torch.argmax(training_labels[i]).item()))
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()

validation_samples, validation_labels = validation_dataset[0]
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(validation_samples[i][0])
    plt.title('Class ' + str(torch.argmax(validation_labels[i]).item()))
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()

#Create new datasets with batch size 128, the above ones are here just for ease of visualization
train_dataset = Dataset(images_train, labels_train, batch_size=128)
validation_dataset = Dataset(images_validation, labels_validation, batch_size=128)


#C)
#Now, we will implement 3 basic operations that will serve as the backbone of our classification neural network.
#These operations are:
#two-dimensional convolution
#two-dimensional max pooling
#element wise Linear Rectified Unit
#do not use torch implementations of these operations when implementing them

class Conv2d(torch.nn.Module):
    #for the purposes of this exercise i am going to assume square kernels, if they were not, kernel_size would be a tuple
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        #initialize the weights and biases
        self.weights = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.randn(out_channels), requires_grad=True)

        #define the stride and padding
        self.stride = stride

    def forward(self, x):
        #get the dimensions of the input
        #print(x.shape)
        batch_size, in_channels, in_height, in_width = x.shape

        #calculate the output dimensions
        out_height = int((in_height - self.kernel_size) / self.stride) + 1
        out_width = int((in_width - self.kernel_size) / self.stride) + 1

        #initialize the output
        out = torch.zeros(batch_size, self.out_channels, out_height, out_width)

        #perform the convolution operation without for loops
        #hint: use the torch.nn.functional.unfold() function and the torch.nn.functional.fold() function
        x_unfolded = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        #print(x_unfolded.shape)
        #out = torch.matmul(self.weights.view(self.out_channels, -1), x_unfolded) + self.bias.view(self.out_channels, 1)

        out = x_unfolded.transpose(1,2).matmul(self.weights.view(self.weights.shape[0], -1).t()).transpose(1, 2) + self.bias.view(self.out_channels, 1)
        #print(out.shape)
        #print(self.bias.view(self.bias.shape[0], 1, 1).shape)
        out = torch.nn.functional.fold(out, output_size=(out_height, out_width), kernel_size=1)
        #print(out.shape)
        #print(out.shape)
        #print("--------")

        return out

class MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        #get the dimensions of the input
        batch_size, in_channels, in_height, in_width = x.shape

        #calculate the output dimensions
        #out_height = int((in_height - self.kernel_size) / self.kernel_size) + 1
        #out_width = int((in_width - self.kernel_size) / self.kernel_size) + 1
        out_height = int((in_height - self.kernel_size) / self.stride) + 1
        out_width = int((in_width - self.kernel_size) / self.stride) + 1

        #initialize the output
        out = torch.zeros(batch_size, in_channels, out_height, out_width)

        """x_unfolded = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        #swap 2nd and 3rd dimension
        out_initial, _ = torch.max(x_unfolded, dim=1)
        print("*******")
        print(self.kernel_size)
        print(x.shape)
        print(x_unfolded.shape)
        print(out_initial.shape)
        #add back the missing dimension
        out = torch.nn.functional.fold(out_initial, output_size=(out_height, out_width), kernel_size=1)
        print(out.shape)
        print("*******")"""
        #print("*******")
        # Unfold the input
        x_unfolded = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        #print(x.shape)
        #print(x_unfolded.shape)
        # Reshape the unfolded tensor to separate the channel and kernel size dimensions
        x_unfolded = x_unfolded.view(batch_size, in_channels, self.kernel_size*self.kernel_size, -1)
        #print(x_unfolded.shape)
        # Apply max pooling
        out_initial, _ = torch.max(x_unfolded, dim=2)
        #print(out_initial.shape)
        # Reshape the result back to the desired output shape
        out = out_initial.view(batch_size, in_channels, out_height, out_width)
        #print(out.shape)

        #print("*******")
        return out

class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.max(x, torch.zeros_like(x))
        return out


#D)
#Now, we will define a neural network using the custom functions from previous exercises.
#This network will produce a vector of 10 elements for each sample, predicting the class it belongs to
#Our neural network will consist of the following basic blocks:
#• Conv2d followed by MaxPool2d followed by ReLU
#
#There should be two such blocks in your network, one following the other.
#The kernel size of both the convolution and maximal pooling layers in both blocks should be equal to 3 (stride = 1)
#The output of the second block should be a tensor with the following dimensions:
#[B, 64, 4, 4], where B is the batch size and 64 is the number of channels
#
#Finally, your model should end with a single convolution layer with a kernel_size of 4 which will produce 10 output channels
#The output of this layer should therefore be a tensor with a dimension equal to [B, 10, 1, 1].
#Apply a flatten operation (torch.nn.Flatten) on the last layer’s output to generate the output of size: [B, 10].

class MyConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            Conv2d(1, 64, kernel_size=3),
            MaxPool2d(kernel_size=3),
            ReLU(),
            Conv2d(64, 64, kernel_size=3),
            MaxPool2d(kernel_size=3),
            ReLU(),
            Conv2d(64, 10, kernel_size=4),
            torch.nn.Flatten())


    def forward(self, x):
        out = self.model(x)
        return out


#Create the same model using the inbuilt PyTorch functions.
#The model should have the same architecture as the one you defined in the previous exercise.

class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3,3)),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(3,3)),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 10, kernel_size=(4,4)),
            torch.nn.Flatten())

    def forward(self, x):
        out = self.model(x)
        return out


#E)
#For this exercise, use the models you constructed in the previous exercises.
#Lets put it all together! Using the Stochastic Gradient Descent algorithm and the Cross-Entropy loss function
#define a function fit, which takes in two parameters: the model you wish to fit and the number of epochs,
#denoting the number of times you will iterate over the training and validation datasets.

def fit(model, number_of_epochs):
    losses_train = np.zeros(number_of_epochs)
    losses_val = np.zeros(number_of_epochs)
    best_model = None
    best_accuracy = 0.0
    #define the loss function and the optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    cur_loss_train = []
    cur_loss_val = []
    #iterate over the epochs
    for epoch in range(number_of_epochs):
        #optimizer.lr = optimizer.lr * 0.95
        #iterate over the training dataset
        print(len(train_dataset))
        for i in range(len(train_dataset)):
            x, y = train_dataset[i]
            x = x.float()
            result = model(x)

            #set value of result across axis 1 to 1 for the correct class and 0 for the rest
            #rez_one_hot = torch.zeros_like(result).float()
            #max_indices = torch.argmax(result, dim=1)
            #rez_one_hot[:, max_indices] = 1.0


            #calculate the loss
            #loss = loss_function(rez_one_hot, y).float()
            loss = loss_function(result, y)
            cur_loss_train.append(loss.item())

            #zero the gradients
            optimizer.zero_grad()

            #calculate the gradients
            loss.backward()

            #update the weights
            optimizer.step()

            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch + 1, i, loss.item()), end="\n")
        losses_train[epoch] = np.mean(cur_loss_train)

        #evaluate the model on the validation dataset
        with torch.no_grad():
            correct = 0
            total = 0
            for j in range(len(validation_dataset)):
                x, y = validation_dataset[j]
                x = x.float()
                #get the predictions
                predictions = model(x)

                #get the predicted class
                _, predicted = torch.max(predictions.data, 1)

                #print(predicted.shape)
                #print(predicted[:, None].shape)
                #print(y.shape)

                #update the total number of samples
                total += y.shape[0]
                #print(total)

                #update the number of correctly classified samples
                #change y from one hot to class
                y_class = torch.argmax(y, dim=1)
                correctly_classified = (predicted == y_class).sum().item()
                correct += correctly_classified
                #print(correctly_classified)

                #correct += (predicted[:, None] == y).sum().item()
                #print(correct)
                #print("---------")

                cur_loss_val.append(loss_function(predictions, y).item())

            losses_val[epoch] = np.mean(cur_loss_val)

            #calculate the accuracy
            accuracy = correct / total

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

            #print the accuracy
            print(f'Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}')


    return best_model, losses_train, losses_val


#best_model_mine, losses_train_mine, losses_val_mine = fit(MyConvNet(), 10)
#best_model, losses_train, losses_val = fit(ConvNet(), 10)

#pytorch model converges better since it has better starting weights, we could fix this by setting starting weights to the same weights as the pytorch model

#compare results of my conv2d layer and pytorch conv2d layer on simple inputs
#also compare relu and maxpool2d layers with correspondign pytorch layers

sample_data = torch.tensor([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11 ,12, 13, 14, 15], [-16, -17, -18, -19, -20], [21, 22, 23, 24, 25]]]])
sample_data = sample_data.float()
print(sample_data.shape)

my_conv2d = Conv2d(1, 1, kernel_size=3)
my_conv2d.weights = torch.nn.Parameter(torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).float())
my_conv2d.bias = torch.nn.Parameter(torch.tensor([0.0]).float())
print(my_conv2d(sample_data))


pytorch_conv2d = torch.nn.Conv2d(1, 1, kernel_size=3)
pytorch_conv2d.weight = torch.nn.Parameter(torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).float())
pytorch_conv2d.bias = torch.nn.Parameter(torch.tensor([0.0]).float())
print(pytorch_conv2d(sample_data))

my_relu = ReLU()
print(my_relu(sample_data))

pytorch_relu = torch.nn.ReLU()
print(pytorch_relu(sample_data))

my_maxpool2d = MaxPool2d(kernel_size=3, stride=2)
print(my_maxpool2d(sample_data))

pytorch_maxpool2d = torch.nn.MaxPool2d(kernel_size=3, stride=2)
print(pytorch_maxpool2d(sample_data))

"""
#plot the loss curves for the training and validation datasets
plt.plot(losses_train_mine, label="Training loss")
plt.plot(losses_val_mine, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(losses_train, label="Training loss")
plt.plot(losses_val, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
"""

#TODO:
#For this exercise, use the fit function and dataset class you constructed in the previous exercises.
#Create a custom neural network model using PyTorch functions and fit the model to
#the training data. Optimize the neural network hyperparameters (number of layers,
#kernel size, etc...) to get the best results on the validation dataset.
#Evaluate your prediction on the validation dataset using the performance.plot_stats
#function (included in the assignment materials) which takes as input your model’s
#predictions and the corresponding correct labels (both in one-hot encoding format
#and numpy ndarrays). The output of the function is the image titled stats.png, containing the class confusion matrix,
#and the macro recall and macro precision values for each class.

class Lenet_5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stack1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=(5,5)),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=(5,5)),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.stack2 = torch.nn.Sequential(
            torch.nn.Linear(400, 120),
            torch.nn.Linear(120, 84),
            torch.nn.Linear(84, 10),
            torch.nn.Softmax(dim=1)
        )
        """self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=(5,5)),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=(5,5)),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Linear(400, 120),
            torch.nn.Linear(120, 84),
            torch.nn.Linear(84, 10),
            torch.nn.Softmax(dim=1)
        )"""

    def forward(self, x):
        #out = self.model(x)
        out1 = self.stack1(x)
        out1 = out1.view(-1, 400)
        out = self.stack2(out1)
        return out


# Defining the convolutional neural network
class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten())
        self.fc = torch.nn.Linear(256, 120)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(120, 84)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(84, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

best_model, losses_train, losses_val = fit(LeNet5(), 10)

#plot the loss curves for the training and validation datasets
plt.plot(losses_train, label="Training loss")
plt.plot(losses_val, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


