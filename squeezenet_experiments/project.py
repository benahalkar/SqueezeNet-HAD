
import os
os.chdir("/home/hb2776/mathDL_project")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

from utils import models, display, train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("GPU name: ", torch.cuda.get_device_name(0))



# define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# loading the dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Get class names from the dataset
classes = train_dataset.classes
num_classes = len(classes)

# Call the function with the train_dataset
display.plot_random_images(train_dataset, classes, "sample_images.png")

# defining the data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
display_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

data_save_path = os.path.join(os.getcwd(), "results.txt")
weights_path = os.path.join(os.getcwd(), "weights", "squeezenet1_0-b66bff10.pth")

# defining the epochs
epochs = 10

# defining the loss function
criterion = nn.CrossEntropyLoss()


def run(model, optimizer, base_folder, restarts, add_noise):
    
    # train the model
    folder_path = os.path.join(os.getcwd(), base_folder)
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    model_save_path = os.path.join(folder_path, f"model_{model.classifier[2].__class__.__name__}.pth")
    predictions_path = os.path.join(folder_path, "predictions_labels.png")
    loss_path = os.path.join(folder_path, "train_val_loss.png")
    accuracy_path = os.path.join(folder_path, "train_val_accuracy.png")
    layer_weights_path = os.path.join(folder_path, "weights.png")
    
    
    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, epoch_times = train.train_model(model, device, optimizer, criterion, train_loader, test_loader, epochs, restarts, model_save_path, add_noise)
    
    # viewing some of the predictions    
    display.visualize_predictions_from_dataset(model, display_loader, device, classes, predictions_path)


    # create images to plot losses and accuracies 
    display.plot_loss_and_accuracy(train_loss_history, val_loss_history, "Training and Validation loss", optimizer, criterion, epochs, accuracy_path)

    display.plot_loss_and_accuracy(train_accuracy_history, val_accuracy_history, "Training and Validation accuracy", optimizer, criterion, epochs, loss_path)
    
    # print the weights
    weights = model.classifier[1].weight.detach().cpu()
    weights = torch.squeeze(weights, dim=-1)
    weights = torch.squeeze(weights, dim=-1)
    display.plot_weight(weights, layer_weights_path)
    
    # saving all progress
    with open(data_save_path, "a") as file:
        file.write(f"Model Activation: {model.classifier[2].__class__.__name__}" + '\n')
        file.write(f"Optimizer name: {optimizer.__class__.__name__}" + '\n')
        file.write(f"Optimizer value: {optimizer.param_groups[0]['lr']}" + '\n')
        file.write(f"Criterion value: {criterion.__class__.__name__}" + '\n')
        file.write("train loss history: " + ' '.join(map(str, train_loss_history)) + '\n')
        file.write("train accuracy history: " + ' '.join(map(str, train_accuracy_history)) + '\n')
        file.write("val loss history: " + ' '.join(map(str, val_loss_history)) + '\n')
        file.write("val accuracy history: " + ' '.join(map(str, val_accuracy_history)) + '\n')
        file.write("epoch time history: " + ' '.join(map(str, epoch_times)) + '\n')
        file.write("classifier weights: " + '\n')
        for row in weights.numpy():
            file.write(' '.join(map(str, row)) + '\n')
        file.write('\n')
        file.close()

    return None

"""
Experiments run
1. Adam optimizer
2. SGD optimizer
3, SGD optimizer with momentum (Accelerated Gradient Descent)
4. Perturbing Gradient Descent
5. SGD optimizer with restarts

1. Relu activation
2. Leaky Relu activation
3. TanH activation

"""


# define the model

model = models.build_model(num_classes, weights_path, "relu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
run(model, optimizer, "adam", restarts=1, add_noise=False)

model = models.build_model(num_classes, weights_path, "relu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
run(model, optimizer, "sgd", restarts=1, add_noise=False)

model = models.build_model(num_classes, weights_path, "relu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
run(model, optimizer, "agd", restarts=1, add_noise=False)

model = models.build_model(num_classes, weights_path, "relu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
run(model, optimizer, "pgd", restarts=1, add_noise=True)

model = models.build_model(num_classes, weights_path, "relu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
run(model, optimizer, "sgdr", restarts=3, add_noise=False)

# >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>

model = models.build_model(num_classes, weights_path, "leaky_relu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
run(model, optimizer, "adam", restarts=1, add_noise=False)

model = models.build_model(num_classes, weights_path, "leaky_relu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
run(model, optimizer, "sgd", restarts=1, add_noise=False)

model = models.build_model(num_classes, weights_path, "leaky_relu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
run(model, optimizer, "agd", restarts=1, add_noise=False)

model = models.build_model(num_classes, weights_path, "leaky_relu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
run(model, optimizer, "pgd", restarts=1, add_noise=True)

model = models.build_model(num_classes, weights_path, "leaky_relu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
run(model, optimizer, "sgdr", restarts=3, add_noise=False)

# >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>

model = models.build_model(num_classes, weights_path, "tanh")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
run(model, optimizer, "adam", restarts=1, add_noise=False)

model = models.build_model(num_classes, weights_path, "tanh")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
run(model, optimizer, "sgd", restarts=1, add_noise=False)

model = models.build_model(num_classes, weights_path, "tanh")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
run(model, optimizer, "agd", restarts=1, add_noise=False)

model = models.build_model(num_classes, weights_path, "tanh")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
run(model, optimizer, "pgd", restarts=1, add_noise=True)

model = models.build_model(num_classes, weights_path, "tanh")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
run(model, optimizer, "sgdr", restarts=3, add_noise=False)
