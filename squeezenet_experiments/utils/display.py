import numpy as np
import matplotlib.pyplot as plt
import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)

from .train import do_inference, do_transpose

def plot_random_images(dataset, classes, save_path, num_images=16):
    # Set up a figure of adequate size
    plt.figure(figsize=(10, 10))

    # Get random indices
    indices = np.random.choice(len(dataset), size=num_images, replace=False)

    for i, idx in enumerate(indices):
        # Get image and label
        image, label = dataset[idx]

        # Unnormalize the image
        image = image / 2 + 0.5

        # Convert tensor to numpy array and transpose to (H, W, C)
        image = image.numpy().transpose((1, 2, 0))

        # Plot the image
        plt.subplot(4, 4, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {classes[label]}")
        plt.axis('off')

    # plt.show()
    plt.savefig(save_path, dpi=500)

    
def visualize_predictions_from_dataset(model, data_loader, device, classes, save_path):
    plt.figure(figsize=(10, 10))


    for i, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)

        output = do_inference(model, images)
        output = output.cpu()[0]

        images = images.cpu()[0]
        images = do_transpose(images)

        labels = int(labels.cpu().item())

        # Plot the image
        plt.subplot(4, 4, i + 1)
        plt.imshow(images)
        plt.title(f"Actual: {classes[labels]}\nPredicted: {classes[output]}", color='green' if output == labels else 'red')
        plt.axis('off')

        if i == 15: break

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.show()
    plt.savefig(save_path, dpi=500)

    
def plot_loss_and_accuracy(train, val, title, optimizer, criterion, epochs, save_path):
    plt.figure(figsize=(10, 5))

    plt.plot(train, color="blue", label="Training")
    plt.plot(val, color="orange", label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"{title} // optim. {optimizer.__class__.__name__} lr={optimizer.param_groups[0]['lr']} // crit. {criterion.__class__.__name__}")
    # plt.show()
    plt.savefig(save_path, dpi=500)

    
def plot_weight(weights, save_path):
    plt.figure(figsize=(20, 20))
    
    for i in range(weights.size(0)):
        plt.plot(weights[i])

    # plt.axis('off')
    # plt.title(f'Filter {i}')
    
    # plt.show()
    plt.savefig(save_path, dpi=500)