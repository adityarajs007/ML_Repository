# Loading CIFAR-10 dataset
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Load the dataset with transformations applied
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Mean and standard deviation used during normalization
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

# Function to unnormalize the image
def unnormalize(img):
    img = img * std[None, None, :] + mean[None, None, :]  # Undo normalization
    return img

# Visualizing the images
for i in range(49):
    plt.subplot(7, 7, 1 + i)
    plt.axis('off')  # Turn off the axis
    image, label = trainset[i]  # Access image and label from the dataset
    image = image.numpy()  # Convert the image tensor to a NumPy array
    image = np.transpose(image, (1, 2, 0))  # Change the order of dimensions (C, H, W) -> (H, W, C)
    image = unnormalize(image)  # Unnormalize the image
    plt.imshow(image)

plt.show()
