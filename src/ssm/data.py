from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class SMNIST(Dataset):
    def __init__(self, train: bool = True):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.MNIST(root='./data', train=train, download=True, transform=self.transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img.flatten(), label

if __name__ == "__main__":
    SMNIST = SMNIST()
    img, label = SMNIST[0]
    print(img.shape, label)