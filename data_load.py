# data_loader.py
import os
from PIL import Image
from torchvision.transforms import transforms
import torch.utils.data as data_loader

class ImageDataset(data_loader.Dataset):
    """
    create a dataset class for image data at a given path and transforms
    """
    def __init__(self, data_path, transform=None):
        
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = os.listdir(data_path)
        for i, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.images.append(image_path)
                self.labels.append(i)

    def __len__(self):

        return len(self.images)
    
    def __getitem__(self, idx):

        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    

def get_data_loader(data_path, transform, batch_size):
    """
    create a data loader for image data at a given path and transforms
    """
    dataset = ImageDataset(data_path, transform)
    data_loader = data_loader.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
    return data_loader