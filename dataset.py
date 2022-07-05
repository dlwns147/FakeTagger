import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
import cv2

class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
        # print(f'len : {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image = Image.open(self.images[idx])
        image = Image.fromarray(cv2.imread(self.images[idx]))
        # image = cv2.imread(self.images[idx])
        
        if self.transform:
            image = self.transform(image)

        return image
    
def split_dataset(path, train_transform = None, test_transform = None, val_ratio = 0.2, test_ratio = 0.2) :
    images = [x.path for x in os.scandir(path) if x.name.endswith(".jpg") or x.name.endswith(".png")]
    total_len = len(images)
    train_images = images[: int(total_len * (1 - val_ratio - test_ratio))]
    val_images = images[int(total_len * (1 - val_ratio - test_ratio)): int(total_len * (1 - test_ratio))]
    test_images = images[int(total_len * (1- test_ratio)): ]
    return CustomDataset(train_images, train_transform), CustomDataset(val_images, train_transform), CustomDataset(test_images, test_transform)
    
    