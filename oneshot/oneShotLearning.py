import os
import random
from PIL import Image
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.classes = ['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for cls in self.classes:
            class_directory = os.path.join(self.root_dir, 'seen_classes', 'train', cls)
            images = [os.path.join(class_directory, img) for img in os.listdir(class_directory)]
            if self.mode == 'train':
                data.extend(images[:int(0.8 * len(images))])
            else:
                data.extend(images[int(0.8 * len(images)):])
        return data

    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self, idx):
        if idx < len(self.data):
            image1_path = self.data[idx]
            class_name = os.path.basename(os.path.dirname(image1_path))
            same_class_images = [p for p in self.data if os.path.basename(os.path.dirname(p)) == class_name and p != image1_path]
            image2_path = random.choice(same_class_images) if same_class_images else image1_path
            label = 1
        else:
            image1_path = self.data[idx - len(self.data)]
            class1_name = os.path.basename(os.path.dirname(image1_path))
            different_class_images = [p for p in self.data if os.path.basename(os.path.dirname(p)) != class1_name]
            image2_path = random.choice(different_class_images)
            label = 0

        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label