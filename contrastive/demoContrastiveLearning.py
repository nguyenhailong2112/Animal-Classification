import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset

def get_simclr_transform():
    transform_list = [
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=11, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)

class SimCLRDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, val_split=0.1):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.classes = ['cat', 'dog', 'chicken', 'pig']
        self.data = self._load_data()
        self._split_data(val_split)

    def _load_data(self):
        data = []
        random.seed(42)
        for cls in self.classes:
            if cls == 'pig':
                class_dir = os.path.join(self.root_dir, 'pig')
            else:
                class_dir = os.path.join(self.root_dir, 'seen_classes', 'train', cls)
            images = os.listdir(class_dir)
            valid_extensions = ('.jpg', '.jpeg', '.png')
            image_paths = [os.path.join(class_dir, img) for img in images if img.lower().endswith(valid_extensions)]
            data.extend(image_paths)
        return data

    def _split_data(self, val_split):
        random.shuffle(self.data)
        val_size = int(len(self.data) * val_split)
        if self.mode == 'train':
            self.data = self.data[val_size:]
        elif self.mode == 'val':
            self.data = self.data[:val_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Bỏ qua ảnh lỗi {img_path}")
            return self.__getitem__((idx + 1) % len(self.data))
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        return img1, img2

# SimCLR Model
class SimCLR(nn.Module):
    def __init__(self, backbone=None, projection_dim=256):
        super(SimCLR, self).__init__()
        if backbone is None:
            backbone = resnet50(weights='IMAGENET1K_V1')
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return nn.functional.normalize(projections, dim=1)

# NTXentLoss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        labels = torch.arange(batch_size)
        labels = torch.cat([labels + batch_size, labels], dim=0).to(z.device)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        loss = self.criterion(sim_matrix, labels)
        return loss