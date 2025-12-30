import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from demoContrastiveLearning import SimCLR
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Contrastive Learning - SimCLR Testing")
    parser.add_argument("--root_dir", default="myAnimalDataset", help="Thư mục chứa dữ liệu")
    parser.add_argument("--checkpoint_path", default="demo_simclr_checkpoints/simclr_best.pth", help="Đường dẫn tới checkpoint tốt nhất")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Số luồng tải dữ liệu")
    parser.add_argument("--linear_epochs", type=int, default=30, help="Số epoch huấn luyện linear classifier")
    return parser.parse_args()

def get_pig_split(root_dir):
    pig_dir = os.path.join(root_dir, 'pig')
    images = os.listdir(pig_dir)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(pig_dir, img) for img in images if img.lower().endswith(valid_extensions)]
    random.seed(42)
    random.shuffle(image_paths)
    pig_test = image_paths[:50]
    pig_train = image_paths[50:500]
    return pig_train, pig_test

# Dataset cho train
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, pig_train=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['cat', 'dog', 'chicken', 'pig']
        self.pig_train = pig_train
        self.data, self.labels = self._load_data()

    def _load_data(self):
        data = []
        labels = []
        random.seed(42)
        for cls_idx, cls in enumerate(self.classes):
            if cls == 'pig':
                image_paths = self.pig_train
            else:
                class_dir = os.path.join(self.root_dir, 'seen_classes', 'train', cls)
                images = os.listdir(class_dir)
                valid_extensions = ('.jpg', '.jpeg', '.png')
                image_paths = [os.path.join(class_dir, img) for img in images if img.lower().endswith(valid_extensions)]
                image_paths = random.sample(image_paths, 450)
            for img_path in image_paths:
                data.append(img_path)
                labels.append(cls_idx)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Bỏ qua ảnh lỗi {img_path}")
            return self.__getitem__((idx + 1) % len(self.data))
        if self.transform:
            img = self.transform(img)
        return img, label

# Dataset cho test
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, pig_test=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['cat', 'dog', 'chicken', 'pig']
        self.pig_test = pig_test
        self.data, self.labels = self._load_data()

    def _load_data(self):
        data = []
        labels = []
        random.seed(42)
        for cls_idx, cls in enumerate(self.classes):
            if cls == 'pig':
                image_paths = self.pig_test
            else:
                class_dir = os.path.join(self.root_dir, 'seen_classes', 'test', cls)
                images = os.listdir(class_dir)
                valid_extensions = ('.jpg', '.jpeg', '.png')
                image_paths = [os.path.join(class_dir, img) for img in images if img.lower().endswith(valid_extensions)]
                image_paths = random.sample(image_paths, 50)
            for img_path in image_paths:
                data.append(img_path)
                labels.append(cls_idx)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Bỏ qua ảnh lỗi {img_path}")
            return self.__getitem__((idx + 1) % len(self.data))
        if self.transform:
            img = self.transform(img)
        return img, label

# Linear Classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=4):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Hàm đánh giá bằng Linear Evaluation
def linear_evaluation(model, train_loader, test_loader, device, epochs=30):
    print("Bắt đầu Linear Evaluation...")
    for param in model.backbone.parameters():
        param.requires_grad = False  # Đóng băng backbone
    classifier = LinearClassifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Huấn luyện Linear Classifier
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Linear Eval [Train] Epoch {epoch + 1}/{epochs}", colour="cyan")
        for img, label in train_bar:
            img, label = img.to(device), label.to(device)
            with torch.no_grad():
                features = model.backbone(img)
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f}")

    # Đánh giá trên tập test
    classifier.eval()
    all_preds = []
    all_labels = []
    test_bar = tqdm(test_loader, desc="Linear Eval [Test]", colour="yellow")
    with torch.no_grad():
        for img, label in test_bar:
            img, label = img.to(device), label.to(device)
            features = model.backbone(img)
            outputs = classifier(features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Linear Evaluation Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for cls, acc in enumerate(per_class_acc):
        print(f"Accuracy for {['cat', 'dog', 'chicken', 'pig'][cls]}: {acc:.4f}")

    # Vẽ Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['cat', 'dog', 'chicken', 'pig'],
                yticklabels=['cat', 'dog', 'chicken', 'pig'])
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix - Linear Evaluation')
    plt.savefig('confusion_matrix_simclr_FINAL.png')
    plt.show()

    return accuracy

# Hàm visualize không gian đặc trưng bằng t-SNE
def visualize_tsne(model, data_loader, device, num_samples=350):
    print("Bắt đầu t-SNE Visualization...")
    model.eval()
    features = []
    labels = []
    sample_bar = tqdm(data_loader, desc="Extracting Features for t-SNE", colour="green")
    with torch.no_grad():
        for img, label in sample_bar:
            img = img.to(device)
            feature = model.backbone(img)
            features.append(feature.cpu().numpy())
            labels.append(label.numpy())
            if len(features) * data_loader.batch_size >= num_samples:
                break
    features = np.concatenate(features, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]

    tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
    tsne_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for cls in range(4):
        indices = labels == cls
        plt.scatter(tsne_features[indices, 0], tsne_features[indices, 1],
                    label=['cat', 'dog', 'chicken', 'pig'][cls], alpha=0.5)
    plt.legend()
    plt.title('t-SNE Visualization of SimCLR Representations')
    plt.savefig('tsne_simclr_FINAL.png')
    plt.show()

# Hàm chính để chạy test
def test():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chia dữ liệu pig
    pig_train, pig_test = get_pig_split(args.root_dir)

    # Transform cho dữ liệu
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Tải model từ checkpoint
    model = SimCLR().to(device)
    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Tải model từ checkpoint: {args.checkpoint_path}")
    else:
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại {args.checkpoint_path}")

    # Tạo dataset và dataloader
    train_dataset = TrainDataset(root_dir=args.root_dir, transform=test_transform, pig_train=pig_train)
    test_dataset = TestDataset(root_dir=args.root_dir, transform=test_transform, pig_test=pig_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"Train dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"Test dataset size: {len(test_dataset)}, Batches: {len(test_loader)}")

    # Linear Evaluation
    linear_accuracy = linear_evaluation(model, train_loader, test_loader, device, epochs=args.linear_epochs)
    print(f"Final Linear Evaluation Accuracy: {linear_accuracy:.4f}")

    # t-SNE Visualization
    visualize_tsne(model, test_loader, device, num_samples=350)
    print("Đánh giá và visualize hoàn tất!")

if __name__ == "__main__":
    test()