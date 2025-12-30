import cv2
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

def split_dataset(source_dir, dest_dir, test_size=0.2, random_state=42):
    # Tạo thư mục train và test
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    total_original = 0
    total_train = 0
    total_test = 0

    test_subfolder = []

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            print("Phân chia Class:", class_name)

            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            file_list = os.listdir(class_path)
            count_original = len(file_list)
            total_original += count_original

            train_files, test_files = train_test_split(file_list, test_size=test_size, random_state=random_state)
            total_train += len(train_files)
            total_test += len(test_files)

            for file in train_files:
                src_file = os.path.join(class_path, file)
                dst_file = os.path.join(train_class_dir, file)
                shutil.copy(src_file, dst_file)

            for file in test_files:
                src_file = os.path.join(class_path, file)
                dst_file = os.path.join(test_class_dir, file)
                shutil.copy(src_file, dst_file)

            print(f"Class {class_name}: Train: {len(train_files)}, Test: {len(test_files)}")

        for file in test_files:
            test_subfolder.append({
                'filename': file,
                'label': class_name
            })

    pd.DataFrame(test_subfolder).to_csv(
        os.path.join(dest_dir, "test.csv"),
        index=False
    )

    print("\n--- TỔNG DỮ LIỆU ---")
    print("Tổng số ảnh ban đầu:", total_original)
    print("Tổng số ảnh bộ Train:", total_train)
    print("Tổng số ảnh bộ Test:", total_test)

def get_class_distribution(root_dir):
    distribution = {}

    for split in ["train", "test"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                num_images = len([
                    f for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f))
                ])

                if class_name not in distribution:
                    distribution[class_name] = {}
                distribution[class_name][split] = num_images

    return distribution

def plot_class_distribution(distribution, categories):
    train_counts = [distribution.get(cls, {}).get("train", 0) for cls in categories]
    test_counts = [distribution.get(cls, {}).get("test", 0) for cls in categories]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35

    plt.bar(x - width / 2, train_counts, width, label="Train", color="cyan")
    plt.bar(x + width / 2, test_counts, width, label="Test", color="yellow")

    plt.title("BIỂU ĐỒ PHÂN PHỐI DỮ LIỆU")
    plt.xticks(x, categories, rotation=45, ha="right")
    plt.legend()

    for i, (train, test) in enumerate(zip(train_counts, test_counts)):
        plt.text(i - width / 2, train + 5, str(train), ha="center")
        plt.text(i + width / 2, test + 5, str(test), ha="center")

    plt.tight_layout()
    plt.show()

class AnimalDataset(Dataset):
    def __init__(self, root, train, transform = None):
        self.categories = ["cat", "chicken", "cow", "dog", "horse", "sheep"]

        if train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

        self.image_paths = []
        self.labels = []

        for class_id, categories in enumerate(self.categories):
            sub_folder_path = os.path.join(data_path, categories)
            for image_name in os.listdir(sub_folder_path):
                # print(image_name) # Chỉ lấy đc tên ảnh thì ko đủ để có thể load đc ảnh
                image_path = os.path.join(sub_folder_path, image_name)
                # print(image_path)
                # image = cv2.imread(image_path)
                # print(image.shape)
                # self.images.append(image)
                self.image_paths.append(image_path)
                self.labels.append(class_id)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # image = cv2.imread(self.image_paths[index])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]

        return image, label

if __name__ == '__main__':
    source_dir = "animal10"
    dest_dir = "myAnimalDataset/seen_classes"
    split_dataset(source_dir, dest_dir)
    distribution = get_class_distribution(dest_dir)
    categories = AnimalDataset(root=dest_dir, train=True).categories
    plot_class_distribution(distribution, categories)