from trainResNet50 import *
import random
import logging
from torchvision.transforms import RandomResizedCrop, RandomRotation, RandomAffine, GaussianBlur

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FewShotDataset(Dataset):
    def __init__(self, root_dir, mode='train', n_way=5, k_shot=5, query_num=5, included_classes=None):
        self.root_dir = root_dir
        self.mode = mode
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_num = query_num
        self.included_classes = included_classes
        self.class_pool = self._load_classes()
        self.transform = self._build_transform(mode)

    def _load_classes(self):
        classes = {}
        if self.mode in ['train', 'val']:
            seen_path = os.path.join(self.root_dir, 'seen_classes', 'train')
            for cls in os.listdir(seen_path):
                if self.included_classes and cls not in self.included_classes:
                    continue
                class_dir = os.path.join(seen_path, cls)
                if os.path.isdir(class_dir):
                    imgs = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
                    random.shuffle(imgs)
                    split_index = int(0.8 * len(imgs))
                    if self.mode == 'train':
                        chosen_imgs = imgs[:split_index]
                    else:
                        chosen_imgs = imgs[split_index:]
                    if len(chosen_imgs) < (self.k_shot + self.query_num):
                        logger.warning(f"Lớp {cls} ({self.mode}) có {len(chosen_imgs)} ảnh, không đủ {self.k_shot + self.query_num}, bỏ qua.")
                        continue
                    classes[cls] = chosen_imgs
        elif self.mode == 'test':
            seen_test_path = os.path.join(self.root_dir, 'seen_classes', 'test')
            for cls in os.listdir(seen_test_path):
                if self.included_classes and cls not in self.included_classes:
                    continue
                class_dir = os.path.join(seen_test_path, cls)
                if os.path.isdir(class_dir):
                    imgs = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
                    if len(imgs) < (self.k_shot + self.query_num):
                        logger.warning(f"Lớp {cls} (test) có {len(imgs)} ảnh, không đủ {self.k_shot + self.query_num}, bỏ qua.")
                        continue
                    classes[cls] = imgs
            if not self.included_classes or 'pig' in self.included_classes:
                pig_test_path = os.path.join(self.root_dir, 'unseen_classes', 'pig', 'test')
                pig_imgs = [os.path.join(pig_test_path, img) for img in os.listdir(pig_test_path)]
                if len(pig_imgs) < (self.k_shot + self.query_num):
                    logger.warning(f"Lớp pig (test) có {len(pig_imgs)} ảnh, không đủ {self.k_shot + self.query_num}, bỏ qua.")
                else:
                    classes['pig'] = pig_imgs
        return classes

    def _build_transform(self, mode):
        base_transform = [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if mode == 'train':
            augmentations = [
                RandomResizedCrop(224, scale=(0.7, 1.0)),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                RandomRotation(degrees=15),
                RandomAffine(degrees=0, translate=(0.1, 0.1)),
                GaussianBlur(kernel_size=3)
            ]
            return Compose(augmentations + base_transform)
        return Compose(base_transform)

    def __len__(self):
        if self.mode == 'train':
            return 320
        elif self.mode == 'val':
            return 64
        else:
            return 64

    def __getitem__(self, idx):
        available_classes = list(self.class_pool.keys())
        selected_classes = random.sample(available_classes, self.n_way)
        class_map = {cls: i for i, cls in enumerate(selected_classes)}
        class_to_idx = {cls: idx for idx, cls in enumerate(['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep'])}
        support_tensors = []
        query_tensors = []
        support_labels = []
        query_labels = []
        fixed_labels = []

        for cls in selected_classes:
            all_imgs = self.class_pool[cls]
            support_imgs = random.sample(all_imgs, self.k_shot)
            remaining_imgs = list(set(all_imgs) - set(support_imgs))
            if len(remaining_imgs) < self.query_num:
                raise ValueError(f"Không đủ ảnh cho query trong lớp {cls}")
            query_imgs = random.sample(remaining_imgs, self.query_num)

            for img_path in support_imgs:
                img = Image.open(img_path).convert('RGB')
                support_tensors.append(self.transform(img))
                support_labels.append(class_map[cls])
            for img_path in query_imgs:
                img = Image.open(img_path).convert('RGB')
                query_tensors.append(self.transform(img))
                query_labels.append(class_map[cls])
                fixed_labels.append(class_to_idx[cls])

        return {
            'support': torch.stack(support_tensors),
            'support_labels': torch.tensor(support_labels),
            'query': torch.stack(query_tensors),
            'query_labels': torch.tensor(query_labels),
            'fixed_labels': torch.tensor(fixed_labels),
            'selected_classes': selected_classes
        }

class FewShotModel(nn.Module):
    def __init__(self, pretrained_path, n_way=5, k_shot=5, query_num=5):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_num = query_num
        self.backbone = resnet50(pretrained=False)
        checkpoint = torch.load(pretrained_path)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
        )
        self.backbone.load_state_dict(checkpoint)
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

    def forward(self, support, query, n_way=None, k_shot=None):
        n_way = n_way or self.n_way
        k_shot = k_shot or self.k_shot
        batch_size = support.size(0)
        support = support.view(batch_size * n_way * k_shot, 3, 224, 224)
        query = query.view(batch_size * n_way * self.query_num, 3, 224, 224)
        s_features = self._extract_features(support)
        q_features = self._extract_features(query)
        s_features = s_features.view(batch_size, n_way, k_shot, -1)
        prototypes = s_features.mean(dim=2)
        q_features = q_features.view(batch_size, n_way * self.query_num, -1)
        distances = torch.cdist(q_features, prototypes)
        similarities = -distances
        similarities = similarities.view(batch_size * n_way * self.query_num, n_way)
        return similarities

    def _extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        x = self.projection(x)
        return x

def create_pig_vs_seen_episode(root_dir, seen_class, k_shot=5, query_num=5, transform=None):
    pig_train_path = os.path.join(root_dir, 'unseen_classes', 'pig', 'train')
    pig_train_images = [os.path.join(pig_train_path, img) for img in os.listdir(pig_train_path)]
    random.shuffle(pig_train_images)
    pig_support_images = pig_train_images[:k_shot]
    pig_test_path = os.path.join(root_dir, 'unseen_classes', 'pig', 'test')
    pig_test_images = [os.path.join(pig_test_path, img) for img in os.listdir(pig_test_path)]
    random.shuffle(pig_test_images)
    pig_query_images = pig_test_images[:query_num]

    seen_class_test_path = os.path.join(root_dir, 'seen_classes', 'test', seen_class)
    seen_class_test_images = [os.path.join(seen_class_test_path, img) for img in os.listdir(seen_class_test_path)]
    random.shuffle(seen_class_test_images)
    seen_support_images = seen_class_test_images[:k_shot]
    seen_query_images = seen_class_test_images[k_shot:k_shot + query_num]

    support_images = []
    support_labels = []
    for img_path in pig_support_images + seen_support_images:
        img = Image.open(img_path).convert('RGB')
        support_images.append(transform(img))
        support_labels.append(0 if img_path in pig_support_images else 1)
    query_images = []
    query_labels = []
    for img_path in pig_query_images + seen_query_images:
        img = Image.open(img_path).convert('RGB')
        query_images.append(transform(img))
        query_labels.append(0 if img_path in pig_query_images else 1)

    return {
        'support': torch.stack(support_images),
        'support_labels': torch.tensor(support_labels),
        'query': torch.stack(query_images),
        'query_labels': torch.tensor(query_labels)
    }