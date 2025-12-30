from trainFewShot import *
from torchvision import transforms
from sklearn.manifold import TSNE

def get_args():
    parser = argparse.ArgumentParser(description="Few-Shot Learning Testing")
    parser.add_argument("--root_dir", default="myAnimalDataset")
    parser.add_argument("--pretrained_path", default="pre_models/best_model.pth", help="Path to pretrained model")
    parser.add_argument("--checkpoint_path", default="fewshot_checkpoints/best_model.pth", help="Path to trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()

class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = ['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep', 'pig']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data = []
        for cls in self.classes:
            if cls == 'pig':
                path = os.path.join(root_dir, 'unseen_classes', 'pig', 'test')
            else:
                path = os.path.join(root_dir, 'seen_classes', 'test', cls)
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                self.data.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        return img, label

def load_model(pretrained_path, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    model = FewShotModel(pretrained_path=pretrained_path,
                         n_way=args['n_way'],
                         k_shot=args['k_shot'],
                         query_num=args['query_num']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def evaluate_seen_classes(model, root_dir, batch_size, device, num_runs=20):
    seen_classes = ['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']
    test_dataset = FewShotDataset(root_dir=root_dir, mode='test', n_way=5, k_shot=5, query_num=5,
                                  included_classes=seen_classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                             collate_fn=lambda x: x)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    writer = SummaryWriter(log_dir='fewshot_logs/test')

    for run in tqdm(range(num_runs), desc="Evaluating seen classes", colour="green"):
        all_labels = []
        all_preds = []
        class_to_idx = {cls: idx for idx, cls in enumerate(seen_classes)}

        model.eval()
        with torch.no_grad():
            support_tensors = []
            support_labels = []
            for cls in seen_classes:
                all_imgs = test_dataset.class_pool[cls]
                support_imgs = random.sample(all_imgs, 5)
                for img_path in support_imgs:
                    img = Image.open(img_path).convert('RGB')
                    support_tensors.append(test_dataset.transform(img))
                    support_labels.append(class_to_idx[cls])
            support = torch.stack(support_tensors).to(device)
            support_labels = torch.tensor(support_labels).to(device)

            support_features = model._extract_features(support)
            prototypes = []
            for i in range(len(seen_classes)):
                class_mask = (support_labels == i)
                class_features = support_features[class_mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)

            for batch in test_loader:
                query = torch.stack([ep['query'] for ep in batch]).to(device)
                fixed_labels = torch.cat([ep['fixed_labels'] for ep in batch]).to(device)
                query_features = model._extract_features(query.view(-1, *query.shape[2:]))

                dists = torch.cdist(query_features, prototypes)
                similarities = -dists
                preds = torch.argmax(similarities, dim=1)

                all_labels.extend(fixed_labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    writer.add_scalar('Test/Seen_Accuracy', avg_accuracy)
    writer.add_scalar('Test/Seen_Precision', avg_precision)
    writer.add_scalar('Test/Seen_Recall', avg_recall)
    writer.add_scalar('Test/Seen_F1', avg_f1)

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(seen_classes)))
    print(
        f"Seen Classes - Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=seen_classes, yticklabels=seen_classes)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix - Seen Classes')
    plt.savefig('seen_confusion_matrix.png', dpi=300)
    plt.close()

    writer.close()

def evaluate_pig_vs_seen(model, root_dir, device, num_runs=10):
    dataset = FewShotDataset(root_dir=root_dir, mode='test')
    transform = dataset.transform
    seen_classes = ['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    writer = SummaryWriter(log_dir='fewshot_logs/test')

    for run in tqdm(range(num_runs), desc="Evaluating pig vs seen", colour="green"):
        all_pig_labels = []
        all_pig_preds = []
        for seen_class in seen_classes:
            episode = create_pig_vs_seen_episode(root_dir, seen_class, k_shot=5, query_num=5, transform=transform)
            support = episode['support'].unsqueeze(0).to(device)
            query = episode['query'].unsqueeze(0).to(device)
            labels = episode['query_labels'].to(device)
            similarities = model(support, query, n_way=2, k_shot=5)
            preds = torch.argmax(similarities, dim=1)
            all_pig_labels.extend(labels.cpu().numpy())
            all_pig_preds.extend(preds.cpu().numpy())
        accuracy = accuracy_score(all_pig_labels, all_pig_preds)
        precision = precision_score(all_pig_labels, all_pig_preds, average='macro')
        recall = recall_score(all_pig_labels, all_pig_preds, average='macro')
        f1 = f1_score(all_pig_labels, all_pig_preds, average='macro')
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    writer.add_scalar('Test/Pig_vs_Seen_Accuracy', avg_accuracy)
    writer.add_scalar('Test/Pig_vs_Seen_Precision', avg_precision)
    writer.add_scalar('Test/Pig_vs_Seen_Recall', avg_recall)
    writer.add_scalar('Test/Pig_vs_Seen_F1', avg_f1)

    cm = confusion_matrix(all_pig_labels, all_pig_preds, labels=[0, 1])
    print(f"Pig vs Seen - Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['pig', 'seen'], yticklabels=['pig', 'seen'])
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix - Pig vs Seen Classes')
    plt.savefig('pig_confusion_matrix.png', dpi=300)
    plt.close()

    writer.close()

def extract_features(model, dataset, device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feat = model._extract_features(imgs)
            features.append(feat.cpu().numpy())
            labels.extend(lbls.numpy())
    features = np.concatenate(features, axis=0)
    return features, labels

def plot_tsne(features, labels, classes):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.6)
    cbar = plt.colorbar(scatter, ticks=range(len(classes)))
    cbar.ax.set_yticklabels(classes)
    plt.title('t-SNE Feature Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('tsne_FewshotLearning.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.pretrained_path, args.checkpoint_path, device)

    evaluate_seen_classes(model, args.root_dir, args.batch_size, device)

    evaluate_pig_vs_seen(model, args.root_dir, device)

    test_dataset_all = TestDataset(args.root_dir)
    features, labels = extract_features(model, test_dataset_all, device)
    classes = ['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep', 'pig']
    plot_tsne(features, labels, classes)