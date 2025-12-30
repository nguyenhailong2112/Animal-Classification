from trainFewShot import *

def evaluate(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path)
    args = checkpoint['args']
    model = FewShotModel(args['pretrained_path'], n_way=5, k_shot=5, query_num=5).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Đánh giá trên seen classes
    test_dataset_seen = FewShotDataset(root_dir=args['root_dir'], mode='test', n_way=5, k_shot=5, query_num=5,
                                       included_classes=['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep'])
    test_loader_seen = DataLoader(test_dataset_seen, batch_size=1, pin_memory=True, collate_fn=lambda x: x)
    all_seen_labels = []
    all_seen_preds = []
    with torch.no_grad():
        for batch in test_loader_seen:
            support = torch.stack([ep['support'] for ep in batch]).to(device)  # Shape: [1, 25, 3, 224, 224]
            query = torch.stack([ep['query'] for ep in batch]).to(device)      # Shape: [1, 25, 3, 224, 224]
            labels = torch.cat([ep['query_labels'] for ep in batch]).to(device)
            similarities = model(support, query, n_way=5, k_shot=5)
            preds = torch.argmax(similarities, dim=1)
            all_seen_labels.extend(labels.cpu().numpy())
            all_seen_preds.extend(preds.cpu().numpy())

    seen_accuracy = accuracy_score(all_seen_labels, all_seen_preds)
    seen_precision = precision_score(all_seen_labels, all_seen_preds, average='macro')
    seen_recall = recall_score(all_seen_labels, all_seen_preds, average='macro')
    seen_f1 = f1_score(all_seen_labels, all_seen_preds, average='macro')

    # Đánh giá trên pig vs seen classes
    dataset = FewShotDataset(root_dir=args['root_dir'], mode='test')
    transform = dataset.transform
    seen_classes = ['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']
    all_pig_labels = []
    all_pig_preds = []
    for seen_class in seen_classes:
        episode = create_pig_vs_seen_episode(args['root_dir'], seen_class, k_shot=2, query_num=5, transform=transform)
        support = episode['support'].unsqueeze(0).to(device)  # Thêm chiều batch: [1, 4, 3, 224, 224]
        query = episode['query'].unsqueeze(0).to(device)      # Thêm chiều batch: [1, 10, 3, 224, 224]
        labels = episode['query_labels'].to(device)
        similarities = model(support, query, n_way=2, k_shot=2)
        preds = torch.argmax(similarities, dim=1)
        all_pig_labels.extend(labels.cpu().numpy())
        all_pig_preds.extend(preds.cpu().numpy())

    pig_accuracy = accuracy_score(all_pig_labels, all_pig_preds)
    pig_precision = precision_score(all_pig_labels, all_pig_preds, average='macro')
    pig_recall = recall_score(all_pig_labels, all_pig_preds, average='macro')
    pig_f1 = f1_score(all_pig_labels, all_pig_preds, average='macro')
    cm = confusion_matrix(all_pig_labels, all_pig_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix - Pig vs Seen Classes')
    plt.savefig('pig_confusion_matrix_after_train.png', dpi=300)
    plt.close()

    print(
        f"Seen Classes - Accuracy: {seen_accuracy:.4f}, Precision: {seen_precision:.4f}, Recall: {seen_recall:.4f}, F1: {seen_f1:.4f}")
    print(
        f"Pig vs Seen - Accuracy: {pig_accuracy:.4f}, Precision: {pig_precision:.4f}, Recall: {pig_recall:.4f}, F1: {pig_f1:.4f}")


if __name__ == '__main__':
    evaluate("fewshot_checkpoints/best_model.pth")