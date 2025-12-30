import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from trainOneShot import *

def evaluate_one_shot(model, root_dir, device, transform):
    model.eval()
    butterfly_train_path = os.path.join(root_dir, 'unseen_classes', 'butterfly', 'train', 'butterfly_train.jpg')
    butterfly_train_img = Image.open(butterfly_train_path).convert('RGB')
    butterfly_train_tensor = transform(butterfly_train_img).unsqueeze(0).to(device)

    test_classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']
    test_data = []
    for cls in test_classes:
        if cls == 'butterfly':
            cls_dir = os.path.join(root_dir, 'unseen_classes', 'butterfly', 'test')
        else:
            cls_dir = os.path.join(root_dir, 'seen_classes', 'test', cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            test_data.append((img_tensor, cls, img_path))

    distances = []
    labels = []
    with torch.no_grad():
        for img, true_cls, _ in test_data:
            distance = model(butterfly_train_tensor, img)
            distances.append(distance.cpu().numpy()[0])
            labels.append(1 if true_cls == 'butterfly' else 0)

    distances = np.array(distances)
    labels = np.array(labels)

    # 1. Normalize Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(distances[labels == 1], bins=20, alpha=0.5, density=True, label='Butterfly', color='blue')
    plt.hist(distances[labels == 0], bins=20, alpha=0.5, density=True, label='Others', color='red')
    plt.axvline(0.3711, color='black', linestyle='--', label='Threshold (0.3711)')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Normalized Distance Distribution for Butterfly vs Others')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'normalized_distance_distribution.png'))
    plt.show()

    # 2. ROC Curve và Tìm Ngưỡng Tối Ưu
    fpr, tpr, roc_thresholds = roc_curve(labels, -distances)  # Đảo dấu vì nhỏ hơn là butterfly
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Butterfly vs Others')
    plt.legend(loc='lower right')
    plt.show()

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -roc_thresholds[optimal_idx]
    print(f'Ngưỡng tối ưu: {optimal_threshold:.4f}')

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, -distances)
    avg_precision = average_precision_score(labels, -distances)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Butterfly vs Others')
    plt.legend(loc='lower left')
    plt.show()

    # 4. Confusion Matrix với Ngưỡng Tối Ưu
    preds = (distances < optimal_threshold).astype(np.int32)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Others', 'Butterfly'], yticklabels=['Others', 'Butterfly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Butterfly vs Others')
    plt.show()

    accuracy = np.mean(preds == labels)
    print(f'Accuracy với ngưỡng tối ưu {optimal_threshold:.4f}: {accuracy:.4f}')

    # 5. Visualize Ảnh Bị Dự Đoán Sai
    misclassified_indices = np.where(preds != labels)[0]
    if len(misclassified_indices) > 0:
        print(f"Số lượng ảnh bị dự đoán sai: {len(misclassified_indices)}")
        for idx in misclassified_indices[:5]:
            img_path = test_data[idx][2]
            true_label = 'Butterfly' if labels[idx] == 1 else 'Others'
            pred_label = 'Butterfly' if preds[idx] == 1 else 'Others'
            plt.figure(figsize=(4, 4))
            plt.imshow(Image.open(img_path))
            plt.title(f'True: {true_label}, Predicted: {pred_label}')
            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    root_directory = 'myAnimalDataset'
    pretrained_path = 'pre_models/best_model.pth'
    checkpoint_directory = 'siamese_checkpoints'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = SiameseNetwork(pretrained_path=pretrained_path).to(device)
    checkpoint = torch.load(os.path.join(checkpoint_directory, "siamese_best.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluate_one_shot(model, root_directory, device, transform)

# Ngưỡng tối ưu: 0.3711
# Accuracy với ngưỡng tối ưu 0.3711: 0.9384
# Số lượng ảnh bị dự đoán sai: 226