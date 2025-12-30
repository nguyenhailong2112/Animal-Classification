import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import argparse

from myModel import CNN
from myDataset import *

def get_args():
    parser = argparse.ArgumentParser(description="Test ResNet50 Model")

    parser.add_argument("--data-path", "-d", type=str, default="myAnimalDataset/seen_classes", help="Path to dataset")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="pre_models/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--image-size", "-i", type=int, default=256, help="Input image size for resize")

    args = parser.parse_args()
    return args

def prediction_plot(image, true_label, pred_label, confidence, all_classes, all_probs):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image", fontsize=11)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    text = f"True Label: {true_label}\n"
    text += f"Predicted: {pred_label} ({confidence:.4f}%)\n"
    text += "Top Predictions:\n"

    y_pos = np.arange(len(all_classes))
    plt.barh(y_pos, all_probs, height=0.5, align='center', color='cyan')
    plt.yticks(y_pos, all_classes, fontsize=10)
    plt.xticks(fontsize=10)
    plt.title("BIỂU ĐỒ DỰ ĐOÁN", fontsize=11)

    for i, v in enumerate(all_probs):
        plt.text(v + 0.01, i, f"{v * 100:.4f}%", color='black', fontsize=9)

    plt.tight_layout()
    plt.show()

def test(image_path, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    file_to_label = dict(zip(test_df['filename'], test_df['label']))

    filename = os.path.basename(image_path)
    true_label = file_to_label[filename]
    class_dir = os.path.join(args.data_path, "test", true_label)
    full_imagepath = os.path.join(class_dir, filename)

    train_dataset = AnimalDataset(root=args.data_path, train=True)
    categories = train_dataset.categories
    num_classes = len(categories)

    # model = CNN(num_classes).to(device)

    model = resnet50()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = Compose([
        Resize((args.image_size, args.image_size)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    original_image = Image.open(full_imagepath).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_prob, pred_class = torch.max(probs, 1)

    top_pred = num_classes
    top_probs, top_indices = torch.topk(probs, top_pred)
    classes = [categories[i] for i in top_indices[0].cpu().numpy()]
    probabilities = top_probs[0].cpu().numpy()

    display_image = original_image.resize((224, 224))
    pred_label = categories[pred_class.item()]

    prediction_plot(
        image=display_image,
        true_label=true_label,
        pred_label=pred_label,
        confidence=pred_prob.item()*100,
        all_classes=classes,
        all_probs=probabilities
    )

    print(f"\n{'KẾT QUẢ DỰ ĐOÁN'}")
    print(f"File ảnh: {filename}")
    print(f"Dự đoán: {pred_label} ({pred_prob.item()*100:.4f}%)")
    print(f"Thực tế: {true_label}")
    print("\nĐiểm dự đoán:")
    for i, (cls, prob) in enumerate(zip(classes, probabilities)):
        print(f"{i + 1}. {cls}: {prob * 100:.4f}%")

if __name__ == '__main__':
    args = get_args()
    test("OIP-vrx7NVyGBBUIFF0l3FVX0gHaFj.jpeg", args)