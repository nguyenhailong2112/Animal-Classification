import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from myDataset import *
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Resize, CenterCrop, Compose, Normalize, RandomResizedCrop, RandomHorizontalFlip,ColorJitter
from torch.optim import SGD
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Animal Classification")

    parser.add_argument("--data-path", "-d", type=str, default="myAnimalDataset/seen_classes", help="path to dataset")
    parser.add_argument("--log-path", "-o", type=str, default="pre_tensorboard", help="Tensorboard log directory")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="pre_models", help="Model checkpoint directory")
    parser.add_argument("--image-size", "-i", type=int, default=256, help="Input image size")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning-rate", "-l", type=float, default=0.001, help="Learning rate of optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--resume", "-r", type=str, help="Path to checkpoint to resume training")

    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="hsv")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    writer.add_figure("confusion_matrix", figure, epoch)

def train(args):
    train_transform = Compose([
        Resize((args.image_size, args.image_size)),
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = Compose([
        Resize((args.image_size, args.image_size)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = AnimalDataset(root=args.data_path, train=True, transform=train_transform)
    test_dataset = AnimalDataset(root=args.data_path, train=False, transform=val_transform)

    num_classes = len(full_dataset.categories)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_labels = [full_dataset[i][1] for i in train_dataset.indices]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    del model.fc
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )

    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    model = model.to(device)

    optimizer = SGD([
        {'params': model.fc.parameters(), 'lr': args.learning_rate},
        {'params': model.layer4.parameters(), 'lr': args.learning_rate / 10}
    ],
        momentum=args.momentum
    )

    writer = SummaryWriter(log_dir=args.log_path)

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        # Training
        model.train()
        epoch_loss = []
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", colour="cyan")

        for batch_idx, (images, labels) in enumerate(progress_bar_train):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + batch_idx)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = []
        progress_bar_val = tqdm(val_loader, desc="Validation", unit="batch", colour="yellow")
        with torch.no_grad():
            for images, labels in progress_bar_val:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.to(device))
                val_loss.append(loss.item())

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = np.mean(val_loss)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_precision = precision_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')

        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/Precision", val_precision, epoch)
        writer.add_scalar("Val/Recall", val_recall, epoch)
        plot_confusion_matrix(writer, confusion_matrix(val_labels, val_preds), full_dataset.categories, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "best_model.pth"))

        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, f"checkpoint_epoch{epoch + 1}.pth"))

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {np.mean(epoch_loss):.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    pre_model = resnet50()
    pre_model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    pre_model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "best_model.pth")))
    pre_model = pre_model.to(device)
    pre_model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = pre_model(images)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"F1: {f1_score(test_labels, test_preds, average='macro'):.4f}")
    print(f"Precision: {precision_score(test_labels, test_preds, average='macro'):.4f}")
    print(f"Recall: {recall_score(test_labels, test_preds, average='macro'):.4f}")

if __name__ == '__main__':
    args = get_args()
    train(args)